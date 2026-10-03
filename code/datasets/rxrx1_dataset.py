"""
RxRx1 single-cell imaging dataset (genetic / siRNA perturbations), 6 channels.

Content = morphological response to an siRNA gene knockdown (``CPD_NAME``).
Style   = experiment / batch artefacts (``BATCH``).

The perturbation is identified by the knocked-down gene symbol (no SMILES).
Following CellFlux, the domain label is the experiment ``BATCH`` (e.g.
``U2OS-01``), not the plate, so ``plate_col`` defaults to ``"BATCH"``.

File-path pattern::

    {img_dir}/{exp}_{plate}/{well}/{site}_{idx}.npy

derived from ``SAMPLE_KEY = "U2OS-{exp}_{plate}_{well}_{site}_{idx}"``,
e.g. ``U2OS-01_1_B02_s1_14`` -> ``{img_dir}/01_1/B02/s1_14.npy``.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ._common import BaseDisentangleDataset, LabelEncoder


# Must match the caption function used when the text embeddings were extracted.
def generate_gene_caption(cell_type, perturbation_type, gene, annot=None):
    """Caption for an RxRx1 image (siRNA gene knockdown)."""
    annot = str(annot).lower()
    prompt = f"{cell_type} cells treated with {perturbation_type}: {gene}"

    if str(gene).upper() == "UNTREATED" or "negative" in annot:
        prompt = f"{cell_type} cells treated with {perturbation_type}: control"
    elif "positive" in annot:
        prompt = f"{cell_type} cells treated with {perturbation_type}: positive control, {gene}"

    if len(prompt) > 512:
        prompt = f"{cell_type} cells treated with {perturbation_type}, {gene}"

    return prompt


class RxRx1Dataset(BaseDisentangleDataset):
    def __init__(
        self,
        img_dir: str,
        fold_path: str,
        transform=None,
        return_moa: bool = False,
        moa_col: str = "ANNOT",
        treatment_col: str = "CPD_NAME",
        plate_col: str = "BATCH",
        encoders: Optional[Dict[str, LabelEncoder]] = None,
        text_embeddings_path: Optional[str] = None,   # CellCLIP text embeddings (.pt)
        text_embedding_key: str = "emb",
    ):
        print("text_embeddings_path: ", text_embeddings_path)
        self.img_dir = img_dir
        self.df = pd.read_csv(fold_path)
        self.transform = transform
        self.return_moa = return_moa
        self.treatment_col = treatment_col
        self.plate_col = plate_col
        self.moa_col = moa_col

        if "Unnamed: 0" in self.df.columns:
            self.df.drop(columns=["Unnamed: 0"], inplace=True)

        self.img_paths = self._build_paths()

        # ---- label encoders ----
        if encoders is not None:
            self.encoders = encoders
        else:
            self.encoders = {
                "treatment": LabelEncoder().fit(self.df[treatment_col]),
                "plate":     LabelEncoder().fit(self.df[plate_col]),
            }
            if return_moa:
                self.encoders["moa"] = LabelEncoder().fit(self.df[moa_col])

        self.treatment_labels = self.encoders["treatment"].transform(self.df[treatment_col])
        self.plate_labels     = self.encoders["plate"].transform(self.df[plate_col])
        if return_moa and "moa" in self.encoders:
            self.moa_labels = self.encoders["moa"].transform(self.df[moa_col])

        # ---- text embedding lookup ----
        self.text_embeddings = None
        if text_embeddings_path is not None:
            self.text_embeddings = self._load_text_embeddings(
                text_embeddings_path, text_embedding_key,
            )

    # ------------------------------------------------------------------ #
    def _load_text_embeddings(self, path, key):
        blob = torch.load(path, map_location="cpu")
        emb_table = blob[key]                                    # [N_unique, 1, D]
        prompt_to_idx = {p: i for i, p in enumerate(blob["prompts"])}

        if "CELL_TYPE" not in self.df.columns:
            self.df["CELL_TYPE"] = self.df["CELL_LINE"]
        self.df["PERT_TYPE"] = "siRNA"
        self.df["perturbation"] = self.df["PERT_TYPE"].str.lower()

        # Rebuild prompts identically to extraction.
        prompts = [
            generate_gene_caption(ct, pert, gene, annot)
            for ct, pert, gene, annot in zip(
                self.df["CELL_TYPE"],
                self.df["perturbation"],
                self.df[self.treatment_col],
                self.df["ANNOT"],
            )
        ]

        missing = [p for p in set(prompts) if p not in prompt_to_idx]
        if missing:
            raise KeyError(
                f"{len(missing)} prompt(s) not in embedding file. "
                f"Example: {missing[0]!r}. Re-run extraction against current df."
            )
        idxs = np.fromiter((prompt_to_idx[p] for p in prompts), dtype=np.int64,
                           count=len(prompts))
        return emb_table[torch.from_numpy(idxs)].contiguous()    # [len(df), 1, D]

    # ------------------------------------------------------------------ #
    def _build_paths(self):
        """SAMPLE_KEY ``U2OS-01_1_B02_s1_14``
          -> exp_plate ``01_1``, well ``B02``, file ``s1_14.npy``
          -> ``{img_dir}/01_1/B02/s1_14.npy``
        """
        paths = []
        for key in self.df["SAMPLE_KEY"].values:
            after_dash = key.split("-", 1)[1]          # "01_1_B02_s1_14"
            parts = after_dash.split("_")              # ["01","1","B02","s1","14"]
            exp_plate = "_".join(parts[:2])            # "01_1"
            well = parts[2]                            # "B02"
            fname = "_".join(parts[3:]) + ".npy"       # "s1_14.npy"
            paths.append(os.path.join(self.img_dir, exp_plate, well, fname))
        return np.array(paths)

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx])
        if self.transform is not None:
            img = self.transform(image=img)["image"]

        labels = {
            "treatment": self.treatment_labels[idx],
            "plate":     self.plate_labels[idx],
            "path":      self.img_paths[idx],
        }
        if self.return_moa and "moa" in self.encoders:
            labels["moa"] = self.moa_labels[idx]
        if self.text_embeddings is not None:
            labels["text_emb"] = self.text_embeddings[idx][0]     # [D] float tensor
        return img, labels

    # ------------------------------------------------------------------ #
    @property
    def num_treatments(self):
        return self.encoders["treatment"].num_classes

    @property
    def num_plates(self):
        return self.encoders["plate"].num_classes

    @property
    def num_moa(self):
        return self.encoders["moa"].num_classes if "moa" in self.encoders else 0


# ====================================================================== #
# Default transforms — RxRx1 has 6 channels
# ====================================================================== #

def default_rxrx1_transform(img_size: int = 96, n_channels: int = 6):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize([0] * n_channels, [1] * n_channels),
        ToTensorV2(),
    ])


# ====================================================================== #
# Public builder called by the factory / VAE trainer
# ====================================================================== #

def build_rxrx1(args, only_test=False, _all=False):
    """
    Return ``(train_set, val_set, test_set)`` or a subset, depending on flags.
    Also patches ``args.data_type``, ``args.out_size``, ``args.data_size``.
    """
    n_channels = getattr(args, "n_channels", 6)
    transform = default_rxrx1_transform(args.img_size, n_channels=n_channels)

    train_fold_path = os.path.join(
        args.fold_path, f"ds_{args.dataset}_train_fold{args.fold}.csv"
    )
    val_fold_path = os.path.join(
        args.fold_path, f"ds_{args.dataset}_val_fold{args.fold}.csv"
    )
    # OOD test fold if present; else reuse val.
    test_fold_path = os.path.join(
        args.fold_path, f"ds_{args.dataset}_test_fold{args.fold}.csv"
    )
    if not os.path.exists(test_fold_path):
        test_fold_path = val_fold_path

    treatment_col = getattr(args, "treatment_col", "CPD_NAME")
    plate_col     = getattr(args, "plate_col",     "BATCH")
    moa_col       = getattr(args, "moa_col",       "ANNOT")
    return_moa    = getattr(args, "return_moa",    False)

    train_df = pd.read_csv(train_fold_path)
    val_df   = pd.read_csv(val_fold_path)
    test_df  = pd.read_csv(test_fold_path)

    all_plates = pd.concat([train_df[plate_col], val_df[plate_col], test_df[plate_col]])
    encoders = {
        "treatment": LabelEncoder().fit(train_df[treatment_col]),
        "plate":     LabelEncoder().fit(all_plates),
    }
    if return_moa:
        all_moa = pd.concat([train_df[moa_col], val_df[moa_col], test_df[moa_col]])
        encoders["moa"] = LabelEncoder().fit(all_moa)

    common = dict(
        img_dir=args.img_dir,
        text_embeddings_path=getattr(args, "text_emb_path", None),
        transform=transform,
        return_moa=return_moa,
        moa_col=moa_col,
        treatment_col=treatment_col,
        plate_col=plate_col,
        encoders=encoders,
    )

    train_set = RxRx1Dataset(fold_path=train_fold_path, **common)
    val_set   = RxRx1Dataset(fold_path=val_fold_path,   **common)
    test_set  = RxRx1Dataset(fold_path=test_fold_path,  **common)

    # Patch args with data metadata
    args.data_type = "img"
    args.out_size  = n_channels
    args.data_size = (args.out_size, args.img_size, args.img_size)

    if only_test:
        return test_set
    if _all:
        return train_set, val_set, test_set
    return train_set, val_set
