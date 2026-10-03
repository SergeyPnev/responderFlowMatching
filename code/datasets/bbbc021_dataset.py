"""
BBBC021 (MCF-7) single-cell imaging dataset.

Content = morphological response to therapeutic treatment (``CPD_NAME``).
Style   = batch / plate / illumination artefacts (``PLATE``).
MOA     = mechanism-of-action label, lives in the ``ANNOT`` column.

File-path pattern::

    {img_dir}/{WEEK}/{PLATE}/{SAMPLE_KEY}.npy

derived from ``SAMPLE_KEY = "{WEEK}_{PLATE}_{TABLE_NUMBER}_{IMAGE_NUMBER}_{OBJECT_NUMBER}"``.

``exclude_compounds`` drops OOD compounds before the label encoders are fit.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ._common import BaseDisentangleDataset, LabelEncoder


# Must match the caption function used when the text embeddings were extracted
# (identical to the CPG one).
def generate_cell_caption(cell_type, perturbation_type, target, target_info=None):
    """Generate a caption for a cell-painting image (compound perturbations)."""
    prompt = (
        f"{cell_type} cells treated with {perturbation_type}:"
        f" {target}, SMILES: {target_info}"
    )

    if "DMSO" in str(target):
        prompt = (
            f"{cell_type} cells treated with {perturbation_type}:"
            f" control, {target}, SMILES: {target_info}"
        )
    elif "control" in str(target).lower():
        prompt = f"{cell_type} cells treated with {perturbation_type}: control"

    if len(prompt) > 512:
        prompt = f"{cell_type} cells treated with SMILES: {target_info}"
        if len(prompt) > 512:
            prompt = f"{cell_type} cells treated with {perturbation_type}, {target}"

    return prompt


class BBBC021Dataset(BaseDisentangleDataset):
    CELL_TYPE = "MCF-7"
    PERTURBATION_TYPE = "Compound"

    def __init__(
        self,
        img_dir: str,
        fold_path: str,
        transform=None,
        return_moa: bool = True,
        moa_col: str = "ANNOT",
        treatment_col: str = "CPD_NAME",
        plate_col: str = "PLATE",
        smiles_col: str = "SMILES",
        exclude_compounds: Optional[List[str]] = None,
        encoders: Optional[Dict[str, LabelEncoder]] = None,
        text_embeddings_path: Optional[str] = None,
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
        self.smiles_col = smiles_col

        if "Unnamed: 0" in self.df.columns:
            self.df.drop(columns=["Unnamed: 0"], inplace=True)

        # ---- OOD compound filtering (must precede encoder fit) ----
        if exclude_compounds:
            before = len(self.df)
            self.df = (
                self.df[~self.df[treatment_col].isin(exclude_compounds)]
                .reset_index(drop=True)
            )
            print(
                f"[BBBC021] excluded {len(exclude_compounds)} OOD compound(s): "
                f"{before} -> {len(self.df)} rows"
            )

        if len(self.df) == 0:
            raise ValueError(
                f"BBBC021Dataset: 0 rows left after filtering {fold_path}. "
                f"Check exclude_compounds."
            )

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
        emb_table = blob[key]                                    # [N_unique, D]
        prompt_to_idx = {p: i for i, p in enumerate(blob["prompts"])}

        # BBBC021 cell type & perturbation type are constant
        if "CELL_TYPE" not in self.df.columns:
            self.df["CELL_TYPE"] = self.CELL_TYPE
        self.df["perturbation"] = self.df["PERT_TYPE"]

        # Rebuild prompts identically to extraction
        prompts = [
            generate_cell_caption(ct, pert, cpd, smi)
            for ct, pert, cpd, smi in zip(
                self.df["CELL_TYPE"],
                self.df["perturbation"],
                self.df[self.treatment_col],
                self.df[self.smiles_col],
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
        return emb_table[torch.from_numpy(idxs)].contiguous()    # [len(df), D]

    # ------------------------------------------------------------------ #
    def _build_paths(self):
        """``SAMPLE_KEY = '{week}_{plate}_{table}_{image}_{object}'`` ->
        ``{img_dir}/{week}/{plate}/{table}_{image}_{object}.npy``."""
        paths = []
        for key in self.df["SAMPLE_KEY"].values:
            parts = key.split("_")
            week = parts[0]                       # e.g. "Week1"
            plate = parts[1]                      # e.g. "22123"
            key = "_".join(key.split("_")[2:])
            fname = f"{key}.npy"
            paths.append(os.path.join(self.img_dir, week, plate, fname))
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
            labels["text_emb"] = self.text_embeddings[idx][0]    # [D] float tensor
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
# Default transforms — BBBC021 has 3 channels
# ====================================================================== #

def default_bbbc_transform(img_size: int = 96, n_channels: int = 3):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize([0] * n_channels, [1] * n_channels),
        ToTensorV2(),
    ])


# ====================================================================== #
# Public builder called by the factory
# ====================================================================== #

def build_bbbc(args, only_test=False, _all=False, simulate_control=True):
    """
    Return ``(train_set, val_set, test_set)`` or a subset, depending on flags.
    Also patches ``args.data_type``, ``args.out_size``, ``args.data_size``.

    ``args.exclude_compounds`` (optional) is dropped from the train and val
    folds before any encoder is fit.
    """
    n_channels = getattr(args, "n_channels", 3)
    transform = default_bbbc_transform(args.img_size, n_channels=n_channels)

    train_fold_path = os.path.join(
    args.fold_path, f"ds_{args.dataset}_train_iid.csv"
    )
    val_fold_path = os.path.join(
    args.fold_path, f"ds_{args.dataset}_val_iid.csv"
    )
    test_fold_path = os.path.join(
        args.fold_path, f"ds_{args.dataset}_test_ood_dmso.csv"
    )

    treatment_col = getattr(args, "treatment_col", "CPD_NAME")
    plate_col     = getattr(args, "plate_col",     "PLATE")
    moa_col       = getattr(args, "moa_col",       "ANNOT")
    smiles_col    = getattr(args, "smiles_col",    "SMILES")
    return_moa    = getattr(args, "return_moa",    True)
    exclude_compounds = list(getattr(args, "exclude_compounds", None) or [])

    # ---- Fit shared encoders across the (filtered) train + val pool ----
    train_df = pd.read_csv(train_fold_path)
    val_df   = pd.read_csv(val_fold_path)
    test_df = pd.read_csv(test_fold_path)

    print("\n\n test df shape: ", test_df.shape, "\n\n")

    if exclude_compounds:
        train_df = train_df[~train_df[treatment_col].isin(exclude_compounds)]
        val_df   = val_df  [~val_df  [treatment_col].isin(exclude_compounds)]
        print(
            f"[build_bbbc] excluded {len(exclude_compounds)} OOD compound(s); "
            f"train={len(train_df)} val={len(val_df)}"
        )

    all_plates = pd.concat([train_df[plate_col], val_df[plate_col], test_df[plate_col]])
    shared_plate_encoder = LabelEncoder().fit(all_plates)

    encoders = {
        "treatment": LabelEncoder().fit(train_df[treatment_col]),
        "plate":     shared_plate_encoder,
    }
    if return_moa:
        # Fit MOA on the union so val MOAs are always encodable.
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
        smiles_col=smiles_col,
        exclude_compounds=exclude_compounds,
        encoders=encoders,
    )

    train_set = BBBC021Dataset(fold_path=train_fold_path, **common)
    val_set   = BBBC021Dataset(fold_path=val_fold_path,   **common)
    test_set  = BBBC021Dataset(fold_path=test_fold_path,   **common)

    # Patch args with data metadata
    args.data_type = "img"
    args.out_size  = n_channels
    args.data_size = (args.out_size, args.img_size, args.img_size)

    if only_test:
        return test_set
    if _all:
        return train_set, val_set, test_set
    return train_set, test_set