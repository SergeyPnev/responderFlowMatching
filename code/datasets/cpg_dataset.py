"""
CPG single-cell imaging dataset.

Content = morphological response to therapeutic treatment (``CPD_NAME``).
Style   = batch / plate / illumination artefacts (``PLATE``).

File-path pattern::

    {img_dir}/{PLATE}/{WELL}/{WELL}_{rest}.npy

derived from ``SAMPLE_KEY = "{PLATE}_{WELL}_{site}_{idx}"``.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ._common import BaseDisentangleDataset, LabelEncoder


import os
import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict

# Must match the caption function used when the text embeddings were extracted.
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


class CPGDataset(BaseDisentangleDataset):
    def __init__(
        self,
        img_dir: str,
        fold_path: str,
        transform=None,
        return_moa: bool = False,
        moa_col: str = "PERT_TYPE",
        treatment_col: str = "CPD_NAME",
        plate_col: str = "PLATE",
        encoders: Optional[Dict[str, LabelEncoder]] = None,
        text_embeddings_path: str = None,        # CellCLIP text embeddings (.pt)
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
            experiment_metadata = pd.read_csv(img_dir + "/metadata/experiment-metadata.tsv", sep="\t")
            plate_to_celltype = (experiment_metadata.groupby('Assay_Plate_Barcode')['Cell_type']
                       .agg(lambda x: x.mode()[0] if len(x.mode()) == 1 else x.iloc[0])
                       .to_dict())
            self.text_embeddings = self._load_text_embeddings(
                text_embeddings_path, text_embedding_key, plate_to_celltype,
            )

    # ------------------------------------------------------------------ #
    def _load_text_embeddings(self, path, key, plate_to_celltype):
        blob = torch.load(path, map_location="cpu")
        emb_table = blob[key]                                    # [N_unique, D]
        prompt_to_idx = {p: i for i, p in enumerate(blob["prompts"])}

        # Attach CELL_TYPE using the same mapping used at extraction
        if "CELL_TYPE" not in self.df.columns:
            assert plate_to_celltype is not None, (
                "Dataset has no CELL_TYPE column — pass plate_to_celltype so "
                "prompts match what was encoded."
            )
            self.df["CELL_TYPE"] = self.df[self.plate_col].map(plate_to_celltype)

        ids = self.df["SAMPLE_KEY"].values
        self.df["perturbation"] = self.df["PERT_TYPE"].str.lower()
        # Rebuild prompts identically to extraction
        prompts = [
            generate_cell_caption(ct, pert, cpd, smi)
            for ct, pert, cpd, smi in zip(
                self.df["CELL_TYPE"],
                self.df["perturbation"],
                self.df[self.treatment_col],
                self.df["SEQUENCE"],
            )
        ]

        # Gather per-row embeddings
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
        paths = []
        for key in self.df["SAMPLE_KEY"].values:
            parts = key.split("_")
            plate = parts[0]
            well = f"{parts[1]}_{parts[2]}"
            fname = "_".join(parts[1:]) + ".npy"
            paths.append(os.path.join(self.img_dir, plate, well, fname))
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
# Default transforms
# ====================================================================== #

def default_cpg_transform(img_size: int = 96, n_channels: int = 5):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize([0] * n_channels, [1] * n_channels),
        ToTensorV2(),
    ])


# ====================================================================== #
# Public builder called by the factory
# ====================================================================== #

def build_cpg(args, only_test=False, _all=False, simulate_control=True):
    """
    Return ``(train_set, val_set, test_set)`` or a subset, depending on flags.
    Also patches ``args.data_type``, ``args.out_size``, ``args.data_size``.
    """
    transform = default_cpg_transform(args.img_size, n_channels=5)

    train_fold_path = os.path.join(
        args.fold_path, f"ds_{args.dataset}_train_fold{args.fold}.csv"
    )
    val_fold_path = os.path.join(
        args.fold_path, f"ds_{args.dataset}_val_fold{args.fold}.csv"
    )

    treatment_col = args.treatment_col
    plate_col = args.plate_col
    train_df = pd.read_csv(train_fold_path)
    val_df = pd.read_csv(val_fold_path)
    all_plates = pd.concat([train_df[plate_col], val_df[plate_col]])
    shared_plate_encoder = LabelEncoder().fit(all_plates)

    encoders = {
        "treatment": LabelEncoder().fit(train_df[treatment_col]),
        "plate":     shared_plate_encoder,
    }

    train_set = CPGDataset(
        img_dir=args.img_dir,
        fold_path=train_fold_path,
        text_embeddings_path=args.text_emb_path,
        transform=transform,
        return_moa=getattr(args, "return_moa", False),
        moa_col=getattr(args, "moa_col", "PERT_TYPE"),
        treatment_col=getattr(args, "treatment_col", "CPD_NAME"),
        plate_col=getattr(args, "plate_col", "PLATE"),
        encoders=encoders,
    )
    val_set = CPGDataset(
        img_dir=args.img_dir,
        fold_path=val_fold_path,
        text_embeddings_path=args.text_emb_path,
        transform=transform,
        return_moa=getattr(args, "return_moa", False),
        moa_col=getattr(args, "moa_col", "PERT_TYPE"),
        treatment_col=getattr(args, "treatment_col", "CPD_NAME"),
        plate_col=getattr(args, "plate_col", "PLATE"),
        encoders=encoders,
    )
    # test_set reuses the val fold
    test_set = CPGDataset(
        img_dir=args.img_dir,
        fold_path=val_fold_path,
        text_embeddings_path=args.text_emb_path,
        transform=transform,
        return_moa=getattr(args, "return_moa", False),
        moa_col=getattr(args, "moa_col", "PERT_TYPE"),
        treatment_col=getattr(args, "treatment_col", "CPD_NAME"),
        plate_col=getattr(args, "plate_col", "PLATE"),
        encoders=encoders,
    )

    # Patch args with data metadata
    args.data_type = "img"
    args.out_size = 5
    args.data_size = (args.out_size, args.img_size, args.img_size)

    if only_test:
        return test_set
    if _all:
        return train_set, val_set, test_set
    return train_set, test_set