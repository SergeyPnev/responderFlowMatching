"""
Per-crop feature extraction for BBBC021 / cpg0000 / RxRx1.

Output: one HDF5 dataset per perturbation key plus sidecar CSVs (prompts,
split, crop index).

    --model dinov2g    crops -> (n_crops, C, 1536); each channel is fed to
                       DINOv2-giant as a grayscale image (global /255)
    --model morphem    crops -> (n_crops, C, 384); MorphEm ViT-S/16, one channel
                       per pass, with the model card's own preprocessing
    --model cellclip   dinov2g h5 -> (n_crops, 512); CellCLIP's
                       CrossChannelFormer over a bag of one crop

    python features/extract_features.py --model dinov2g --dataset bbbc021 \\
        --img_dir IMG_DIR --fold_path FOLD_DIR --output_h5 dinov2g_ind.h5
    python features/extract_features.py --model cellclip --dataset bbbc021 \\
        --from_h5 dinov2g_ind.h5 --output_h5 cellclip_ind.h5 [--channel_map none]
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ===================================================================== #
# Channel identity of the pretrained CrossChannelFormer table
# ===================================================================== #
# CrossChannelFormer adds row i of a learned (5, d) ``channel_embed`` to channel
# i and has no other positional signal, so a channel's identity is its row.
# Row order of the released checkpoint, as traced through CellCLIP's
# preprocessing.
BRAY_SLOTS: Tuple[str, ...] = ("Mito", "ER", "RNA", "AGP", "DNA")
SLOT_OF = {name: i for i, name in enumerate(BRAY_SLOTS)}


# ===================================================================== #
# Prompt templates -- verbatim copies of the loaders' functions, so an
# existing text-embedding cache keyed off these strings stays reusable.
# ===================================================================== #
def generate_cell_caption(cell_type, perturbation_type, target, target_info=None):
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


def generate_gene_caption(cell_type, perturbation_type, gene, annot=None):
    annot = str(annot).lower()
    prompt = f"{cell_type} cells treated with {perturbation_type}: {gene}"
    if str(gene).upper() == "UNTREATED" or "negative" in annot:
        prompt = f"{cell_type} cells treated with {perturbation_type}: control"
    elif "positive" in annot:
        prompt = (
            f"{cell_type} cells treated with {perturbation_type}:"
            f" positive control, {gene}"
        )
    if len(prompt) > 512:
        prompt = f"{cell_type} cells treated with {perturbation_type}, {gene}"
    return prompt


# ===================================================================== #
# Path builders -- one per dataset, matching datasets/*.py
# ===================================================================== #
def _path_bbbc021(img_dir: str, key: str) -> str:
    """``Week1_22123_1_11_3.0`` -> ``{img_dir}/Week1/22123/1_11_3.0.npy``."""
    parts = key.split("_")
    return os.path.join(img_dir, parts[0], parts[1], "_".join(parts[2:]) + ".npy")


def _path_cpg(img_dir: str, key: str) -> str:
    """``{PLATE}_{WELL}_{site}_{idx}`` -> ``{img_dir}/{PLATE}/{w}/{rest}.npy``."""
    parts = key.split("_")
    return os.path.join(
        img_dir, parts[0], f"{parts[1]}_{parts[2]}", "_".join(parts[1:]) + ".npy"
    )


def _path_rxrx1(img_dir: str, key: str) -> str:
    """``U2OS-01_1_B02_s1_14`` -> ``{img_dir}/01_1/B02/s1_14.npy``."""
    parts = key.split("-", 1)[1].split("_")
    return os.path.join(
        img_dir, "_".join(parts[:2]), parts[2], "_".join(parts[3:]) + ".npy"
    )


# ===================================================================== #
# Dataset registry
# ===================================================================== #
@dataclass
class DatasetSpec:
    n_channels: int
    channel_names: Tuple[str, ...]
    # dataset channel index -> pretrained CrossChannelFormer row
    cellclip_slots: Tuple[str, ...]
    path_of: Callable[[str, str], str]
    folds: Tuple[Tuple[str, str], ...]      # (split name, filename template)
    treatment_col: str
    plate_col: str
    moa_col: str
    caption: str                            # "compound" | "gene"
    smiles_col: Optional[str] = None
    cell_type: Optional[str] = None         # constant, or None -> resolved
    pert_type: str = "Compound"
    optional_folds: Tuple[str, ...] = field(default_factory=tuple)
    # alternative fold-file layout; ``resolve_folds`` picks by what is on disk
    alt_folds: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    alt_optional_folds: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def cellclip_rows(self) -> Tuple[int, ...]:
        return tuple(SLOT_OF[s] for s in self.cellclip_slots)

    @property
    def layouts(self) -> Tuple[Tuple[Tuple[Tuple[str, str], ...],
                                     Tuple[str, ...]], ...]:
        out = [(self.folds, self.optional_folds)]
        if self.alt_folds:
            out.append((self.alt_folds, self.alt_optional_folds))
        return tuple(out)


DATASETS: Dict[str, DatasetSpec] = {
    # beta-tubulin has no Cell Painting counterpart; "Mito" below is an
    # arbitrary pick, which is why --channel_map none exists.
    "bbbc021": DatasetSpec(
        n_channels=3,
        channel_names=("DNA", "F-Actin", "beta-Tubulin"),
        cellclip_slots=("DNA", "AGP", "Mito"),
        path_of=_path_bbbc021,
        # the iid/ood partition
        folds=(("train", "ds_bbbc021_train_iid.csv"),
               ("val", "ds_bbbc021_val_iid.csv"),
               ("test", "ds_bbbc021_test_ood_dmso.csv"),
               ("test_ood", "ds_bbbc021_test_ood.csv")),
        treatment_col="CPD_NAME",
        plate_col="PLATE",
        moa_col="ANNOT",
        smiles_col="SMILES",
        caption="compound",
        cell_type="MCF-7",
        pert_type="Compound",
        optional_folds=("test_ood",),
        # alternative layout: one unpartitioned file over every crop
        alt_folds=(("all", "bbbc021_df_all.csv"),),
    ),
    "cpg": DatasetSpec(
        n_channels=5,
        channel_names=("DNA", "RNA", "ER", "AGP", "Mito"),
        cellclip_slots=("DNA", "RNA", "ER", "AGP", "Mito"),
        path_of=_path_cpg,
        folds=(("train", "ds_cpg_train_fold{fold}.csv"),
               ("val", "ds_cpg_val_fold{fold}.csv"),
               ("test", "ds_cpg_test_fold{fold}.csv")),
        treatment_col="CPD_NAME",
        plate_col="PLATE",
        moa_col="PERT_TYPE",
        smiles_col="SEQUENCE",
        caption="compound",
        cell_type=None,                      # from metadata/experiment-metadata.tsv
        optional_folds=("test",),            # build_cpg reuses val as test
    ),
    # golgi shares the AGP row: Cell Painting's AGP channel is actin + Golgi +
    # plasma membrane, so two RxRx1 channels legitimately land on it.
    "rxrx1": DatasetSpec(
        n_channels=6,
        channel_names=("nuclei", "ER", "actin", "nucleoli-RNA", "mito", "golgi"),
        cellclip_slots=("DNA", "ER", "AGP", "RNA", "Mito", "AGP"),
        path_of=_path_rxrx1,
        folds=(("train", "ds_rxrx1_train_fold{fold}.csv"),
               ("val", "ds_rxrx1_val_fold{fold}.csv"),
               ("test", "ds_rxrx1_test_fold{fold}.csv")),
        treatment_col="CPD_NAME",
        plate_col="BATCH",
        moa_col="ANNOT",
        caption="gene",
        pert_type="siRNA",
        optional_folds=("test",),
    ),
}


# ===================================================================== #
# Metadata
# ===================================================================== #
def resolve_folds(spec: DatasetSpec, fold_path: str, fold: int):
    """``(folds, optional_folds)`` for whichever layout ``fold_path`` holds:
    the first one whose non-optional files are all present."""
    layouts = spec.layouts
    for folds, optional in layouts:
        req = [t for sp, t in folds if sp not in optional]
        if all(os.path.exists(os.path.join(fold_path, t.format(fold=fold)))
               for t in req):
            if len(layouts) > 1:
                print(f"[extract] fold layout: {', '.join(t.format(fold=fold) for _, t in folds)}")
            return folds, optional
    looked = "\n".join(
        "    " + " ".join(t.format(fold=fold) for sp, t in folds
                          if sp not in optional)
        for folds, optional in layouts)
    raise FileNotFoundError(
        f"no known fold layout is complete in {fold_path}\n"
        f"  looked for, in order:\n{looked}\n"
        f"  found: {sorted(f for f in os.listdir(fold_path) if f.endswith('.csv'))[:12]}"
        if os.path.isdir(fold_path) else f"{fold_path} is not a directory")


def load_folds(spec: DatasetSpec, fold_path: str, fold: int,
               exclude_compounds: Optional[Sequence[str]]) -> pd.DataFrame:
    """Concatenate every fold CSV, tagging each row with its split."""
    folds, optional_folds = resolve_folds(spec, fold_path, fold)
    frames = []
    for split, template in folds:
        path = os.path.join(fold_path, template.format(fold=fold))
        if not os.path.exists(path):
            if split in optional_folds:
                print(f"[extract] {split} fold absent, skipping: {path}")
                continue
            raise FileNotFoundError(path)
        part = pd.read_csv(path)
        if "Unnamed: 0" in part.columns:
            part = part.drop(columns=["Unnamed: 0"])
        part["__split__"] = split
        frames.append(part)
        print(f"[extract] {split:<9}{len(part):>8} crops  {path}")

    df = pd.concat(frames, ignore_index=True)

    if exclude_compounds:
        before = len(df)
        df = df[~df[spec.treatment_col].isin(exclude_compounds)].reset_index(drop=True)
        print(f"[extract] excluded {len(exclude_compounds)} compound(s): "
              f"{before} -> {len(df)} rows")

    missing = [c for c in (spec.treatment_col, spec.plate_col, "SAMPLE_KEY")
               if c not in df.columns]
    if missing:
        raise KeyError(
            f"fold CSVs are missing required column(s) {missing}. "
            f"Present: {sorted(df.columns)}"
        )

    n_unique = df["SAMPLE_KEY"].nunique()
    print(f"[extract] total {len(df)} crops across {len(frames)} fold file(s), "
          f"{n_unique} unique SAMPLE_KEY")
    if n_unique != len(df):
        dup = df[df.duplicated("SAMPLE_KEY", keep=False)]
        combos = (dup.groupby("SAMPLE_KEY")["__split__"]
                  .agg(lambda x: "+".join(sorted(set(x)))).value_counts())
        print(f"  [warn] {len(df) - n_unique} duplicate crop(s); the same image "
              f"is encoded once per fold it appears in:")
        for combo, n in combos.items():
            print(f"           {n:>8} crops in {combo}")
        print("         With --split_aware_keys they land in separate h5 groups, "
              "which is what\n         the finetune path wants. Deduplicate "
              "downstream if you need one row per crop.")
    return df


def report_coverage(spec: DatasetSpec, df: pd.DataFrame, fold_path: str,
                    fold: int, img_dir: str, n_probe: int = 50) -> None:
    """Report what will be extracted, without running the model: crops per
    split, fold files on disk that are not used, and a probe of crop paths."""
    import glob
    import random

    print("\n[coverage] per split")
    print(df.groupby("__split__").size().to_string())
    print(f"  unique SAMPLE_KEY across all splits: {df['SAMPLE_KEY'].nunique()}")

    used = {os.path.basename(t.format(fold=fold)) for _, t in spec.folds}
    on_disk = sorted(os.path.basename(f) for f in
                     glob.glob(os.path.join(fold_path, "ds_*.csv")))
    unused = [f for f in on_disk if f not in used]
    print(f"\n[coverage] fold files in {fold_path}")
    for f in on_disk:
        print(f"  {'USED    ' if f in used else 'not used'}  {f}")
    if unused:
        extra = set()
        for f in unused:
            try:
                extra |= set(pd.read_csv(os.path.join(fold_path, f))["SAMPLE_KEY"])
            except Exception as e:                       # noqa: BLE001
                print(f"  [warn] could not read {f}: {e}")
        new_keys = extra - set(df["SAMPLE_KEY"])
        print(f"\n[coverage] the {len(unused)} unused file(s) hold "
              f"{len(new_keys)} crop(s) that are in NO used fold.")
        if new_keys:
            print("           Those images will NOT be extracted. Add them to "
                  "DATASETS[...].folds if they belong.")
            for k in list(sorted(new_keys))[:3]:
                print(f"             e.g. {k}")

    print(f"\n[coverage] probing {n_probe} crop paths on disk")
    rng = random.Random(0)
    sample = rng.sample(range(len(df)), min(n_probe, len(df)))
    bad = [k for k in (df["SAMPLE_KEY"].iloc[i] for i in sample)
           if not os.path.exists(spec.path_of(img_dir, k))]
    if bad:
        print(f"  [FAIL] {len(bad)}/{len(sample)} missing, e.g.")
        for k in bad[:3]:
            print(f"           {k} -> {spec.path_of(img_dir, k)}")
    else:
        print(f"  OK all {len(sample)} resolve, e.g. "
              f"{spec.path_of(img_dir, df['SAMPLE_KEY'].iloc[sample[0]])}")

    print(f"\n[coverage] would extract {len(df)} crops x {spec.n_channels} "
          f"channels = {len(df) * spec.n_channels} forward passes")


def resolve_cell_type(spec: DatasetSpec, df: pd.DataFrame, img_dir: str) -> pd.Series:
    """CELL_TYPE per row, by the same rule each loader uses at training time."""
    if "CELL_TYPE" in df.columns:
        return df["CELL_TYPE"]
    if spec.cell_type is not None:
        return pd.Series([spec.cell_type] * len(df), index=df.index)
    if "CELL_LINE" in df.columns:
        return df["CELL_LINE"]
    # cpg: plate barcode -> cell type, as in CPGDataset._load_text_embeddings
    meta = pd.read_csv(os.path.join(img_dir, "metadata", "experiment-metadata.tsv"),
                       sep="\t")
    plate_to_celltype = (
        meta.groupby("Assay_Plate_Barcode")["Cell_type"]
        .agg(lambda x: x.mode()[0] if len(x.mode()) == 1 else x.iloc[0])
        .to_dict()
    )
    return df[spec.plate_col].map(plate_to_celltype)


def perturbation_key(row, mode, treatment_col, plate_col, split_aware) -> str:
    base = (str(row[treatment_col]) if mode == "treatment"
            else f"{row[treatment_col]}__{row[plate_col]}")
    return f"{base}__{row['__split__']}" if split_aware else base


def write_sidecars(spec: DatasetSpec, df: pd.DataFrame, img_dir: str,
                   output_h5: str, split_aware: bool) -> None:
    """Write the prompts.csv and split.csv sidecars next to the h5."""
    df = df.copy()
    df["CELL_TYPE"] = resolve_cell_type(spec, df, img_dir)
    pert_type = (df["PERT_TYPE"].str.lower() if "PERT_TYPE" in df.columns
                 else pd.Series([spec.pert_type] * len(df), index=df.index))

    rows = []
    for pkey, sub in df.groupby("__pkey__"):
        f0 = sub.iloc[0]
        pt = pert_type.loc[sub.index[0]]
        if spec.caption == "gene":
            prompt = generate_gene_caption(
                f0["CELL_TYPE"], pt, f0[spec.treatment_col],
                f0.get(spec.moa_col),
            )
        else:
            prompt = generate_cell_caption(
                f0["CELL_TYPE"], pt, f0[spec.treatment_col],
                f0.get(spec.smiles_col) if spec.smiles_col else None,
            )
        rows.append({
            "perturbation_key": pkey,
            "prompt": prompt,
            "moa": f0[spec.moa_col] if spec.moa_col in sub.columns else "",
        })
    pd.DataFrame(rows).to_csv(output_h5 + ".prompts.csv", index=False)
    print(f"[extract] prompts -> {output_h5}.prompts.csv")

    if split_aware:
        split_map = {k: s["__split__"].iloc[0] for k, s in df.groupby("__pkey__")}
    else:
        pri = {"train": 0, "val": 1, "test": 2, "test_ood": 3}
        split_map = {}
        for pkey, sub in df.groupby("__pkey__"):
            uniq = sub["__split__"].unique()
            if len(uniq) > 1:
                print(f"  [warn] key {pkey!r} spans folds {list(uniq)} -- "
                      f"collapsing. Consider --split_aware_keys.")
            split_map[pkey] = max(sub["__split__"], key=lambda s: pri.get(s, 99))

    split_df = pd.DataFrame(
        [{"perturbation_key": k, "split": v} for k, v in split_map.items()]
    )
    split_df.to_csv(output_h5 + ".split.csv", index=False)
    print(f"[extract] split   -> {output_h5}.split.csv")
    print(split_df["split"].value_counts().to_string())

    if split_df["split"].value_counts().get("train", 0) == 0:
        raise RuntimeError(
            "Train split is empty. With IID folds this means --no-split_aware_keys "
            "collapsed each compound's train and val cells into one group."
        )


def h5_dataset_paths(h5: h5py.File) -> list:
    """Every dataset path in the file, depth-first.

    A perturbation name containing '/' (e.g. 'mevinolin/lovastatin') is nested
    by h5py, so top-level ``keys()`` would miss it; the full path is the key.
    """
    paths: list = []
    h5.visititems(lambda name, obj: paths.append(name)
                  if isinstance(obj, h5py.Dataset) else None)
    return sorted(paths)


def write_h5(output_h5: str, df: pd.DataFrame, buf: np.ndarray,
             attrs: Dict[str, object]) -> None:
    print(f"[extract] writing {output_h5} ...")
    os.makedirs(os.path.dirname(os.path.abspath(output_h5)), exist_ok=True)
    index = []
    with h5py.File(output_h5, "w") as h5:
        for pkey, sub in tqdm(df.groupby("__pkey__"), desc="HDF5 groups"):
            h5.create_dataset(pkey, data=buf[sub.index.to_numpy()],
                              compression="gzip", compression_opts=4)
            index.append(pd.DataFrame({"perturbation_key": pkey,
                                       "row": np.arange(len(sub)),
                                       "SAMPLE_KEY": sub["SAMPLE_KEY"].to_numpy()}))
        for k, v in attrs.items():
            h5.attrs[k] = v

    # Crop identity, which the h5 itself does not carry: row `r` of group
    # `perturbation_key` is this SAMPLE_KEY.
    pd.concat(index, ignore_index=True).to_csv(output_h5 + ".index.csv",
                                               index=False)
    print(f"[extract] crop index -> {output_h5}.index.csv")


# ===================================================================== #
# Stage 1 -- one grayscale pass per channel
# ===================================================================== #
class CellChannelDataset(Dataset):
    """One item per (crop, channel): the model input for that channel.

    ``prep`` maps the raw channel (float32 on the stored [0, 255] scale) to
    the tensor the model is called with.
    """

    def __init__(self, df: pd.DataFrame, spec: DatasetSpec, img_dir: str,
                 prep: Callable[[np.ndarray], torch.Tensor]):
        self.df = df.reset_index(drop=True)
        self.spec = spec
        self.img_dir = img_dir
        self.prep = prep
        self.index = [(i, c) for i in range(len(self.df))
                      for c in range(spec.n_channels)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        cell_i, chan = self.index[j]
        path = self.spec.path_of(self.img_dir,
                                 self.df["SAMPLE_KEY"].iloc[cell_i])
        arr = np.load(path)
        C = self.spec.n_channels
        if arr.ndim == 3 and arr.shape[-1] != C and arr.shape[0] == C:
            arr = arr.transpose(1, 2, 0)
        if arr.ndim != 3 or arr.shape[-1] < C:
            raise ValueError(f"unexpected array shape {arr.shape} at {path}")

        return self.prep(arr[..., chan].astype(np.float32)), cell_i, chan


@torch.no_grad()
def extract_per_channel(args, spec: DatasetSpec, df: pd.DataFrame,
                        prep: Callable[[np.ndarray], torch.Tensor],
                        forward: Callable[[torch.Tensor], torch.Tensor],
                        D: int, desc: str) -> np.ndarray:
    """``(n_crops, C, D)`` -- ``forward`` run on every channel of every crop."""
    dl = DataLoader(CellChannelDataset(df, spec, args.img_dir, prep),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    buf = np.zeros((len(df), spec.n_channels, D), dtype=np.float32)
    for x, cell_i, chan in tqdm(dl, desc=desc):
        feats = forward(x.to(args.device, non_blocking=True))
        buf[cell_i.numpy(), chan.numpy()] = feats.float().cpu().numpy()
    return buf


def build_dino(model_name: str, device: str, image_size: Optional[int]):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    if image_size is None:
        # 224 for DINOv2
        image_size = (proc.crop_size["height"] if getattr(proc, "crop_size", None)
                      else proc.size.get("shortest_edge", 224))
    tfm = transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.Normalize(mean=proc.image_mean, std=proc.image_std),
    ])

    def prep(g: np.ndarray) -> torch.Tensor:
        # Global /255: crops are stored in [0, 255]; no per-crop stretch.
        rgb = np.repeat(g[:, :, None] / 255.0, 3, axis=2)
        return tfm(torch.from_numpy(rgb).permute(2, 0, 1))

    print(f"[extract] {model_name} @ {image_size}px, "
          f"mean={proc.image_mean} std={proc.image_std}")
    return model, prep


def run_dinov2(args, spec: DatasetSpec, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    dino, prep = build_dino(args.dinov2_model, args.device, args.image_size)
    D = dino.config.hidden_size

    def forward(x: torch.Tensor) -> torch.Tensor:
        out = dino(pixel_values=x)
        return (out.pooler_output if getattr(out, "pooler_output", None) is not None
                else out.last_hidden_state[:, 0])

    buf = extract_per_channel(args, spec, df, prep, forward, D, "DINOv2 features")

    return buf, {
        "model": "dinov2g",
        "dinov2_model": args.dinov2_model,
        "channel_order": ",".join(spec.channel_names),
        "feature_dim": D,
        "intensity_norm": "global/255+imagenet",
    }


# ===================================================================== #
# Stage 1b -- MorphEm, DINO "Bag of Channels" ViT-S/16 trained on CHAMMI-75
# ===================================================================== #
# One grayscale pass per channel, (n, C, 384). Preprocessing follows the model
# card: raw [0, 255] input, saturation-noise injection, per-image instance norm.
def inject_saturation_noise(x: torch.Tensor, low: float = 200.0,
                            high: float = 255.0) -> torch.Tensor:
    """Replace saturated (exactly 255) pixels with U(low, high) noise.

    Applied at inference too, as on the model card. Draws from the global
    torch RNG inside the DataLoader worker, so a pass is reproducible only for
    a fixed ``--seed`` / ``--num_workers`` / ``--batch_size``.
    """
    return torch.where(x == 255.0, torch.empty_like(x).uniform_(low, high), x)


def per_image_normalize(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """``nn.InstanceNorm2d(1, affine=False)`` on a single image."""
    return (x - x.mean()) / torch.sqrt(x.var(unbiased=False) + eps)


def build_morphem(model_name: str, device: str, image_size: Optional[int]):
    from transformers import AutoModel, PreTrainedModel
    # The checkpoint's own ViT class never calls post_init(), so on
    # transformers >= 5 all_tied_weights_keys is missing and the loader raises.
    # Nothing in this ViT is tied: shim an empty mapping for the load only.
    tie_shim = not hasattr(PreTrainedModel, "all_tied_weights_keys")
    if tie_shim:
        PreTrainedModel.all_tied_weights_keys = {}
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    finally:
        if tie_shim:
            del PreTrainedModel.all_tied_weights_keys
    if not hasattr(model, "all_tied_weights_keys"):
        model.all_tied_weights_keys = {}
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    if model.config.in_chans != 1:
        raise RuntimeError(
            f"{model_name} takes in_chans={model.config.in_chans}; the bag-of-"
            f"channels recipe this stage implements expects a 1-channel model"
        )
    if image_size is None:
        image_size = int(model.config.img_size)         # 224
    resize = transforms.Resize((image_size, image_size), antialias=True)

    def prep(g: np.ndarray) -> torch.Tensor:
        # No /255: the raw scale is what the injector and the norm expect.
        x = torch.from_numpy(g).unsqueeze(0)            # (1, H, W) in [0, 255]
        return resize(per_image_normalize(inject_saturation_noise(x)))

    print(f"[extract] {model_name} @ {image_size}px, ViT-S/{model.config.patch_size} "
          f"bag-of-channels, saturation noise + per-image instance norm")
    return model, prep


def run_morphem(args, spec: DatasetSpec, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    model, prep = build_morphem(args.morphem_model, args.device, args.image_size)
    D = int(model.config.embed_dim)

    def forward(x: torch.Tensor) -> torch.Tensor:
        return model.forward_features(x)["x_norm_clstoken"]

    buf = extract_per_channel(args, spec, df, prep, forward, D, "MorphEm features")

    return buf, {
        "model": "morphem",
        "morphem_model": args.morphem_model,
        "channel_order": ",".join(spec.channel_names),
        "feature_dim": D,
        "intensity_norm": "saturation-noise+per-image-instancenorm",
    }


# ===================================================================== #
# Stage 2 -- CellCLIP CrossChannelFormer over a bag of one crop
# ===================================================================== #
def resolve_channel_map(spec: DatasetSpec, arg: str) -> Optional[Tuple[int, ...]]:
    """``auto`` | ``none`` | ``"0:4,1:3,2:0"`` -> rows of the pretrained table."""
    if arg == "none":
        return None
    if arg == "auto":
        return spec.cellclip_rows
    rows = [None] * spec.n_channels
    for pair in arg.split(","):
        tgt, src = (int(v) for v in pair.strip().split(":"))
        if not 0 <= tgt < spec.n_channels:
            raise ValueError(f"channel {tgt} out of range for C={spec.n_channels}")
        if not 0 <= src < len(BRAY_SLOTS):
            raise ValueError(f"pretrained row {src} out of range")
        rows[tgt] = src
    if any(r is None for r in rows):
        raise ValueError(f"--channel_map must cover all {spec.n_channels} channels")
    return tuple(rows)


def _stub_molecule_deps() -> None:
    """Register empty ``graphium`` / ``torch_geometric`` if they are missing.

    ``src/clip/model.py`` imports both at module level for the molecule
    tower; the image tower never touches them. The stubs only satisfy the
    import statements.
    """
    import importlib
    import sys
    import types

    def missing(name: str) -> bool:
        try:
            importlib.import_module(name)
            return False
        except ImportError:
            return True

    def module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        if "." in name:
            parent, _, leaf = name.rpartition(".")
            setattr(sys.modules[parent], leaf, mod)
        return mod

    def stub_fn(name: str) -> Callable:
        def fail(*_a, **_kw):
            raise RuntimeError(f"{name} is a stub -- install graphium to use "
                               f"the molecule tower")
        return fail

    if missing("graphium.config._loader"):
        module("graphium")
        module("graphium.config")
        loader = module("graphium.config._loader")
        for fn in ("load_accelerator", "load_architecture", "load_datamodule",
                   "load_yaml_config"):
            setattr(loader, fn, stub_fn(f"graphium.config._loader.{fn}"))
        module("graphium.nn")
        module("graphium.nn.architectures")
        arch = module("graphium.nn.architectures.global_architectures")
        arch.FullGraphMultiTaskNetwork = type(
            "FullGraphMultiTaskNetwork", (torch.nn.Module,), {})
        print("[cellclip] graphium not installed -- stubbed (molecule tower "
              "unavailable, image tower unaffected)")

    if missing("torch_geometric.data"):
        module("torch_geometric")
        data = module("torch_geometric.data")
        data.Data = type("Data", (), {})
        data.Batch = type("Batch", (), {})
        print("[cellclip] torch_geometric not installed -- stubbed")


def load_cellclip(ckpt: str, ckpt_filename: str, input_dim: int, loss_type: str,
                  device: str):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _stub_molecule_deps()
    from src.helper import load as cellclip_load

    if os.path.isfile(ckpt):
        path = ckpt
    else:
        from huggingface_hub import hf_hub_download
        print(f"[cellclip] downloading {ckpt}/{ckpt_filename} ...")
        path = hf_hub_download(repo_id=ckpt, filename=ckpt_filename)
    # Load on CPU: safetensors' rust backend is finicky about device strings.
    model = cellclip_load(model_path=path, device="cpu", model_type="cell_clip",
                          input_dim=input_dim, loss_type=loss_type)
    return model.to(device).eval()


def install_channel_table(model, spec: DatasetSpec,
                          rows: Optional[Tuple[int, ...]]) -> None:
    """Re-point the (5, d) channel table at the dataset's C channels, in place."""
    table = model.visual.channel_embed.detach()
    if table.shape[0] != len(BRAY_SLOTS):
        raise RuntimeError(
            f"pretrained channel table has {table.shape[0]} rows, expected "
            f"{len(BRAY_SLOTS)} -- the slot semantics in BRAY_SLOTS no longer apply"
        )
    if rows is None:
        new = torch.zeros(spec.n_channels, table.shape[1],
                          dtype=table.dtype, device=table.device)
        print("[cellclip] channel embedding ZEROED -- the transformer is "
              "permutation-invariant over channels and asserts no identity")
    else:
        new = table[list(rows)].clone()
        for i, (name, r) in enumerate(zip(spec.channel_names, rows)):
            print(f"[cellclip] ch{i} {name:<14} -> pretrained row {r} "
                  f"({BRAY_SLOTS[r]})")
    model.visual.channel_embed = torch.nn.Parameter(new, requires_grad=False)
    model.visual.input_channels = spec.n_channels


@torch.no_grad()
def run_cellclip(args, spec: DatasetSpec) -> None:
    rows = resolve_channel_map(spec, args.channel_map)
    model = load_cellclip(args.pretrained_ckpt, args.ckpt_filename,
                          args.input_dim, args.loss_type, args.device)
    install_channel_table(model, spec, rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_h5)), exist_ok=True)
    with h5py.File(args.from_h5, "r") as src, h5py.File(args.output_h5, "w") as dst:
        keys = h5_dataset_paths(src)
        nested = [k for k in keys if "/" in k]
        if nested:
            print(f"[cellclip] {len(nested)} key(s) hold a '/' and are nested "
                  f"in the h5: {', '.join(nested[:4])}"
                  f"{' ...' if len(nested) > 4 else ''}")
        for pkey in tqdm(keys, desc="CellCLIP groups"):
            feats = src[pkey][:]                              # (n, C, D)
            if feats.shape[1] != spec.n_channels:
                raise ValueError(f"{pkey}: C={feats.shape[1]}, "
                                 f"expected {spec.n_channels}")
            if feats.shape[2] != args.input_dim:
                raise ValueError(f"{pkey}: D={feats.shape[2]}, "
                                 f"--input_dim={args.input_dim}")
            out = []
            for s in range(0, len(feats), args.batch_size):
                # Bag of one: MILPooling over M=1 is the identity (softmax of a
                # single logit is 1), so it is skipped -- calling it would also
                # emit NaN for an all-zero channel, which its mask sends to -inf.
                x = torch.from_numpy(feats[s:s + args.batch_size]).clone()
                x = x.to(args.device).type(model.dtype)
                out.append(model.encode_image(x).float().cpu().numpy())
            dst.create_dataset(pkey, data=np.concatenate(out).astype(np.float32),
                               compression="gzip", compression_opts=4)

        dst.attrs["model"] = "cellclip"
        dst.attrs["source_h5"] = os.path.abspath(args.from_h5)
        dst.attrs["channel_order"] = ",".join(spec.channel_names)
        dst.attrs["channel_map"] = ("none" if rows is None else
                                    ",".join(f"{i}:{r}" for i, r in enumerate(rows)))
        dst.attrs["pretrained_slots"] = ",".join(BRAY_SLOTS)
        dst.attrs["feature_dim"] = int(model.visual.output_dim)
        dst.attrs["bag_size"] = 1
        for k in ("dinov2_model", "intensity_norm"):
            if k in src.attrs:
                dst.attrs[k] = src.attrs[k]

    for suffix in (".prompts.csv", ".split.csv"):
        side = args.from_h5 + suffix
        if os.path.exists(side):
            pd.read_csv(side).to_csv(args.output_h5 + suffix, index=False)
            print(f"[cellclip] copied {suffix} -> {args.output_h5}{suffix}")


# ===================================================================== #
# CLI
# ===================================================================== #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model", required=True,
                   choices=("dinov2g", "morphem", "cellclip"))
    p.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    p.add_argument("--output_h5", required=True)

    # dinov2g / morphem stages
    p.add_argument("--img_dir")
    p.add_argument("--fold_path",
                   help="Dir holding the ds_<dataset>_<split>*.csv fold files")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--dinov2_model", default="facebook/dinov2-giant",
                   choices=["facebook/dinov2-small", "facebook/dinov2-base",
                            "facebook/dinov2-large", "facebook/dinov2-giant"])
    p.add_argument("--morphem_model", default="CaicedoLab/MorphEm",
                   help="HF repo id or local dir. Loaded with "
                        "trust_remote_code -- it ships its own ViT.")
    p.add_argument("--image_size", type=int, default=None,
                   help="Default: 224 -- the processor's crop size for "
                        "dinov2g, config.img_size for morphem.")
    p.add_argument("--exclude_compounds", nargs="*", default=None)

    # cellclip stage
    p.add_argument("--from_h5", help="dinov2g h5 to read per-channel features from")
    p.add_argument("--channel_map", default="auto",
                   help="'auto' (per-dataset default), 'none' (zero the channel "
                        "table), or explicit 'tgt:src' pairs e.g. '0:4,1:3,2:0' "
                        "where src indexes " + "/".join(BRAY_SLOTS))
    p.add_argument("--pretrained_ckpt", default="suinleelab/CellCLIP")
    p.add_argument("--ckpt_filename", default="model.safetensors")
    p.add_argument("--input_dim", type=int, default=1536)
    p.add_argument("--loss_type", default="cwcl")

    # grouping / sidecars
    p.add_argument("--perturbation_key", default="treatment",
                   choices=("treatment", "treatment+plate"))
    p.add_argument("--split_aware_keys", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Append '__<split>' to every key. REQUIRED for IID folds.")

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit_crops", type=int, default=None,
                   help="Smoke test: extract only ~N crops, sampled evenly "
                        "across the fold files so every one is exercised.")
    p.add_argument("--dry_run", action="store_true",
                   help="Report fold coverage and probe crop paths, then exit "
                        "without loading a model.")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    required = ({"from_h5"} if args.model == "cellclip"
                else {"img_dir", "fold_path"})
    for name in sorted(required):
        if getattr(args, name) is None:
            p.error(f"--{name} is required for --model {args.model}")
    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)          # morphem's saturation noise
    spec = DATASETS[args.dataset]

    if args.model == "cellclip":
        run_cellclip(args, spec)
        return

    df = load_folds(spec, args.fold_path, args.fold, args.exclude_compounds)

    if args.dry_run:
        if args.limit_crops:
            print("[extract] --dry_run reports full coverage; "
                  "--limit_crops ignored")
        report_coverage(spec, df, args.fold_path, args.fold, args.img_dir)
        return

    if args.limit_crops:
        # Spread the sample over every fold file and as many perturbations as
        # it will reach.
        per = max(1, args.limit_crops // df["__split__"].nunique())
        rng = np.random.default_rng(args.seed)
        take = [rng.choice(g.index.to_numpy(), min(len(g), per), replace=False)
                for _, g in df.groupby("__split__")]
        df = df.loc[np.sort(np.concatenate(take))].reset_index(drop=True)
        print(f"[extract] --limit_crops: {len(df)} crops, "
              f"{per} per split, sampled at random")

    df["__pkey__"] = df.apply(
        lambda r: perturbation_key(r, args.perturbation_key, spec.treatment_col,
                                   spec.plate_col, args.split_aware_keys),
        axis=1,
    )
    print(f"[extract] {args.dataset}: crops={len(df)}  "
          f"unique {spec.treatment_col}={df[spec.treatment_col].nunique()}  "
          f"unique perturbations={df['__pkey__'].nunique()}  "
          f"C={spec.n_channels} ({', '.join(spec.channel_names)})")

    runner = run_morphem if args.model == "morphem" else run_dinov2
    buf, attrs = runner(args, spec, df)
    attrs["perturbation_key_mode"] = args.perturbation_key
    write_h5(args.output_h5, df, buf, attrs)
    write_sidecars(spec, df, args.img_dir, args.output_h5, args.split_aware_keys)


if __name__ == "__main__":
    main()
