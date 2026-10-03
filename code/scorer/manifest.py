"""Per-crop manifest (plate, well, field, dose) for the embedding h5.

The h5 holds one dataset per ``{CPD_NAME}__{split}`` key; row ``i`` of a group
is the ``i``-th row of the concatenated fold CSVs carrying that key, so
replaying ``load_folds`` reproduces the alignment. ``verify_against_h5`` binds
the manifest to the features by SAMPLE_KEY through the ``.index.csv`` sidecar
when it exists, and by group length otherwise.

    BBBC021  SAMPLE_KEY = {WEEK}_{PLATE}_{TABLE}_{IMAGE}_{OBJECT}
    CPG      SAMPLE_KEY = {PLATE}_{WELL}_{site}_{idx}
    RxRx1    SAMPLE_KEY = {CELL}-{exp}_{plate}_{WELL}_{site}_{idx}

``python scorer/manifest.py --dataset ... --fold_path ...`` prints the fold columns
and a parsed key; it needs no model and no embeddings.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.extract_features import (DATASETS, h5_dataset_paths,  # noqa: E402
                              load_folds, perturbation_key)
from scorer.perturbation_id import apply_identity                 # noqa: E402

# The perturbation/control identity rule is part of the manifest cache file name.
IDENTITY_TAG = {"cpg": "_broad_state", "rxrx1": "_batch_negctl"}

# Columns a fold CSV might already carry, in preference order.
WELL_COLS = ("WELL", "WELL_POSITION", "Image_Metadata_Well_DAPI", "Metadata_Well")
DOSE_COLS = ("DOSE", "CONCENTRATION", "Image_Metadata_Concentration",
             "Metadata_mmoles_per_liter")


# ===================================================================== #
# SAMPLE_KEY -> plate / well / site
# ===================================================================== #
def _parse_keys(dataset: str, keys: pd.Series) -> pd.DataFrame:
    p = keys.str.split("_")
    if dataset == "bbbc021":
        return pd.DataFrame({
            "week": p.str[0],
            "key_plate": p.str[1],
            "table_number": pd.to_numeric(p.str[2], errors="coerce"),
            "image_number": pd.to_numeric(p.str[3], errors="coerce"),
            "object_number": pd.to_numeric(p.str[4], errors="coerce"),
        }, index=keys.index)
    if dataset == "cpg":
        return pd.DataFrame({
            "key_plate": p.str[0], "well": p.str[1],
            "site": p.str[2], "object_number": pd.to_numeric(p.str[3],
                                                             errors="coerce"),
        }, index=keys.index)
    if dataset == "rxrx1":
        # U2OS-01_1_B02_s1_14 -> experiment U2OS-01, plate 1, well B02, site s1.
        # The cell line has to stay in the plate id: HEPG2-01_1 and U2OS-01_1
        # are different physical plates.
        cell = keys.str.split("-", n=1).str[0]
        q = keys.str.split("-", n=1).str[1].str.split("_")
        return pd.DataFrame({
            "key_plate": cell + "-" + q.str[0] + "_" + q.str[1],
            "well": q.str[2],
            "site": q.str[3], "object_number": pd.to_numeric(q.str[4],
                                                             errors="coerce"),
        }, index=keys.index)
    raise KeyError(dataset)


def _well_rc(well: pd.Series) -> pd.DataFrame:
    """'L03' -> row 12, col 3. Used for the plate-edge artefact check."""
    m = well.astype(str).str.extract(r"^([A-Za-z]+)0*(\d+)$")
    row = m[0].str.upper().apply(
        lambda s: sum((ord(c) - 64) * 26 ** i
                      for i, c in enumerate(reversed(str(s))))
        if isinstance(s, str) else np.nan)
    return pd.DataFrame({"well_row": row,
                         "well_col": pd.to_numeric(m[1], errors="coerce")},
                        index=well.index)


# ===================================================================== #
# Manifest
# ===================================================================== #
def build_manifest(dataset: str, fold_path: str, fold: int = 0,
                   image_csv: Optional[str] = None,
                   moa_csv: Optional[str] = None,
                   exclude_compounds: Optional[List[str]] = None,
                   perturbation_key_mode: str = "treatment",
                   split_aware_keys: bool = True) -> pd.DataFrame:
    """One row per (h5 group, row index), with everything needed downstream."""
    spec = DATASETS[dataset]
    df = load_folds(spec, fold_path, fold, exclude_compounds)
    df["__pkey__"] = df.apply(
        lambda r: perturbation_key(r, perturbation_key_mode, spec.treatment_col,
                                   spec.plate_col, split_aware_keys), axis=1)

    parts = []
    for pkey, sub in df.groupby("__pkey__"):
        blk = sub.copy()
        blk["pkey"] = pkey
        blk["row"] = np.arange(len(blk))
        parts.append(blk)
    man = pd.concat(parts, ignore_index=True)
    # after the pkey, which addresses the h5 and must stay on treatment_col
    man = apply_identity(dataset, man, spec.treatment_col)

    keep = [c for c in (spec.treatment_col, spec.plate_col, spec.moa_col,
                        "SAMPLE_KEY", "__split__", "pkey", "row",
                        "pert", "pert_role", "control_label")
            if c in man.columns]
    man = man[keep + [c for c in WELL_COLS + DOSE_COLS if c in man.columns]]
    man = man.rename(columns={spec.treatment_col: "compound",
                              spec.plate_col: "plate",
                              spec.moa_col: "moa",
                              "__split__": "split"})
    man["compound"] = man.pop("pert")

    man = pd.concat([man, _parse_keys(dataset, man["SAMPLE_KEY"])], axis=1)
    if "plate" not in man.columns:
        man["plate"] = man["key_plate"]
    if dataset == "rxrx1":
        # `plate` (the contrast, normalisation and control-pairing group) is
        # the experiment (BATCH): RxRx1 has one negative-control well per
        # physical plate. Wells and fields are addressed through `phys_plate`,
        # because the same well name exists on every plate of an experiment.
        man["batch"] = man["plate"]
        man["phys_plate"] = man["key_plate"]
        print(f"[manifest] rxrx1: group `plate` = BATCH "
              f"({man['plate'].nunique()} experiments over "
              f"{man['phys_plate'].nunique()} physical plates, kept as "
              f"'phys_plate')")

    # --- well ---------------------------------------------------------- #
    have = [c for c in WELL_COLS if c in man.columns]
    if have:
        man["well"] = man[have[0]].astype(str)
        print(f"[manifest] well from fold column {have[0]!r}")
    elif "well" in man.columns:
        print("[manifest] well parsed out of SAMPLE_KEY")
    elif dataset == "bbbc021" and image_csv:
        man = _join_bbbc021_image_csv(man, image_csv)
    elif dataset == "bbbc021" and {"table_number", "image_number"} <= set(man.columns):
        man = _bbbc021_well_from_blocks(man)
    else:
        fid = (man["image_number"].astype(str) if "image_number" in man.columns
               else man["SAMPLE_KEY"].astype(str))
        man["well"] = man["plate"].astype(str) + ":F" + fid
        print("[manifest] ! no well information. Falling back to the FIELD as "
              "the analysis unit.\n"
              "           Crops in one field are a subset of one well, so this "
              "is a strictly weaker\n"
              "           grouping: same-well fields land in different folds "
              "and AUROC reads high.\n"
              "           Pass --image_csv BBBC021_v1_image.csv to fix it.")

    pp = man["phys_plate"] if "phys_plate" in man.columns else man["plate"]
    man["well_id"] = pp.astype(str) + "|" + man["well"].astype(str)
    man = pd.concat([man, _well_rc(man["well"])], axis=1)

    # --- field, and cells-per-field as the density proxy ---------------- #
    fld = (["plate", "table_number", "image_number"] if dataset == "bbbc021"
           else ["phys_plate" if "phys_plate" in man.columns else "plate",
                 "well", "site"])
    man["field_id"] = man[fld].astype(str).agg("|".join, axis=1)
    man["cells_per_field"] = man.groupby("field_id")["row"].transform("size")

    # --- dose ----------------------------------------------------------- #
    dose = [c for c in DOSE_COLS if c in man.columns]
    if dose:
        man["concentration"] = pd.to_numeric(man[dose[0]], errors="coerce")
    elif "concentration" not in man.columns:
        man["concentration"] = np.nan

    if moa_csv and man["concentration"].notna().any():
        man = _join_moa_csv(man, moa_csv)

    # --- controls, dedup ------------------------------------------------ #
    before = len(man)
    man = man.drop_duplicates("SAMPLE_KEY", keep="first").reset_index(drop=True)
    if len(man) < before:
        print(f"[manifest] dropped {before - len(man)} duplicate crop(s) "
              f"(the same image encoded once per fold it appears in)")
    return man


BBBC021_FIELDS_PER_WELL = 4


def _bbbc021_well_from_blocks(man: pd.DataFrame) -> pd.DataFrame:
    """Recover the well from (plate, table, image number). No image CSV needed.

    BBBC021 images every well as exactly four consecutively numbered fields, so
    ``(image_number - 1) // 4`` indexes the well within a (plate, table). This
    recovers the well grouping (what the folds and the bootstrap need) but not
    the well name, so ``well_row`` / ``well_col`` stay unset; pass --image_csv
    for those.
    """
    blk = (man["image_number"] - 1) // BBBC021_FIELDS_PER_WELL
    man["well"] = ("W" + man["table_number"].astype("Int64").astype(str)
                   + "_" + blk.astype("Int64").astype(str))
    n = man.groupby([man["plate"].astype(str), man["well"]])["image_number"].nunique()
    print(f"[manifest] well reconstructed from the image-number block "
          f"((image_number-1)//{BBBC021_FIELDS_PER_WELL} within plate+table): "
          f"{len(n)} wells, median {n.median():.0f} field(s) each "
          f"(expect {BBBC021_FIELDS_PER_WELL})")
    over = int((n > BBBC021_FIELDS_PER_WELL).sum())
    if over:
        print(f"           ! {over} well(s) exceed {BBBC021_FIELDS_PER_WELL} "
              f"fields -- the block assumption does not hold on this crop set, "
              f"pass --image_csv BBBC021_v1_image.csv instead")
    return man


def _join_bbbc021_image_csv(man: pd.DataFrame, image_csv: str) -> pd.DataFrame:
    """(TableNumber, ImageNumber) -> well, concentration, replicate."""
    img = pd.read_csv(image_csv)
    cols = {c.lower(): c for c in img.columns}
    need = {"tablenumber": None, "imagenumber": None,
            "image_metadata_well_dapi": "well",
            "image_metadata_concentration": "concentration",
            "image_metadata_compound": "compound_img",
            "replicate": "replicate"}
    missing = [k for k in ("tablenumber", "imagenumber",
                           "image_metadata_well_dapi") if k not in cols]
    if missing:
        raise KeyError(f"{image_csv} lacks {missing}; present: {list(img.columns)}")
    ren = {cols[k]: (v or k) for k, v in need.items() if k in cols}
    img = img[list(ren)].rename(columns=ren)
    img["table_number"] = pd.to_numeric(img["tablenumber"], errors="coerce")
    img["image_number"] = pd.to_numeric(img["imagenumber"], errors="coerce")
    img = img.drop(columns=["tablenumber", "imagenumber"]).drop_duplicates(
        ["table_number", "image_number"])

    out = man.merge(img, on=["table_number", "image_number"], how="left")
    miss = out["well"].isna().mean()
    print(f"[manifest] joined {image_csv}: {(1 - miss) * 100:.1f}% of crops "
          f"got a well")
    if miss > 0.01:
        print("           ! more than 1% unmatched -- check that the image csv "
              "covers every week")
    if "compound_img" in out.columns:
        a = out["compound"].astype(str).str.lower()
        b = out["compound_img"].astype(str).str.lower()
        bad = float((a != b)[out["compound_img"].notna()].mean())
        print(f"           compound disagreement with the fold CSV: {bad * 100:.2f}%")
        out = out.drop(columns=["compound_img"])
    return out


def _join_moa_csv(man: pd.DataFrame, moa_csv: str) -> pd.DataFrame:
    """BBBC021_v1_moa.csv -- MoA is per compound-CONCENTRATION, not per compound."""
    moa = pd.read_csv(moa_csv)
    cols = {c.lower(): c for c in moa.columns}
    try:
        moa = moa.rename(columns={cols["compound"]: "compound",
                                  cols["concentration"]: "concentration",
                                  cols["moa"]: "moa_ref"})
    except KeyError:
        print(f"[manifest] ! {moa_csv} has no compound/concentration/moa "
              f"columns: {list(moa.columns)}")
        return man
    moa = moa[["compound", "concentration", "moa_ref"]]
    moa["compound"] = moa["compound"].astype(str).str.lower()
    out = man.copy()
    out["_c"] = out["compound"].astype(str).str.lower()
    out = out.merge(moa, left_on=["_c", "concentration"],
                    right_on=["compound", "concentration"], how="left",
                    suffixes=("", "_y")).drop(columns=["_c", "compound_y"],
                                              errors="ignore")
    print(f"[manifest] MoA labels on {out['moa_ref'].notna().mean() * 100:.1f}% "
          f"of crops ({out['moa_ref'].nunique()} classes)")
    return out


def verify_against_h5(man: pd.DataFrame, h5_path: str) -> pd.DataFrame:
    """Bind the manifest to the stored features; hard-fail if it cannot.

    With the ``.index.csv`` sidecar, SAMPLE_KEY is mapped to the stored
    ``(perturbation_key, row)``, so the manifest is rebound by crop identity
    and any --fold_path over the same crops works. Without it the replayed
    fold-CSV order is used and the check falls back to row counts.
    """
    idx_path = h5_path + ".index.csv"
    if os.path.exists(idx_path):
        idx = (pd.read_csv(idx_path)[["SAMPLE_KEY", "perturbation_key", "row"]]
               .drop_duplicates("SAMPLE_KEY", keep="first"))
        out = man.merge(idx, on="SAMPLE_KEY", how="left", suffixes=("", "_h5"))
        missing = out["perturbation_key"].isna()
        if missing.any():
            ex = out.loc[missing, ["SAMPLE_KEY", "pkey", "row"]].head(10)
            raise RuntimeError(
                f"{int(missing.sum())} of {len(man)} manifest crop(s) are not "
                f"in {os.path.basename(idx_path)}. Repartitioning alone cannot "
                f"cause this -- the fold CSVs describe a different crop "
                f"universe than the features were built from. Re-extract, or "
                f"point --fold_path at the directory the h5 was built from.\n"
                + ex.to_string(index=False))
        moved = int(((out["pkey"] != out["perturbation_key"])
                     | (out["row"] != out["row_h5"])).sum())
        out["pkey"] = out["perturbation_key"]
        out["row"] = out["row_h5"].astype(int)
        out = out.drop(columns=["perturbation_key", "row_h5"])
        print(f"[manifest] bound {len(out)} crop(s) by SAMPLE_KEY against "
              f"{os.path.basename(idx_path)}"
              + (f"; {moved} rebound -- the fold partition differs from the "
                 f"one the features were extracted under, which is fine, the "
                 f"split does not enter any number" if moved else ""))
        return out

    with h5py.File(h5_path, "r") as f:
        shapes = {k: f[k].shape for k in h5_dataset_paths(f)}
    counts = man.groupby("pkey")["row"].agg(["size", "max"])
    # dedup means size can be < stored rows; max+1 is the count before dedup
    bad = []
    for pkey, r in counts.iterrows():
        if pkey not in shapes:
            bad.append(f"{pkey}: not in h5")
        elif shapes[pkey][0] < r["max"] + 1:
            bad.append(f"{pkey}: h5 has {shapes[pkey][0]} rows, "
                       f"manifest indexes {int(r['max']) + 1}")
    extra = sorted(set(shapes) - set(counts.index))
    if bad:
        raise RuntimeError(
            "manifest and h5 disagree -- the fold CSVs are not the ones the "
            "features were extracted from:\n  " + "\n  ".join(bad[:10])
            + "\n(this h5 predates the .index.csv sidecar, so the check is on "
              "row counts only; re-extracting writes the index and the check "
              "becomes per-crop)")
    print(f"[manifest] verified {len(counts)} group(s) against {os.path.basename(h5_path)}"
          + (f"; {len(extra)} h5 group(s) not in the manifest" if extra else ""))
    return man


# ===================================================================== #
# Embedding access
# ===================================================================== #
class EmbeddingStore:
    """Reads only the h5 groups a contrast needs, so C=5/6 datasets fit in RAM.

    dinov2g is stored as (n, C, 1536) and flattened to (n, C*1536); CellCLIP is
    already (n, 512).
    """

    def __init__(self, h5_path: str, cache_keys: Optional[List[str]] = None,
                 max_cache_gb: float = 8.0):
        self.path = h5_path
        self.f = h5py.File(h5_path, "r")
        self.attrs = dict(self.f.attrs)
        self.keys = set(h5_dataset_paths(self.f))
        k0 = next(iter(sorted(self.keys)))
        sh = self.f[k0].shape
        self.dim = int(np.prod(sh[1:]))
        self.n_channels = int(sh[1]) if len(sh) == 3 else 1
        self._cache: Dict[str, np.ndarray] = {}
        # Control blocks are read once per contrast, so they are cached up to
        # the budget; the rest are read per contrast.
        used, budget, skipped = 0, max_cache_gb * 1e9, 0
        for k in cache_keys or []:
            if k not in self.keys:
                continue
            nb = int(np.prod(self.f[k].shape)) * 4
            if used + nb > budget and self._cache:
                skipped += 1
                continue
            self._cache[k] = self._read(k)
            used += nb
        print(f"[store] {os.path.basename(h5_path)}  D={self.dim} "
              f"({self.n_channels} x {self.dim // self.n_channels})  "
              f"{len(self.keys)} groups"
              + (f", {len(self._cache)} cached ({used / 1e9:.1f} GB)"
                 if self._cache else "")
              + (f", {skipped} group(s) over --max_cache_gb, read per contrast"
                 if skipped else ""))

    def _read(self, pkey: str) -> np.ndarray:
        a = self.f[pkey][:]
        return a.reshape(len(a), -1).astype(np.float32)

    def matrix(self, man: pd.DataFrame) -> np.ndarray:
        """Rows of ``man``, in ``man``'s order."""
        man = man.reset_index(drop=True)          # index == position
        out = np.empty((len(man), self.dim), dtype=np.float32)
        for pkey, sub in man.groupby("pkey"):
            blk = self._cache.get(pkey)
            if blk is None:
                blk = self._read(pkey)
            out[sub.index.to_numpy()] = blk[sub["row"].to_numpy()]
        return out

    def close(self):
        self.f.close()


# ===================================================================== #
# CLI -- inspect only
# ===================================================================== #
def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    p.add_argument("--fold_path", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--h5", help="verify row alignment against this file")
    p.add_argument("--image_csv")
    p.add_argument("--moa_csv")
    p.add_argument("--out")
    args = p.parse_args()

    spec = DATASETS[args.dataset]
    raw = load_folds(spec, args.fold_path, args.fold, None)
    print(f"\n[inspect] fold CSV columns: {sorted(raw.columns)}")
    print(f"[inspect] example SAMPLE_KEY: {raw['SAMPLE_KEY'].iloc[0]}")
    print(_parse_keys(args.dataset, raw["SAMPLE_KEY"].head(3)).to_string())
    print(f"[inspect] well column present: "
          f"{[c for c in WELL_COLS if c in raw.columns] or 'NO'}")
    print(f"[inspect] dose column present: "
          f"{[c for c in DOSE_COLS if c in raw.columns] or 'NO'}\n")

    man = build_manifest(args.dataset, args.fold_path, args.fold,
                         args.image_csv, args.moa_csv)
    if args.h5:
        man = verify_against_h5(man, args.h5)
    print(f"\n[inspect] {len(man)} crops, {man['compound'].nunique()} compounds, "
          f"{man['plate'].nunique()} plates, {man['well_id'].nunique()} wells, "
          f"{man['field_id'].nunique()} fields")
    print(f"[inspect] crops per well: median "
          f"{man.groupby('well_id').size().median():.0f}")
    print(man.groupby("compound").size().sort_values(ascending=False)
          .head(10).to_string())
    if args.out:
        man.to_parquet(args.out, index=False)
        print(f"[inspect] -> {args.out}")


if __name__ == "__main__":
    main()
