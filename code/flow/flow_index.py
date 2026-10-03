"""Build the flow training index: one row per crop, every split, no scores.

    crop_id  path  y  split  compound  dose  moa  plate  well  text_idx

The crop universe is the scorer's manifest parquet; the index is joined with
``crop_scores.parquet`` on ``crop_id`` at load time, and ``unit_id`` comes from
the score cache. Which splits train is set in the training config.

    python flow/flow_index.py --manifest MAN.parquet --img_dir DIR --out idx.parquet
    python flow/flow_index.py ... --text_emb cellclip_text_full_emb.pt --fold_path DIR
    python flow/flow_index.py ... --check_paths 500
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

from typing import List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bbbc021_path(img_dir: str, sample_key: str) -> str:
    """``Week1_22123_1_11_3.0`` -> ``{img_dir}/Week1/22123/1_11_3.0.npy``,
    as ``BBBC021Dataset._build_paths``."""
    p = sample_key.split("_")
    return os.path.join(img_dir, p[0], p[1], "_".join(p[2:]) + ".npy")


def cpg_path(img_dir: str, sample_key: str) -> str:
    """``BR00117010_M01_9_144`` -> ``{img_dir}/BR00117010/M01_9/M01_9_144.npy``,
    as ``CPGDataset._build_paths``."""
    p = sample_key.split("_")
    return os.path.join(img_dir, p[0], f"{p[1]}_{p[2]}", "_".join(p[1:]) + ".npy")


def rxrx1_path(img_dir: str, sample_key: str) -> str:
    """``U2OS-01_1_B02_s1_14`` -> ``{img_dir}/01_1/B02/s1_14.npy``, as
    ``RxRx1Dataset._build_paths``."""
    p = sample_key.split("-", 1)[1].split("_")
    return os.path.join(img_dir, "_".join(p[:2]), p[2], "_".join(p[3:]) + ".npy")


PATH_OF = {"bbbc021": bbbc021_path, "cpg": cpg_path, "rxrx1": rxrx1_path}
# the manifest labels CPG/RxRx1 controls CONTROL (perturbation_id.py)
CONTROL_OF = {"bbbc021": "DMSO", "cpg": "CONTROL", "rxrx1": "CONTROL"}


def attach_text(idx: pd.DataFrame, fold_path: str, text_emb: str,
                pert_type: Optional[str] = None) -> pd.DataFrame:
    """Resolve each crop to a row of the CellCLIP text-embedding table.

    Prompts are rebuilt with ``generate_cell_caption`` and matched against the
    prompts stored in the .pt. The spelling of the perturbation type is
    ambiguous, so each candidate is tried and the one that resolves every
    prompt wins. A fold CSV needs ``SAMPLE_KEY``, ``CPD_NAME`` and ``SMILES``;
    ``PERT_TYPE`` is optional.
    """
    import torch
    from datasets.bbbc021_dataset import generate_cell_caption, BBBC021Dataset

    need = ["SAMPLE_KEY", "CPD_NAME", "SMILES"]
    parts, skipped = [], []
    for f in sorted(glob.glob(os.path.join(fold_path, "*.csv"))):
        d = pd.read_csv(f)
        miss = sorted(set(need) - set(d.columns))
        if miss:
            skipped.append((os.path.basename(f), miss))
            continue
        cols = need + (["PERT_TYPE"] if "PERT_TYPE" in d.columns else [])
        parts.append(d[cols])
        print(f"  [read] {os.path.basename(f)}: {len(d)} rows"
              + ("" if "PERT_TYPE" in d.columns else ", no PERT_TYPE column"))
    for name, miss in skipped:
        print(f"  [skip] {name}: missing {miss}")
    if not parts:
        raise SystemExit(f"no fold CSV under {fold_path} carries {need}")
    meta = pd.concat(parts, ignore_index=True).drop_duplicates("SAMPLE_KEY")

    m = idx.merge(meta, left_on="crop_id", right_on="SAMPLE_KEY", how="left")
    gap = m["CPD_NAME"].isna()
    if gap.any():
        raise SystemExit(
            f"{int(gap.sum())} of {len(m)} crop(s) in the manifest are in no "
            f"fold CSV under {fold_path} that carries {need}. Read the [read]/"
            f"[skip] lines above: a file skipped for a missing column is the "
            f"usual cause.")

    blob = torch.load(text_emb, map_location="cpu", weights_only=False)
    table = list(blob["prompts"])
    prompt_to_idx = {p: i for i, p in enumerate(table)}

    # Candidate spellings, most likely first. A per-row column beats a constant.
    cands: List[Tuple[str, pd.Series]] = []
    if pert_type:
        cands.append((f"--pert_type {pert_type!r}",
                      pd.Series([pert_type] * len(m), index=m.index)))
    if "PERT_TYPE" in m.columns and m["PERT_TYPE"].notna().any():
        col = m["PERT_TYPE"].fillna(BBBC021Dataset.PERTURBATION_TYPE).astype(str)
        cands += [("PERT_TYPE column, lowercased", col.str.lower()),
                  ("PERT_TYPE column, as written", col)]
    cands += [("constant 'Compound'",
               pd.Series(["Compound"] * len(m), index=m.index)),
              ("constant 'compound'",
               pd.Series(["compound"] * len(m), index=m.index))]

    tried = []
    for name, pt in cands:
        prompts = [generate_cell_caption(BBBC021Dataset.CELL_TYPE, p, c, s_)
                   for p, c, s_ in zip(pt, m["CPD_NAME"], m["SMILES"])]
        hit = sum(p in prompt_to_idx for p in prompts)
        tried.append((name, hit, prompts))
        if hit == len(prompts):
            idx = idx.copy()
            idx["text_idx"] = [prompt_to_idx[p] for p in prompts]
            print(f"[text] perturbation type: {name} -- all {len(prompts)} "
                  f"prompts resolved to {idx['text_idx'].nunique()} distinct "
                  f"row(s) of {tuple(blob['emb'].shape)}")
            return idx

    msg = ["no candidate perturbation type resolved every prompt:"]
    msg += [f"    {n:<32} {h}/{len(m)} matched" for n, h, _ in tried]
    msg.append("\n  constructed (best candidate):")
    msg += [f"    {p!r}" for p in tried[0][2][:3]]
    msg.append("  stored in the .pt:")
    msg += [f"    {p!r}" for p in table[:3]]
    msg.append("\n  Pass the right spelling with --pert_type, or re-extract "
               "the prompt embeddings against the current fold CSVs.")
    raise SystemExit("\n".join(msg))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", required=True,
                   help="the scorer's manifest parquet -- the crop universe")
    p.add_argument("--img_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="bbbc021", choices=tuple(PATH_OF))
    p.add_argument("--control", default=None,
                   help="control label(s); default DMSO on bbbc021, CONTROL "
                        "on cpg/rxrx1")
    p.add_argument("--text_emb", default=None,
                   help="cellclip_text_full_emb.pt; without it the trainer has "
                        "no conditioning and must be run unconditionally")
    p.add_argument("--fold_path", default=None,
                   help="fold dir carrying CPD_NAME/SMILES, for --text_emb")
    p.add_argument("--pert_type", default=None,
                   help="perturbation type used when the prompts were "
                        "embedded. Auto-detected against the stored prompts; "
                        "pass it only if detection fails.")
    p.add_argument("--check_paths", type=int, default=200,
                   help="probe this many .npy paths on disk; 0 to skip")
    a = p.parse_args()

    man = pd.read_parquet(a.manifest)
    if a.dataset != "bbbc021" and "pert_role" not in man.columns:
        raise SystemExit(f"{a.manifest} has no pert_role column: it was built "
                         f"before perturbation_id.py, with the old "
                         f"{a.dataset} identity. Rebuild it.")
    if "pert_role" in man.columns and man["pert_role"].eq("positive_control").any():
        # CellFlux trains on neither side of these. They stay in the scorer's
        # universe, so cache_scores reports them as scored-but-absent (a WARN).
        n_pc = int(man["pert_role"].eq("positive_control").sum())
        man = man[~man["pert_role"].eq("positive_control")].reset_index(drop=True)
        print(f"[index] dropped {n_pc} positive-control crop(s), as CellFlux")
    control = a.control or CONTROL_OF[a.dataset]
    names = {c.strip().upper() for c in control.split(",")}
    is_ctrl = man["compound"].astype(str).str.upper().isin(names)

    # No unit_id on purpose: the scorer's contrast defines the unit (post-QC),
    # and the trainer takes unit_id from the score cache.
    idx = pd.DataFrame({
        "crop_id": man["SAMPLE_KEY"].astype(str),
        "path": [PATH_OF[a.dataset](a.img_dir, k) for k in man["SAMPLE_KEY"]],
        "y": (~is_ctrl).astype(int),
        "split": man["split"].astype(str) if "split" in man.columns else "all",
        "compound": man["compound"].astype(str),
        "dose": man["concentration"],
        "moa": man["moa"] if "moa" in man.columns else None,
        "plate": man["plate"].astype(str),
        "well": man["well_id"].astype(str),
    })
    if idx["crop_id"].duplicated().any():
        raise SystemExit(f"{int(idx['crop_id'].duplicated().sum())} duplicate "
                         f"crop_id in the manifest")

    if a.text_emb:
        if not a.fold_path:
            raise SystemExit("--text_emb needs --fold_path")
        idx = attach_text(idx, a.fold_path, a.text_emb, a.pert_type)

    if a.check_paths:
        rng = np.random.default_rng(0)
        probe = rng.choice(len(idx), min(a.check_paths, len(idx)), replace=False)
        miss = [idx["path"].iloc[i] for i in probe
                if not os.path.exists(idx["path"].iloc[i])]
        print(f"[paths] probed {len(probe)}, missing {len(miss)}"
              + (f"; e.g. {miss[0]}" if miss else ""))
        if miss:
            raise SystemExit("paths do not resolve -- check --img_dir")

    idx.to_parquet(a.out, index=False)
    n_t = int((idx.y == 1).sum())
    print(f"\n[index] {len(idx)} crops -> {a.out}")
    print(f"  treated {n_t}, control {len(idx) - n_t}")

    # The index holds every split, so cache_scores.py's crop-set check is a
    # total comparison; train_splits / control_splits are set in the config.
    print("\n  per split (the trainer selects from these, it does not get to "
          "invent them):")
    g = (idx.assign(cls=np.where(idx.y == 1, "treated", "control"))
         .groupby(["split", "cls"])
         .agg(crops=("crop_id", "size"), compounds=("compound", "nunique"),
              plates=("plate", "nunique")))
    print("    " + g.to_string().replace("\n", "\n    "))

    # Pairing needs a same-plate control, and controls are not spread evenly
    # across the split files: print the coverage of each control pool.
    print("\n  treated crops with no same-plate control, by control pool:")
    pools = {"train": {"train"}, "train+test (default)": {"train", "test"},
             "all splits": set(idx["split"].unique())}
    for sp in sorted(idx.loc[idx.y == 1, "split"].unique()):
        t = idx[(idx.y == 1) & (idx.split == sp)]
        line = []
        for nm, pool in pools.items():
            c = set(idx.loc[(idx.y == 0) & idx.split.isin(pool), "plate"])
            n = int((~t["plate"].isin(c)).sum())
            line.append(f"{nm} {n} ({100 * n / len(t):.1f}%)")
        print(f"    {sp:<9} " + ";  ".join(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
