"""Collapse a frozen scorer run into the score tables the flow trains against.

    python scorer/cache_scores.py --run_dir RUN --manifest MAN.parquet --out DIR

Reads a finished ``run_hidden.py`` output directory and writes:

    crop_scores.parquet    one row per treated crop, primary key ``crop_id``
    unit_scores.parquet    one row per unit (join on ``unit_id``)
    crop_universe.parquet  ``crop_id, y, kept`` for every crop, controls included

Each carries ``scorer_config_hash``, ``scorer_code_hash`` and
``data_index_hash`` in its parquet metadata. Controls get no score: a control
crop belongs to every contrast on its plate. ``s`` is the responder weight
(clipped to ``--s_lo/--s_hi``), ``h`` the out-of-fold posterior.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scorer import scorer_config as sc                                    # noqa: E402


# --------------------------------------------------------------------------- #
def write_stamped(df: pd.DataFrame, path: str, meta: Dict[str, str]) -> None:
    """Parquet + a provenance block in the file's key-value metadata."""
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    kv = {**(tbl.schema.metadata or {}),
          **{k.encode(): str(v).encode() for k, v in meta.items()}}
    pq.write_table(tbl.replace_schema_metadata(kv), path)


def read_stamp(path: str) -> Dict[str, str]:
    md = pq.read_schema(path).metadata or {}
    return {k.decode(): v.decode() for k, v in md.items()
            if not k.decode().startswith("pandas")}


# --------------------------------------------------------------------------- #
def load_run(run_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Every per-contrast score file, the unit manifest, and the run config."""
    cfg_path = os.path.join(run_dir, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}

    files = sorted(glob.glob(os.path.join(run_dir, "scores", "*.parquet")))
    if not files:
        raise SystemExit(f"no score parquets under {run_dir}/scores")
    parts = []
    for f in files:
        d = pd.read_parquet(f)
        # {compound}__{dose}__{plate-tag}.parquet: the file name is the unit_id
        d["unit_id"] = os.path.basename(f)[: -len(".parquet")]
        parts.append(d)
    scores = pd.concat(parts, ignore_index=True)

    umf_path = os.path.join(run_dir, "units_manifest.parquet")
    if not os.path.exists(umf_path):
        raise SystemExit(
            f"no units_manifest.parquet under {run_dir}. It is written by "
            f"run_hidden.py from v4 on; a run without it predates the fold "
            f"fix and must not be cached.")
    umf = pd.read_parquet(umf_path)
    print(f"[run] {len(files)} contrast(s), {len(scores)} scored crop-rows, "
          f"{len(umf)} attempted unit(s)")
    return scores, umf, cfg


def unit_table(run_dir: str, umf: pd.DataFrame) -> pd.DataFrame:
    """pi_hat / auroc / has_plateau per unit, from e1_e2_compounds.csv."""
    e1 = pd.read_csv(os.path.join(run_dir, "e1_e2_compounds.csv"))
    ptag = e1["plates"].astype(str).str.split("|").apply(
        lambda p: "-".join(p) if len(p) <= 3 else f"{len(p)}plates")
    e1["unit_id"] = (e1["compound"].astype(str) + "__"
                     + e1["concentration"].astype(str) + "__" + ptag
                     ).str.replace("/", "-", regex=False)
    keep = ["unit_id", "compound", "concentration", "plates", "moa", "auroc",
            "auroc_lo", "auroc_hi", "pi", "pi_lo", "pi_hi", "pi_spread",
            "has_plateau", "pi_from_auroc", "centroid_dist", "n_wells_t",
            "n_wells_c", "n_crops_t", "n_crops_c", "n_plates", "k",
            "panel"]
    u = e1[[c for c in keep if c in e1.columns]].rename(
        columns={"concentration": "dose", "pi": "pi_hat", "n_wells_t": "n_wells"})
    st = umf.set_index("unit_id")[["status", "fold_assignment_hash"]] \
        if "fold_assignment_hash" in umf.columns else umf.set_index("unit_id")[["status"]]
    return u.join(st, on="unit_id")


def recompute_qc(man: pd.DataFrame, covariates: str, control: str,
                 mode: str, q: float) -> pd.Series:
    """Replay the run's QC to recover the per-crop ``kept`` flag.

    The score files hold only the crops that passed QC and the manifest is
    pre-QC, so the mask is recomputed through ``run_hidden.qc_defect`` and then
    checked against the crops the run scored.
    """
    from scorer import run_hidden as rh

    cov = pd.read_parquet(covariates)
    m = man.merge(cov, on="SAMPLE_KEY", how="left")
    names = {c.strip().upper() for c in str(control).split(",")}
    m["is_control"] = m["compound"].astype(str).str.upper().isin(names)
    if mode == "none":
        return pd.Series(True, index=range(len(m)))
    keep, _ = rh.qc_defect(m, q, mode=mode)
    return pd.Series(keep, index=range(len(m)))


def clip_scores(trt: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Clip s into (0,1); the raw value is kept as ``s_raw``.

    ``0 ** gamma`` is 0, so an unclipped zero would delete a crop from training
    rather than down-weight it.
    """
    trt = trt.copy()
    trt["s_raw"] = trt["s"].to_numpy(float)
    trt["s"] = np.clip(trt["s_raw"], lo, hi)
    return trt


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run_dir", required=True,
                   help="a finished run_hidden.py output dir (has scores/, "
                        "units_manifest.parquet, e1_e2_compounds.csv)")
    p.add_argument("--manifest", required=True,
                   help="the crop manifest parquet the run used, for the "
                        "data_index_hash and the control crop ids")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--config", default=sc.DEFAULT, choices=sorted(sc.FROZEN))
    p.add_argument("--covariates", default=None,
                   help="the covariates parquet the run used, to replay its QC")
    p.add_argument("--s_lo", type=float, default=1e-3)
    p.add_argument("--s_hi", type=float, default=1 - 1e-3)
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing cache. Regenerating scores "
                        "mid-experiment invalidates every run that used them.")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    crop_path = os.path.join(a.out, "crop_scores.parquet")
    if os.path.exists(crop_path) and not a.force:
        old = read_stamp(crop_path)
        raise SystemExit(
            f"{crop_path} exists (config {old.get('scorer_config_hash')}, "
            f"data {old.get('data_index_hash')}). Scores are written once and "
            f"then read; --force to overwrite, and re-run every training job "
            f"that used them.")

    scores, umf, cfg = load_run(a.run_dir)
    man = pd.read_parquet(a.manifest).reset_index(drop=True)
    dhash = sc.data_index_hash(man["SAMPLE_KEY"])
    is_ctl = (man["compound"].astype(str).str.upper()
              .isin({c.strip().upper()
                     for c in str(cfg.get("control", "DMSO")).split(",")}))

    qc_dropped = None
    kept = pd.Series(True, index=man.index)
    cov = a.covariates or cfg.get("covariates")
    if cov and os.path.exists(cov):
        kept = recompute_qc(man, cov, cfg.get("control", "DMSO"),
                            str(cfg.get("qc", "optics")),
                            float(cfg.get("qc_q", 0.001)))
        drop = ~kept.to_numpy()
        qc_dropped = set(man.loc[drop, "SAMPLE_KEY"].astype(str))
        print(f"[qc] replayed --qc {cfg.get('qc')} q={cfg.get('qc_q')}: "
              f"{len(qc_dropped)} crop(s) dropped "
              f"({int((drop & is_ctl.to_numpy()).sum())} control, "
              f"{int((drop & ~is_ctl.to_numpy()).sum())} treated)")

    # Controls the flow may pair with: QC failures are excluded here too.
    ctl_ids = man.loc[is_ctl.to_numpy() & kept.to_numpy(), "SAMPLE_KEY"]
    print(f"[man] {len(man)} crops, {int(is_ctl.sum())} control "
          f"({len(ctl_ids)} after QC), data_index_hash {dhash}")

    # ---- treated rows ----------------------------------------------------- #
    trt = scores[scores["y"] == 1].copy()
    units = unit_table(a.run_dir, umf)
    meta_cols = ["unit_id", "compound", "dose", "moa"]
    trt = trt.merge(units[meta_cols], on="unit_id", how="left", validate="m:1")
    if trt["compound"].isna().any():
        raise SystemExit(f"{int(trt['compound'].isna().sum())} scored crop(s) "
                         f"join to no unit row -- the unit_id built from the "
                         f"score filename does not match e1_e2_compounds.csv")

    trt = clip_scores(trt, a.s_lo, a.s_hi)

    # ---- treated-crop coverage -------------------------------------------- #
    treated_universe = set(man.loc[~is_ctl.to_numpy(), "SAMPLE_KEY"].astype(str))
    scored = set(trt["crop_id"])
    unscored = treated_universe - scored
    by_qc = unscored & (qc_dropped or set())
    print(f"\n== coverage ==\n  treated crops in the manifest  {len(treated_universe)}"
          f"\n  with a cached score            {len(scored)} "
          f"({100 * len(scored) / max(len(treated_universe), 1):.2f}%)"
          f"\n  dropped by QC before scoring   {len(by_qc)}"
          f"\n  unscored for any OTHER reason  {len(unscored) - len(by_qc)}")
    rest = unscored - by_qc
    if rest:
        u = man[man["SAMPLE_KEY"].isin(rest)]
        by = u.groupby(["compound", "concentration"], dropna=False).size()
        print(f"  the latter in {len(by)} unit(s), largest: "
              + ", ".join(f"{c}@{d} ({n})" for (c, d), n in
                          by.sort_values(ascending=False).head(5).items()))

    # ---- write ------------------------------------------------------------ #
    stamp = {**sc.stamp(a.config), "data_index_hash": dhash,
             "run_dir": a.run_dir, "s_lo": a.s_lo, "s_hi": a.s_hi,
             "n_treated_crops": len(trt), "n_units": len(units),
             "n_control_crops": len(ctl_ids),
             "n_qc_dropped": len(qc_dropped) if qc_dropped is not None else None}
    cols = ["crop_id", "unit_id", "compound", "dose", "moa", "plate", "well_id",
            "field_id", "y", "s", "h", "fold", "k", "s_raw"]
    trt = trt[[c for c in cols if c in trt.columns]].sort_values("crop_id")
    write_stamped(trt, crop_path, stamp)
    write_stamped(units, os.path.join(a.out, "unit_scores.parquet"), stamp)
    # Every crop, its class and its QC verdict (the trainer drops QC failures).
    universe = pd.DataFrame({
        "crop_id": man["SAMPLE_KEY"].astype(str),
        "y": (~is_ctl).astype(int),
        "kept": kept.to_numpy().astype(bool),
    }).sort_values("crop_id")
    write_stamped(universe, os.path.join(a.out, "crop_universe.parquet"), stamp)
    json.dump(stamp, open(os.path.join(a.out, "provenance.json"), "w"), indent=2)

    print(f"\n[cache] -> {a.out}")
    for k, v in stamp.items():
        print(f"  {k:<20} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
