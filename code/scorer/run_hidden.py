"""Per-crop responder scorer on stored embeddings. CPU only.

    python scorer/run_hidden.py --dataset bbbc021 --embedding morphem --n_pcs 50 \
        --fold_path $FOLDS --covariates $COV
    python scorer/run_hidden.py --dataset bbbc021 --embedding cellclip_nochan --no_pca ...
    python scorer/run_hidden.py --dataset rxrx1 --embedding morphem --pooling per_plate ...

Per unit it writes the out-of-fold posterior ``h`` and responder weight ``s``
of every crop (``scores/*.parquet``), and the unit's AUROC and responder
fraction ``pi`` with well-bootstrap intervals (``e1_e2_compounds.csv``).

The analysis unit is a compound-concentration when a concentration is
available and a compound otherwise. Controls are taken from every plate the
unit pools, so plate identity is balanced across the classes.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scorer import core
from scorer.manifest import (IDENTITY_TAG, EmbeddingStore, build_manifest,
                      verify_against_h5)

EMBEDDINGS = {
    "dinov2g": "dinov2g_ind.h5",
    "morphem": "morphem_ind.h5",
    "cellclip_map": "cellclip_map_ind.h5",
    "cellclip_nochan": "cellclip_nochan_ind.h5",
}
TIMES: Dict[str, float] = {}          # wall clock per stage, printed at the end


def fmt_dt(sec: float) -> str:
    if sec < 90:
        return f"{sec:.1f}s"
    m, r = divmod(int(sec), 60)
    return f"{m}m{r:02d}s" if m < 60 else f"{m // 60}h{m % 60:02d}m"


@contextmanager
def stage(name: str):
    """Time a block, print it, and add it to the end-of-run summary."""
    t0 = time.time()
    yield
    add_time(name, time.time() - t0)
    print(f"[time] {name}  {fmt_dt(TIMES[name])}")


def add_time(name: str, dt: float) -> None:
    TIMES[name] = TIMES.get(name, 0.0) + dt


# BBBC021 strong-effect compounds (cytoskeletal).
STRONG = ("demecolcine", "taxol", "vincristine", "cytochalasin d",
          "cytochalasin b", "latrunculin b")

# Defaults for --control. CPG and RxRx1 controls are relabelled CONTROL by
# manifest.py (see perturbation_id.py).
CONTROLS = {"bbbc021": "DMSO", "cpg": "CONTROL", "rxrx1": "CONTROL"}


def panel_of(tbl: pd.DataFrame, dataset: str, n: int) -> pd.Series:
    """Label units ``strong`` / ``weak`` in the output table.

    BBBC021 uses the named strong panel; elsewhere strong is the top of the
    control-centroid-distance ranking. Weak is the bottom of that ranking on
    every dataset.
    """
    lab = pd.Series("", index=tbl.index)
    # Rank units, not rows: under --pooling per_plate one unit has a row per plate.
    unit = tbl["compound"].astype(str) + "|" + tbl["concentration"].astype(str)
    d = tbl.groupby(unit)["centroid_dist"].mean()
    named = (tbl["compound"].astype(str).str.lower().isin(STRONG)
             if dataset == "bbbc021" else pd.Series(False, index=tbl.index))
    if named.any():
        lab[named] = "strong"
    elif d.notna().any():
        r = d.rank(method="first", ascending=False)
        lab[unit.isin(r[r <= n].index)] = "strong"
    r = d.rank(method="first")
    lab[(lab == "") & unit.isin(r[r <= n].index)] = "weak"
    return lab


# ===================================================================== #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, choices=("bbbc021", "cpg", "rxrx1"))
    p.add_argument("--embedding", required=True,
                   help="one of " + "/".join(EMBEDDINGS) + ", or a path to an h5")
    p.add_argument("--emb_dir",
                   default="/path/to/workdir/results/iclr/embeddings")
    p.add_argument("--fold_path", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--out_dir", default="results/hidden")

    p.add_argument("--image_csv", help="BBBC021_v1_image.csv -- gives well AND "
                                       "concentration. Without it the unit of "
                                       "analysis degrades to the field.")
    p.add_argument("--moa_csv", help="BBBC021_v1_moa.csv (per compound-concentration)")
    p.add_argument("--covariates", help="parquet from covariates.py: "
                                        "foreground_frac, dna_integrated")
    p.add_argument("--control", default=None,
                   help="control label(s), comma separated. Default per "
                        "dataset: " + ", ".join(f"{k}={v}" for k, v in
                                                CONTROLS.items()))

    p.add_argument("--n_pcs", type=int, default=50)
    p.add_argument("--no_pca", action="store_true",
                   help="fit the logistic regression on the scaled features "
                        "directly. Cheap at D=512 (CellCLIP), slow at D=4608.")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--unit", default="auto",
                   choices=("auto", "compound", "compound_dose"))
    p.add_argument("--pooling", default="pooled", choices=("pooled", "per_plate"))
    p.add_argument("--min_treated_wells", type=int, default=3)
    p.add_argument("--min_treated_crops", type=int, default=50)
    p.add_argument("--max_control_crops", type=int, default=8000,
                   help="per contrast, subsampled by whole wells")
    p.add_argument("--limit_units", type=int, default=None)
    p.add_argument("--units", help="comma-separated compound names to run. "
                                   "The way to re-run a handful of units under "
                                   "--pooling per_plate after a pooled pass.")
    p.add_argument("--panel_n", type=int, default=5,
                   help="size of each tail of the centroid-distance panel, "
                        "used where there is no MoA list")
    p.add_argument("--max_cache_gb", type=float, default=8.0,
                   help="RAM for the control block. Above it, control groups "
                        "are read per contrast instead of held.")
    p.add_argument("--folds", default="lowo", choices=("lowo", "stratified"),
                   help="lowo: when the minority class has <=3 wells, assign "
                        "them leave-one-out and deterministically, k = that "
                        "count, controls dealt round-robin. Seed-independent, "
                        "and the same rotation E5r uses. stratified reproduces "
                        "v3's StratifiedGroupKFold(shuffle=True).")
    p.add_argument("--qc", default="optics",
                   choices=("optics", "defect", "foreground", "none"),
                   help="optics: saturation / illumination-ramp / unreadable "
                        "only -- the criteria that measure the OPTICS. This is "
                        "the frozen v4 setting. defect adds focus_ch* and "
                        "plate-relative intensity, which v4 measured to be "
                        "phenotype detectors on BBBC021 (focus_ch1 dropped "
                        "3151 treated crops against 1 for focus_ch2; real "
                        "defocus is not channel-confined), and failed the 3 pp "
                        "acceptance rule at 8.1 pp. foreground reproduces v3's "
                        "foreground_frac filter, which is a phenotype and "
                        "deletes responders. none keeps every crop and is the "
                        "sensitivity check reported beside optics.")
    p.add_argument("--qc_q", type=float, default=0.001,
                   help="fixed low quantile of the CONTROL distribution that "
                        "sets each criterion's threshold. Not tuned.")
    p.add_argument("--no_figures", action="store_true")
    return p.parse_args()


def n_pcs_of(args) -> Optional[int]:
    return None if args.no_pca else args.n_pcs


def tag_of(args) -> str:
    """Output subdirectory: embedding, PCA setting and QC mode (the QC mode
    changes the control set, hence every AUROC)."""
    name = os.path.basename(args.embedding).replace(".h5", "")
    return (f"{name}_" + ("nopca" if args.no_pca else f"pca{args.n_pcs}")
            + f"_qc{args.qc}")


# ===================================================================== #
# Manifest + QC
# ===================================================================== #
# Which criterion families each --qc mode applies. ``optics`` keeps the
# statistics that measure the instrument, not the specimen.
QC_FAMILIES = {
    "optics": ("satfrac", "illum_r2", "unreadable"),
    "defect": ("focus", "satfrac", "illum_r2", "absz_mean", "unreadable"),
}


def qc_defect(man: pd.DataFrame, q: float, mode: str = "optics",
              ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Defect-only QC: imaging failure, never cell state.

    ``mode="optics"`` applies saturation, the illumination ramp and the
    unreadable-crop rule. ``mode="defect"`` adds per-channel Laplacian focus
    and plate-relative intensity, which also respond to the phenotype
    (cytoskeletal compounds remove high-frequency texture and change staining
    intensity). ``foreground_frac`` is not a criterion for the same reason.

    Every threshold is a quantile of the control crops only, applied as one
    absolute number to both classes. Returns (keep mask, per-criterion report).
    """
    fams = QC_FAMILIES[mode]
    ctl = man["is_control"].to_numpy()
    keep = np.ones(len(man), dtype=bool)
    rep = []

    def apply(col, thr, bad, why):
        if thr is None or not np.isfinite(thr):
            return
        m = bad(man[col].to_numpy(), thr)
        m &= man[col].notna().to_numpy()          # unreadable crops: see below
        rep.append({"criterion": col, "rule": why, "threshold": float(thr),
                    "n_drop": int(m.sum()),
                    "n_drop_control": int((m & ctl).sum()),
                    "n_drop_treated": int((m & ~ctl).sum()),
                    "rate_control": float((m & ctl).sum() / max(ctl.sum(), 1)),
                    "rate_treated": float((m & ~ctl).sum() / max((~ctl).sum(), 1))})
        keep[m] = False

    chans = sorted(int(c[len("focus_ch"):]) for c in man.columns
                   if c.startswith("focus_ch"))
    for i in (chans if "focus" in fams else []):
        c = f"focus_ch{i}"
        apply(c, man.loc[ctl, c].quantile(q), lambda v, t: v < t, "low = blurred")
    for i in (sorted(int(c[len("satfrac_ch"):]) for c in man.columns
                     if c.startswith("satfrac_ch")) if "satfrac" in fams else []):
        c = f"satfrac_ch{i}"
        apply(c, man.loc[ctl, c].quantile(1 - q), lambda v, t: v > t,
              "high = saturated")
    for i in (sorted(int(c[len("illum_r2_ch"):]) for c in man.columns
                     if c.startswith("illum_r2_ch")) if "illum_r2" in fams else []):
        c = f"illum_r2_ch{i}"
        apply(c, man.loc[ctl, c].quantile(1 - q), lambda v, t: v > t,
              "high = illumination ramp")

    # Plate-relative intensity: z against that plate's own controls; the cut on
    # |z| is one number taken from the pooled control |z|.
    for i in (sorted(int(c[len("mean_ch"):]) for c in man.columns
                     if c.startswith("mean_ch") and c[len("mean_ch"):].isdigit())
              if "absz_mean" in fams else []):
        c = f"mean_ch{i}"
        z = np.full(len(man), np.nan)
        for _, idx in man.groupby("plate").indices.items():
            ref = man[c].to_numpy()[idx][ctl[idx]]
            ref = ref[np.isfinite(ref)]
            if len(ref) < 8:
                continue
            m0 = np.median(ref)
            s0 = 1.4826 * np.median(np.abs(ref - m0))
            z[idx] = (man[c].to_numpy()[idx] - m0) / (s0 + 1e-8)
        az = np.abs(z)
        man = man.assign(**{f"_absz_{c}": az})
        apply(f"_absz_{c}", np.nanquantile(az[ctl], 1 - q), lambda v, t: v > t,
              "plate-relative intensity outlier")

    # A crop with no covariate row was unreadable on disk.
    if chans and "unreadable" in fams:
        miss = man[f"focus_ch{chans[0]}"].isna().to_numpy()
        if miss.any():
            rep.append({"criterion": "unreadable", "rule": "no covariate row",
                        "threshold": np.nan, "n_drop": int(miss.sum()),
                        "n_drop_control": int((miss & ctl).sum()),
                        "n_drop_treated": int((miss & ~ctl).sum()),
                        "rate_control": float((miss & ctl).sum() / max(ctl.sum(), 1)),
                        "rate_treated": float((miss & ~ctl).sum() / max((~ctl).sum(), 1))})
            keep[miss] = False
    return keep, pd.DataFrame(rep)


def load_manifest(args, h5_path: str, out: str) -> pd.DataFrame:
    # The split directory is part of the cache key: a different split yields
    # different pkey/row addressing.
    split_dir = os.path.basename(os.path.normpath(args.fold_path))
    cache = os.path.join(args.out_dir,
                         f"manifest_{args.dataset}_f{args.fold}_{split_dir}"
                         + IDENTITY_TAG.get(args.dataset, "")
                         + ("_img" if args.image_csv else "") + ".parquet")
    if os.path.exists(cache):
        man = pd.read_parquet(cache)
        print(f"[man] cached {cache}  ({len(man)} crops)")
    else:
        man = build_manifest(args.dataset, args.fold_path, args.fold,
                             args.image_csv, args.moa_csv)
        os.makedirs(args.out_dir, exist_ok=True)
        man.to_parquet(cache, index=False)
    man = verify_against_h5(man, h5_path)

    if args.covariates:
        cov = pd.read_parquet(args.covariates)
        man = man.merge(cov, on="SAMPLE_KEY", how="left")
        print(f"[man] covariates: "
              f"{[c for c in cov.columns if c != 'SAMPLE_KEY']}")

    names = {c.strip().upper() for c in args.control.split(",")}
    man["is_control"] = man["compound"].astype(str).str.upper().isin(names)
    print(f"[man] {len(man)} crops, {int(man['is_control'].sum())} control "
          f"({'/'.join(sorted(names))}), {man['plate'].nunique()} plates, "
          f"{man['well_id'].nunique()} wells")

    # ``kept`` is a column, not a deletion: the unit manifest needs the pre-QC
    # denominators.
    rep = pd.DataFrame()
    if args.qc in ("optics", "defect"):
        if not any(c.startswith("focus_ch") for c in man.columns):
            raise SystemExit(
                f"--qc {args.qc} needs the defect covariates (focus_ch*, "
                "satfrac_ch*, illum_r2_ch*, mean_ch*). Re-run covariates.py "
                "-- the v3 parquet predates them -- and pass --covariates. "
                "focus_ch* is read by --qc optics too: not as a criterion, "
                "but as the unreadable-crop flag.")
        keep, rep = qc_defect(man, args.qc_q, mode=args.qc)
    elif args.qc == "foreground":
        if "foreground_frac" not in man.columns:
            raise SystemExit("--qc foreground needs foreground_frac")
        tau = man.loc[man["is_control"], "foreground_frac"].quantile(0.05)
        keep = (man["foreground_frac"] >= tau).to_numpy()
        print(f"[qc] foreground tau={tau:.4f}")
    else:
        keep = np.ones(len(man), dtype=bool)
    man["kept"] = keep

    ctl = man["is_control"].to_numpy()
    r_c = float((~keep & ctl).sum() / max(ctl.sum(), 1))
    r_t = float((~keep & ~ctl).sum() / max((~ctl).sum(), 1))
    print(f"[qc] mode={args.qc} q={args.qc_q}: dropped control {r_c * 100:.2f}%, "
          f"treated {r_t * 100:.2f}%  (gap {abs(r_t - r_c) * 100:.2f} pp)")
    if len(rep):
        print(rep.to_string(index=False))
        rep.to_csv(os.path.join(out, "qc_report.csv"), index=False)
    return man


def units_of(args, man: pd.DataFrame) -> List[Dict]:
    """Analysis units: compound, or compound-concentration when dose is known."""
    use_dose = (args.unit == "compound_dose" or
                (args.unit == "auto" and man["concentration"].notna().any()))
    trt = man[~man["is_control"]]
    keys = ["compound", "concentration"] if use_dose else ["compound"]
    print(f"[units] keyed by {keys}")
    out = []
    for k, sub in trt.groupby(keys, dropna=False):
        k = k if isinstance(k, tuple) else (k,)
        out.append({"compound": k[0],
                    "concentration": float(k[1]) if use_dose and pd.notna(k[1])
                    else np.nan,
                    "idx": sub.index.to_numpy()})
    out.sort(key=lambda u: (str(u["compound"]), u["concentration"]))
    if args.units:
        want = {c.strip().lower() for c in args.units.split(",")}
        out = [u for u in out if str(u["compound"]).lower() in want]
        missing = want - {str(u["compound"]).lower() for u in out}
        print(f"[units] --units kept {len(out)}"
              + (f"; no such compound: {sorted(missing)}" if missing else ""))
    return out


def subsample_controls(ctrl: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Drop whole control wells until under the cap, evenly across plates."""
    if len(ctrl) <= cap:
        return ctrl
    rng = np.random.default_rng(seed)
    per_plate = {p: rng.permutation(s["well_id"].unique())
                 for p, s in ctrl.groupby("plate")}
    keep, n = [], 0
    for i in range(max(len(v) for v in per_plate.values())):
        for p, wells in per_plate.items():
            if i >= len(wells):
                continue
            w = wells[i]
            take = ctrl[ctrl["well_id"] == w]
            if n + len(take) > cap and n > 0 and len(keep) >= 2 * len(per_plate):
                continue
            keep.append(take)
            n += len(take)
        if n >= cap:
            break
    return pd.concat(keep).sort_index()


def unit_manifest_row(args, man_pre: pd.DataFrame, u: Dict, pl: List,
                      label: Dict, res: Dict) -> Dict:
    """One row per attempted unit: pre/post-QC counts, fold info, status and
    skip reason."""
    on = man_pre[man_pre["plate"].isin(pl)]
    t = on[(~on["is_control"]) & (on["compound"] == u["compound"])]
    if pd.notna(u["concentration"]):
        t = t[t["concentration"] == u["concentration"]]
    c = on[on["is_control"]]
    tk, ck = t[t["kept"]], c[c["kept"]]
    moa = (t["moa"].dropna().iloc[0] if "moa" in t.columns
           and t["moa"].notna().any() else None)
    return {
        "unit_id": f"{u['compound']}__{u['concentration']}__"
                   f"{'-'.join(map(str, pl))}".replace("/", "-"),
        "compound": u["compound"], "dose": u["concentration"], "moa": moa,
        "plates": label["plates"],
        "n_wells_pre_qc": int(t["well_id"].nunique()),
        "n_wells_post_qc": int(tk["well_id"].nunique()),
        "n_crops_pre_qc_treated": int(len(t)),
        "n_crops_post_qc_treated": int(len(tk)),
        "n_crops_pre_qc_control": int(len(c)),
        "n_crops_post_qc_control": int(len(ck)),
        "retention_treated": len(tk) / max(len(t), 1),
        "retention_control": len(ck) / max(len(c), 1),
        "k": res.get("row", {}).get("k"),
        "seed": args.seed,
        "folds_mode": args.folds,
        "fold_assignment_hash": res.get("row", {}).get("fold_assignment_hash"),
        "status": res.get("status", "scored"),
        "skip_reason": res.get("skip_reason"),
    }


# ===================================================================== #
# One contrast
# ===================================================================== #
def run_compound(store: EmbeddingStore, trt: pd.DataFrame, ctrl: pd.DataFrame,
                 args, label: Dict) -> Optional[Dict]:
    """Score one contrast set and compute everything that lives inside it."""
    n_wt = trt["well_id"].nunique()
    if n_wt < args.min_treated_wells:
        return {"status": "skipped_well_filter",
                "skip_reason": f"{n_wt} treated well(s) < --min_treated_wells "
                               f"{args.min_treated_wells}"}
    if len(trt) < args.min_treated_crops:
        return {"status": "skipped_crop_filter",
                "skip_reason": f"{len(trt)} treated crop(s) < "
                               f"--min_treated_crops {args.min_treated_crops}"}
    ctrl = subsample_controls(ctrl, args.max_control_crops, args.seed)
    if ctrl["well_id"].nunique() < 2:
        return {"status": "skipped_well_filter",
                "skip_reason": f"{ctrl['well_id'].nunique()} control well(s) "
                               f"on these plates"}

    t_load = time.time()
    both = pd.concat([trt, ctrl]).reset_index(drop=True)
    y = (~both["is_control"]).to_numpy().astype(int)
    wells = both["well_id"].to_numpy()
    X = store.matrix(both)

    tm = {"load": time.time() - t_load}

    # RMS shift between the treated and control centroids, in control-MAD
    # units. Never a classifier input; only used to pick the strong/weak panel.
    m0, s0 = core._robust_stats(X[y == 0])
    z0 = (np.median(X[y == 1], axis=0) - m0) / (s0 + 1e-8)
    centroid_dist = float(np.sqrt(np.mean(z0 ** 2)))

    t0 = time.time()
    try:
        folds = core.make_folds(y, wells, args.n_splits, args.seed,
                                mode=args.folds)
    except core.SingleClassFold as e:
        print(f"  [skip] {label} : {e}")
        return {"status": "failed_fold_single_class", "skip_reason": str(e)}
    except ValueError as e:
        print(f"  [skip] {label} : {e}")
        return {"status": "failed_fold_single_class", "skip_reason": str(e)}
    h = core.hidden_scores(X, y, wells, n_pcs=n_pcs_of(args),
                           n_splits=args.n_splits, C=args.C, seed=args.seed,
                           folds_mode=args.folds, folds=folds)
    tm["fit"] = time.time() - t0

    a = core.auroc(y, h)
    ph = core.pi_hat(h[y == 1], h[y == 0])
    t0 = time.time()
    ci = (core.well_bootstrap(h, y, wells, B=args.bootstrap, seed=args.seed)
          if args.bootstrap else {"auroc": (np.nan, np.nan), "pi": (np.nan, np.nan)})
    tm["bootstrap"] = time.time() - t0
    # class_weight="balanced" already rewrites the prior, so prior_ratio is 1.
    s = core.responder_weight(h, ph["pi"], prior_ratio=1.0)
    thr = float(np.quantile(h[y == 0], 0.95))

    row = dict(label)
    row.update({
        "k": len(folds), "fold_assignment_hash": core.fold_hash(wells, folds),
        "n_crops_t": int(y.sum()), "n_crops_c": int((1 - y).sum()),
        "n_wells_t": int(n_wt), "n_wells_c": int(ctrl["well_id"].nunique()),
        "n_plates": int(both["plate"].nunique()),
        "centroid_dist": centroid_dist,
        "auroc": a, "auroc_lo": ci["auroc"][0], "auroc_hi": ci["auroc"][1],
        "pi": ph["pi"], "pi_lo": ci["pi"][0], "pi_hi": ci["pi"][1],
        "pi_spread": ph["spread"], "has_plateau": ph["has_plateau"],
        "pi_from_auroc": core.pi_from_auroc(a),
        "mean_score_treated": float(h[y == 1].mean()),
        "mean_score_all": float(h.mean()),
        "frac_responder": float((s[y == 1] > 0).mean()),
    })

    # Which test fold each crop landed in, written per crop so the score cache
    # can verify that every score is out-of-fold.
    fold_of = np.full(len(y), -1, dtype=np.int16)
    for fi, (_, te) in enumerate(folds):
        fold_of[te] = fi
    assert (fold_of >= 0).all(), "a crop landed in no test fold"

    scores = pd.DataFrame({
        "crop_id": both["SAMPLE_KEY"].to_numpy(), "plate": both["plate"].to_numpy(),
        "well_id": wells, "field_id": both["field_id"].to_numpy(),
        "y": y, "h": h, "s": s, "fold": fold_of, "k": np.int16(len(folds)),
    })
    wells_tbl = (scores[scores.y == 1].groupby("well_id")
                 .agg(n_crops=("h", "size"), mean_score=("h", "mean"),
                      high_frac=("h", lambda v: float((v > thr).mean())),
                      mean_s=("s", "mean")).reset_index())
    for k, v in label.items():
        wells_tbl[k] = v

    extras = {"curve": pd.DataFrame({**label, "q": ph["qs"], "pi_q": ph["curve"]}),
              "wells": wells_tbl, "scores": scores}

    row.update({f"sec_{k}": round(v, 1) for k, v in tm.items()})
    row["sec"] = round(sum(tm.values()), 1)
    return {"row": row, "extras": extras, "times": tm, "status": "scored"}


# ===================================================================== #
def add_edge_dist(man: pd.DataFrame) -> pd.DataFrame:
    """Distance of the well from the plate edge, in wells (spatial artefact
    covariate)."""
    if man["well_row"].notna().any():
        r, c = man["well_row"], man["well_col"]
        man["edge_dist"] = np.minimum.reduce([
            r - r.min(), r.max() - r, c - c.min(), c.max() - c]).astype(float)
    return man


def report_timing(out: str, t_run: float,
                  tbl: Optional[pd.DataFrame] = None) -> None:
    """Where the wall clock went. Also written to timings.csv."""
    total = time.time() - t_run
    print("\n== timing ==")
    for k, v in sorted(TIMES.items(), key=lambda kv: -kv[1]):
        if v >= 0.05:
            print(f"  {k:<20} {fmt_dt(v):>9}   {100 * v / total:4.1f}%")
    print(f"  {'TOTAL':<20} {fmt_dt(total):>9}")
    if tbl is not None and len(tbl):
        print(f"  per contrast: median {fmt_dt(tbl['sec'].median())}, "
              f"max {fmt_dt(tbl['sec'].max())}, over {len(tbl)} contrast(s)")
    pd.DataFrame([{"stage": k, "seconds": round(v, 2),
                   "pct": round(100 * v / total, 1)}
                  for k, v in sorted(TIMES.items(), key=lambda kv: -kv[1])]
                 + [{"stage": "TOTAL", "seconds": round(total, 2), "pct": 100.0}]
                 ).to_csv(os.path.join(out, "timings.csv"), index=False)


def main():
    t_run = time.time()
    args = parse_args()
    args.control = args.control or CONTROLS[args.dataset]
    h5_path = (args.embedding if args.embedding.endswith(".h5") else
               os.path.join(args.emb_dir, args.dataset,
                            EMBEDDINGS[args.embedding]))
    out = os.path.join(args.out_dir, args.dataset, tag_of(args))
    os.makedirs(os.path.join(out, "scores"), exist_ok=True)
    print(f"[run] {h5_path}\n[run] -> {out}\n[run] "
          f"{'NO PCA' if args.no_pca else f'PCA K={args.n_pcs}'}")

    with stage("manifest"):
        man_pre = load_manifest(args, h5_path, out)
        man_pre = add_edge_dist(man_pre)
    # Everything downstream sees only the crops that passed QC; ``man_pre``
    # is kept for the unit manifest's pre-QC denominators.
    man = man_pre[man_pre["kept"]].reset_index(drop=True)
    ctrl_all = man[man["is_control"]]
    if not len(ctrl_all):
        top = (man.groupby("compound").agg(crops=("row", "size"),
                                           wells=("well_id", "nunique"),
                                           plates=("plate", "nunique"))
               .sort_values("crops", ascending=False).head(15))
        raise SystemExit(f"no crops with compound in {args.control!r}. Pass "
                         f"--control with one of the labels below:\n"
                         + top.to_string())
    with stage("load controls"):
        store = EmbeddingStore(h5_path,
                               cache_keys=sorted(ctrl_all["pkey"].unique()),
                               max_cache_gb=args.max_cache_gb)
    json.dump({**vars(args),
               "h5": h5_path, "dim": store.dim, "attrs": {
                   k: str(v) for k, v in store.attrs.items()}},
              open(os.path.join(out, "config.json"), "w"), indent=2)

    # ---- AUROC and pi per unit ------------------------------------------ #
    units = units_of(args, man)
    if args.limit_units:
        units = units[: args.limit_units]
    print(f"\n== scoring {len(units)} unit(s) ==")
    if args.pooling == "pooled":
        span = np.median([man.loc[u["idx"], "plate"].nunique() for u in units])
        if span > 4:
            print(f"  note: the median unit spans {span:.0f} plates and they "
                  f"are pooled into one contrast, so the robust z is taken "
                  f"over all of them. --pooling per_plate keeps them separate.")

    rows, skipped, umf = [], [], []
    acc = {k: [] for k in ("curve", "wells")}
    t_loop = time.time()
    for i, u in enumerate(units, 1):
        trt_all = man.loc[u["idx"]]
        plates = sorted(trt_all["plate"].unique())
        groups = ([(plates, trt_all)] if args.pooling == "pooled"
                  else [([p], g) for p, g in trt_all.groupby("plate")])
        for pl, trt in groups:
            ctrl = man[man["is_control"] & man["plate"].isin(pl)]
            label = {"compound": u["compound"], "concentration": u["concentration"],
                     "plates": "|".join(map(str, pl))}
            res = run_compound(store, trt, ctrl, args, label)
            umf.append(unit_manifest_row(args, man_pre, u, pl, label, res))
            if "row" not in res:
                skipped.append({**label, **res})
                continue
            rows.append(res["row"])
            for k, v in res["times"].items():
                add_time({"load": "read h5", "fit": "fit",
                          "bootstrap": "bootstrap"}.get(k, k), v)
            for k, v in res["extras"].items():
                if k == "scores":
                    # a long joined plate list overruns the 255-byte filename cap
                    ptag = ("-".join(map(str, pl)) if len(pl) <= 3
                            else f"{len(pl)}plates")
                    name = (f"{u['compound']}__{u['concentration']}__"
                            f"{ptag}").replace("/", "-")
                    v.to_parquet(os.path.join(out, "scores", name + ".parquet"),
                                 index=False)
                else:
                    acc[k].append(v)
            r = res["row"]
            done = time.time() - t_loop
            eta = done / i * (len(units) - i)
            print(f"  [{i}/{len(units)}] {str(u['compound'])[:28]:<28} "
                  f"c={u['concentration']:<7.3g} auroc {r['auroc']:.3f} "
                  f"[{r['auroc_lo']:.2f},{r['auroc_hi']:.2f}]  pi {r['pi']:.3f} "
                  f"(spread {r['pi_spread']:.2f}{'' if r['has_plateau'] else ' NO PLATEAU'})"
                  f"  mean_h {r['mean_score_treated']:.3f}  "
                  f"{r['n_wells_t']}w  {fmt_dt(r['sec'])} "
                  f"(read {fmt_dt(r.get('sec_load', 0))} "
                  f"fit {fmt_dt(r.get('sec_fit', 0))} "
                  f"boot {fmt_dt(r.get('sec_bootstrap', 0))})"
                  f"  eta {fmt_dt(eta)}")

    seen = {(r["compound"], r["dose"], r["plates"]) for r in umf}
    for u in units_of(args, man_pre):
        trt_all = man_pre.loc[u["idx"]]
        pl = sorted(trt_all["plate"].unique())
        label = {"compound": u["compound"], "concentration": u["concentration"],
                 "plates": "|".join(map(str, pl))}
        if (u["compound"], u["concentration"], label["plates"]) in seen:
            continue
        umf.append(unit_manifest_row(
            args, man_pre, u, pl, label,
            {"status": "skipped_crop_filter",
             "skip_reason": "every crop of this unit was dropped by QC"}))
    if not umf:
        print("[units_manifest] no unit was even attempted")
        umf = [{"unit_id": None, "status": "none_attempted"}]
    pd.DataFrame(umf).to_parquet(
        os.path.join(out, "units_manifest.parquet"), index=False)
    ustat = pd.DataFrame(umf)["status"].value_counts()
    print(f"[units_manifest] {len(umf)} attempted unit(s) -> "
          + ", ".join(f"{k} {v}" for k, v in ustat.items()))
    lost = pd.DataFrame(umf)
    lost = (lost[lost["n_wells_post_qc"] < lost["n_wells_pre_qc"]]
            if "n_wells_post_qc" in lost.columns else lost.iloc[:0])
    if len(lost):
        print(f"  ! {len(lost)} unit(s) lost a WHOLE treated well to QC -- a "
              f"different mechanism from fold assignment:")
        print(lost[["unit_id", "n_wells_pre_qc", "n_wells_post_qc",
                    "retention_treated", "status"]].to_string(index=False))
    else:
        print("  no unit lost a whole treated well to QC")

    if skipped:
        print(f"  [skipped] {len(skipped)} contrast(s) under "
              f"--min_treated_wells {args.min_treated_wells} / "
              f"--min_treated_crops {args.min_treated_crops}: "
              + ", ".join(str(s_["compound"]) for s_ in skipped[:8]))
    tbl = pd.DataFrame(rows)
    if not len(tbl):
        print("no unit met --min_treated_wells / --min_treated_crops")
        store.close()
        report_timing(out, t_run)
        return
    for k, path in (("curve", "e2_pi_curves.csv"), ("wells", "wells.csv")):
        if acc[k]:
            pd.concat(acc[k], ignore_index=True).to_csv(
                os.path.join(out, path), index=False)

    # ---- by mechanism ----------------------------------------------------- #
    t_e3 = time.time()
    if True:
        moa_col = ("moa_ref" if "moa_ref" in man.columns else
                   "moa" if "moa" in man.columns else None)
        if moa_col:
            key = ["compound", "concentration"] if man["concentration"].notna().any() \
                else ["compound"]
            lut = (man[~man["is_control"]].groupby(key)[moa_col]
                   .agg(lambda v: v.dropna().iloc[0] if v.notna().any() else None))
            tbl["moa"] = pd.MultiIndex.from_frame(tbl[key]).map(lut) \
                if len(key) > 1 else tbl["compound"].map(lut)
            e3 = (tbl.dropna(subset=["moa"]).groupby("moa")
                  .agg(n=("pi", "size"), pi_mean=("pi", "mean"),
                       pi_min=("pi", "min"), pi_max=("pi", "max"),
                       auroc_mean=("auroc", "mean"),
                       plateau_frac=("has_plateau", "mean"))
                  .sort_values("pi_mean", ascending=False).reset_index())
            e3.to_csv(os.path.join(out, "e3_mechanism.csv"), index=False)
            print("\n== by mechanism ==")
            print(e3.to_string(index=False))
    add_time("by mechanism", time.time() - t_e3)

    # ---- dose ------------------------------------------------------------ #
    t_e4 = time.time()
    if tbl["concentration"].notna().any():
        e4 = (tbl.dropna(subset=["concentration"])
              .sort_values(["compound", "concentration"])
              [["compound", "concentration", "auroc", "pi", "pi_lo", "pi_hi",
                "pi_spread", "has_plateau", "n_wells_t"]])
        e4["pi_delta"] = e4.groupby("compound")["pi"].diff()
        e4.to_csv(os.path.join(out, "e4_dose.csv"), index=False)
        print(f"\n== dose == {int((e4.groupby('compound').size() > 1).sum())}"
              f"/{e4['compound'].nunique()} compound(s) have more than one dose")
    add_time("dose", time.time() - t_e4)

    # the MoA column is attached in E3, so the table is written after it
    tbl["panel"] = panel_of(tbl, args.dataset, args.panel_n)
    tbl.sort_values("auroc", ascending=False).to_csv(
        os.path.join(out, "e1_e2_compounds.csv"), index=False)

    if not args.no_figures:
        with stage("figures"):
            try:
                from scorer import plots
                plots.make_all(out)
            except Exception as e:                            # noqa: BLE001
                print(f"[figures] skipped: {e}")
    store.close()

    report_timing(out, t_run, tbl)
    print(f"\n[run] done -> {out}")


if __name__ == "__main__":
    main()
