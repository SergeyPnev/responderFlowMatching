"""Control weightings for the real-data arms: one input score column, one
variant column out.

Label-free modes (``dilution_controls.py`` covers the dilution sets, which
carry a ground truth):

* **shuffled** -- s permuted within each (unit, plate) cell. The weight
  multiset per cell, the ESS and w_max are untouched; only which crop carries
  which weight changes.
* **well_mean** -- every crop in a well gets that well's mean s (cell selection
  vs well selection).
* **nuisance** -- s refit from crop covariates alone (cells_per_field,
  edge_dist, foreground_frac), in the scorer's own per-unit contrast sets and
  folds. On CPG edge_dist is constant within a unit's treated crops (one
  treated well per plate): use `--features cells_per_field,foreground_frac`.

The output keeps every other column, so the arms differ only in w.

    python scorer/weight_variants.py --scores $P4/cache/crop_scores_cap0.95.parquet \\
        --mode shuffled --out $P4/cache/crop_scores_shuffled.parquet
    python scorer/weight_variants.py --scores ... --mode nuisance \\
        --covariates $F/rxrx1/crop_covariates_v4.parquet \\
        --run_dir $R/rxrx1/morphem_pca50_qcoptics --out ...
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

S_LO, S_HI = 1e-3, 1 - 1e-3          # cache_scores.py's bounds, kept identical
# cells_per_field and edge_dist are derived as the scorer derives them;
# foreground_frac is a column of the covariates table.
FROM_COV = ("foreground_frac",)


def nuisance_features(d: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """cells_per_field, edge_dist and foreground_frac for one contrast set."""
    f = pd.DataFrame(index=d.index)
    f["cells_per_field"] = d.groupby("field_id")["crop_id"].transform("size")
    rc = d["well_id"].astype(str).str.extract(r"([A-Z]+)(\d+)\s*$")
    if rc.notna().all().all():
        r = rc[0].map(lambda s: sum((ord(ch) - 64) * 26 ** i
                                    for i, ch in enumerate(reversed(s))))
        c = rc[1].astype(int)
        f["edge_dist"] = np.minimum.reduce([r - r.min(), r.max() - r,
                                            c - c.min(), c.max() - c]).astype(float)
    for col in FROM_COV:
        if col in cov.columns:
            f[col] = cov[col].reindex(d["crop_id"].astype(str)).to_numpy()
    return f


def shuffled(t: pd.DataFrame, seed: int) -> np.ndarray:
    """Permute s within every (unit_id, plate) cell."""
    rng = np.random.default_rng(seed)
    s = t["s"].to_numpy(float).copy()
    for _, idx in t.groupby(["unit_id", "plate"], sort=False).indices.items():
        s[idx] = rng.permutation(s[idx])
    return s


def well_mean(t: pd.DataFrame) -> np.ndarray:
    """Every crop in a well carries that well's mean s."""
    return t.groupby("well_id")["s"].transform("mean").to_numpy(float)


def _fit_folds(X, y, fold):
    """Cross-fitted probabilities in the scorer's own folds, or None."""
    from sklearn.linear_model import LogisticRegression
    p = np.full(len(y), np.nan)
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        if len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        lr = LogisticRegression(class_weight="balanced", max_iter=2000)
        lr.fit((X[tr] - mu) / sd, y[tr])
        p[te] = lr.predict_proba((X[te] - mu) / sd)[:, 1]
    return None if np.isnan(p).any() else p


def nuisance(t: pd.DataFrame, run_dir: str, cov: pd.DataFrame,
             features: str = None) -> np.ndarray:
    """Refit every unit on covariates alone, in the scorer's own folds.

    Each unit's ``scores/<unit>.parquet`` holds its post-QC contrast set and
    the well-grouped ``fold`` the real scorer used; `pi` is `2*auroc - 1` of
    this fit. A unit the fit cannot handle keeps its real s.
    """
    import glob
    from sklearn.metrics import roc_auc_score
    sys_path_hidden()
    from scorer.core import responder_weight

    key = "crop_id" if "crop_id" in cov.columns else "SAMPLE_KEY"
    if key not in cov.columns:
        raise SystemExit("the covariates table has neither crop_id nor "
                         "SAMPLE_KEY")
    cov = cov.drop_duplicates(key).set_index(cov[key].astype(str))

    out = t["s"].to_numpy(float).copy()
    row_of = {c: i for i, c in enumerate(t["crop_id"].astype(str))}
    files = sorted(glob.glob(os.path.join(run_dir, "scores", "*.parquet")))
    if not files:
        raise SystemExit(f"no scores/*.parquet under {run_dir}")
    done = skipped = 0
    aurocs, solo = [], {}
    used = None
    for f in files:
        d = pd.read_parquet(f)
        X = nuisance_features(d, cov)
        if features:
            X = X[[c for c in features.split(",") if c in X.columns]]
        if used is None:
            used = list(X.columns)
            print(f"[nuisance] features: {', '.join(used)}")
        if X.isna().any().any() or d["y"].nunique() < 2 or not len(X.columns):
            skipped += 1
            continue
        y, fold = d["y"].to_numpy(int), d["fold"].to_numpy()
        p = _fit_folds(X.to_numpy(float), y, fold)
        if p is None or len(np.unique(y)) < 2:
            skipped += 1
            continue
        auroc = roc_auc_score(y, p)
        aurocs.append(auroc)
        # which covariate carries it: the same fit on one column at a time
        for c in X.columns:
            pc = _fit_folds(X[[c]].to_numpy(float), y, fold)
            if pc is not None:
                solo.setdefault(c, []).append(roc_auc_score(y, pc))
        s_new = responder_weight(p, max(0.0, 2 * auroc - 1))
        for cid, si, yi in zip(d["crop_id"].astype(str), s_new, y):
            if yi == 1 and cid in row_of:
                out[row_of[cid]] = si
        done += 1
    if not aurocs:
        raise SystemExit("[nuisance] no unit could be refit")
    print(f"[nuisance] {done} unit(s) refit, {skipped} kept their real s")
    print(f"[nuisance] mean covariate-only AUROC {np.mean(aurocs):.3f} "
          f"(median {np.median(aurocs):.3f})")
    for c, v in solo.items():
        print(f"           {c:<18} alone {np.mean(v):.3f}")
    return out


def sys_path_hidden() -> None:
    """Put this directory on sys.path for the scorer modules."""
    import sys
    h = os.path.dirname(os.path.abspath(__file__))
    if h not in sys.path:
        sys.path.insert(0, os.path.normpath(h))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("-s", "--scores", required=True,
                   help="a crop_scores parquet")
    p.add_argument("-m", "--mode", required=True,
                   choices=("shuffled", "well_mean", "nuisance"))
    p.add_argument("-o", "--out", default=None,
                   help="default crop_scores_<mode>.parquet next to --scores")
    p.add_argument("-c", "--covariates", default=None, help="nuisance mode")
    p.add_argument("--features", default=None,
                   help="nuisance mode: comma-separated subset of "
                        "cells_per_field, edge_dist, foreground_frac")
    p.add_argument("-r", "--run_dir", default=None,
                   help="nuisance mode: the HiDDEN run dir, for its scores/ "
                        "contrast sets and fold column")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if not a.out:
        a.out = os.path.join(os.path.dirname(os.path.abspath(a.scores)),
                             f"crop_scores_{a.mode}.parquet")
    t = pd.read_parquet(a.scores)
    for c in ("crop_id", "unit_id", "s", "y"):
        if c not in t.columns:
            raise SystemExit(f"{a.scores} has no {c} column")
    s0 = t["s"].to_numpy(float)

    if a.mode == "shuffled":
        s = shuffled(t, a.seed)
    elif a.mode == "well_mean":
        s = well_mean(t)
    else:
        if not (a.covariates and a.run_dir):
            raise SystemExit("--mode nuisance needs --covariates and --run_dir")
        s = nuisance(t, a.run_dir, pd.read_parquet(a.covariates), a.features)

    s = np.clip(s, S_LO, S_HI)
    out = t.copy()
    out["s"] = s
    out["s_source"] = a.mode
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_parquet(a.out, index=False)

    # what changed, and what must not have: the per-cell multiset for shuffled
    same = float(np.mean(np.isclose(s, s0)))
    print(f"[{a.mode}] {len(t)} crop(s), {t['unit_id'].nunique()} unit(s)\n"
          f"  mean s {s0.mean():.4f} -> {s.mean():.4f}   "
          f"corr(s, s_orig) {np.corrcoef(s, s0)[0, 1]:.4f}   "
          f"unchanged {100 * same:.1f}%")
    if a.mode == "shuffled":
        cell = t.groupby(["unit_id", "plate"], sort=False).indices
        bad = [k for k, ix in cell.items()
               if not np.allclose(np.sort(s[ix]), np.sort(s0[ix]))]
        if bad:
            raise SystemExit(f"{len(bad)} cell(s) changed their s multiset; "
                             f"the shuffled arm would not be ESS-matched")
        print("  per-cell s multiset preserved -> same ESS and w_max as the "
              "real-weight arm at every gamma")
    json.dump({"mode": a.mode, "scores_in": a.scores, "seed": a.seed,
               "n": int(len(t)), "mean_s_in": float(s0.mean()),
               "mean_s_out": float(s.mean())},
              open(a.out + ".json", "w"), indent=2)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
