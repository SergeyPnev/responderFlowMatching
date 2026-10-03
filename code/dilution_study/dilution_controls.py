"""The two control weightings for the dilution grid: `oracle` and `shuffled`.

* oracle   -- s = 1 on the real crops, the clip floor on the injected ones:
  the ground-truth mask, an upper bound for the estimated weights.
* shuffled -- the same s values permuted within each (unit, plate) cell, so
  the weight multiset (normalisation, ESS, w_max) is unchanged and only the
  assignment to crops is random.

Both rewrite only the score column; ``flow_index.parquet`` and
``crop_universe.parquet`` are reused untouched.

    python dilution_study/dilution_controls.py --sets $SETS --mode oracle
    python dilution_study/dilution_controls.py --sets $SETS --mode shuffled --qs 0.1,0.25
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

S_LO, S_HI = 1e-3, 1 - 1e-3      # cache_scores.py's bounds, kept identical


def build(sets: str, q: float, mode: str, seed: int) -> str:
    qd = os.path.join(sets, f"q{q:g}")
    M = pd.read_parquet(os.path.join(qd, "members.parquet"))
    s = M["s"].to_numpy(float).copy()

    if mode == "oracle":
        s = np.where(M["role"].to_numpy() == "real", S_HI, S_LO)
    elif mode == "shuffled":
        # within cell: same multiset, only the crop each weight lands on moves
        rng = np.random.default_rng((seed, int(round(q * 1000))))
        idx = np.arange(len(M))
        for _, g in M.groupby(["unit_id", "plate"], sort=True):
            j = idx[g.index.to_numpy()]
            s[j] = s[rng.permutation(j)]
    else:                                                       # pragma: no cover
        raise SystemExit(f"unknown mode {mode!r}")

    s = np.clip(s, S_LO, S_HI)
    out = os.path.join(qd, f"crop_scores_{mode}.parquet")
    pd.DataFrame({"crop_id": M["crop_id"], "unit_id": M["unit_id"], "s": s,
                  "plate": M["plate"], "y": 1}).to_parquet(out, index=False)

    real, inj = M["role"].to_numpy() == "real", M["role"].to_numpy() == "injected"
    # the same normalised-weight share dilution_build.w_share reports
    w = pd.Series(s, index=M.index)
    w = w / w.groupby([M["unit_id"], M["plate"]]).transform("mean")
    print(f"  q={q:<5g} s_real {s[real].mean():.3f}  s_inj {s[inj].mean():.3f}  "
          f"w_inj_share {w[inj].sum() / w.sum():.4f}  "
          f"(crop share {inj.mean():.4f})  -> {os.path.basename(out)}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sets", required=True, help="the built sets dir")
    p.add_argument("--mode", required=True, choices=("oracle", "shuffled"))
    p.add_argument("--qs", default=None,
                   help="comma-separated; default every q{n}/ under --sets")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    qs = ([float(x) for x in a.qs.split(",")] if a.qs else
          sorted(float(d[1:]) for d in os.listdir(a.sets)
                 if d.startswith("q") and os.path.isdir(os.path.join(a.sets, d))))
    print(f"[{a.mode}] {len(qs)} set(s) under {a.sets}")
    for q in qs:
        build(a.sets, q, a.mode, a.seed)
    print(f"\nTrain them with the same runner, pointed at the new column:\n"
          f"  SCORES=crop_scores_{a.mode}.parquet SUFFIX=_{a.mode} \\\n"
          f"      bash flow/run_dilution_arm.sh <gpu> <q> 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
