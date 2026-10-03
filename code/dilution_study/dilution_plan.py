"""Pre-launch checks for the dilution study. Plans, asserts, reports. Trains nothing.

A dilution set replaces a fraction ``1-q`` of a unit's treated crops with
same-plate control crops relabelled as that unit. Reports per-plate donor
arithmetic, donor overlap between compounds sharing a plate, n_real per unit
at every q, and the fixed n.

Sampling rules:
  * donors come from the same plate as the crops they replace;
  * a crop is never both source and target: whole control wells go to either
    the donor pool or the source pool;
  * donors are sampled per compound without replacement, not disjoint between
    compounds;
  * real treated crops are sampled uniformly, never by score.

    python dilution_study/dilution_plan.py
    python dilution_study/dilution_plan.py --donor_frac 0.8 --thr 0.95
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilution_study.dilution_core import (DEFAULT_P4, QS, controls,   # noqa: E402
                            draw_cell, fixed_n, load, select_units,
                            split_control_wells)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--p4", default=DEFAULT_P4)
    p.add_argument("--index", default=None)
    p.add_argument("--out", default=None, help="where to write the plan "
                                               "(default <p4>/dilution)")
    p.add_argument("--thr", type=float, default=0.95,
                   help="pi_hat floor for a unit to enter the study")
    p.add_argument("--donor_frac", type=float, default=0.8,
                   help="share of each plate's control WELLS reserved as "
                        "injection donors; the rest stay as flow sources")
    p.add_argument("--min_source_wells", type=int, default=2,
                   help="a plate with fewer source wells than this is flagged: "
                        "every treated crop on it would draw its control "
                        "partner from a single well")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out_dir = a.out or os.path.join(a.p4, "dilution")
    os.makedirs(out_dir, exist_ok=True)
    us, sc, un, fi = load(a.p4, a.index)
    tr = select_units(us, sc, un, fi, a.thr)
    ctl = controls(un, fi)

    print(f"\n== population ==")
    print(f"pi_hat >= {a.thr}: {tr['unit_id'].nunique()} train units, {len(tr)} "
          f"treated crops on {tr['plate'].nunique()} plates")
    def _sel(t):
        k = us[us["pi_hat"] >= t]
        if "has_plateau" in us.columns:
            k = k[k["has_plateau"].fillna(False)]
        return set(k["unit_id"])
    band = [(t, _sel(t)) for t in (0.8, 0.9, 0.95, 1.0)]
    in_study = set(tr["unit_id"])
    print("  qualifying units, all splits: "
          + ", ".join(f"{t}:{len(u)}" for t, u in band))
    print("  of those, in the train split: "
          + ", ".join(f"{t}:{len(u & in_study)}" for t, u in band))
    same = {t for t, u in band if len(u & in_study) == len(in_study)}
    if len(same) > 1:
        print(f"  thresholds {sorted(same)} select the SAME train units -- an "
              f"empty pi_hat band, not a coincidence. Recorded, not swept.")

    donor_pool, source_pool, P = split_control_wells(ctl, a.donor_frac, a.seed)
    cell = fixed_n(tr, P)
    N = int(cell["n"].sum())
    print(f"\n== fixed n ==\nN = {N} crops, identical at every q "
          f"({len(cell)} (unit, plate) cells, {cell['unit_id'].nunique()} units)")
    print(f"  per cell: min {cell['n'].min()}, median {int(cell['n'].median())}, "
          f"max {cell['n'].max()}")
    print("  asserted: q*n <= real available AND (1-q)*n <= plate donor pool, "
          "for every q in " + str(QS))

    # ---- per-plate donor arithmetic at the tightest q ---------------------- #
    qmin = min(QS)
    dem = (cell.assign(need=cell["n"] - np.floor(qmin * cell["n"]).astype(int))
           .groupby("plate")["need"].sum())
    P["n_compounds"] = cell.groupby("plate")["unit_id"].nunique().reindex(P.index).fillna(0).astype(int)
    P[f"demand_q{qmin}"] = dem.reindex(P.index).fillna(0).astype(int)
    P["reuse"] = (P[f"demand_q{qmin}"] / P["donor_slots"].replace(0, np.nan)).round(2)
    P["short_sources"] = P["source_wells"] < a.min_source_wells
    used = P[P["n_compounds"] > 0].copy()

    print(f"\n== per-plate donor arithmetic (donor_frac {a.donor_frac}, "
          f"whole wells) ==")
    print(used.sort_values("reuse", ascending=False).to_string())
    print(f"\n  plates in study: {len(used)};  demand/supply > 1 on "
          f"{int((used['reuse'] > 1).sum())} plate(s) -- that is donor REUSE "
          f"across compounds, which the brief permits; within a compound the "
          f"draw is without replacement and the assertion above covers it.")
    n_short = int(used["short_sources"].sum())
    if n_short:
        print(f"  ! {n_short} plate(s) have < {a.min_source_wells} source "
              f"well(s). Every treated crop on those plates would draw its "
              f"control partner from one well: {int(used[used.short_sources]['source_crops'].median())} "
              f"crops median. Lower --donor_frac.")

    # ---- donor overlap between compounds sharing a plate, at q=qmin -------- #
    draw = {}
    for _, r in cell.iterrows():
        ids = tr.loc[(tr.unit_id == r.unit_id) & (tr.plate == r.plate),
                     "crop_id"].astype(str).to_numpy()
        _, inj = draw_cell(r.unit_id, r.plate, int(r.n), qmin, ids,
                           donor_pool[r.plate], a.seed)
        draw[(r.unit_id, r.plate)] = set(inj)
    per_unit = pd.Series({u: len(set().union(*[v for (uu, _), v in draw.items() if uu == u]))
                          for u in cell["unit_id"].unique()})
    ov = []
    for plate, g in cell.groupby("plate"):
        for u1, u2 in itertools.combinations(sorted(g["unit_id"].unique()), 2):
            s1, s2 = draw[(u1, plate)], draw[(u2, plate)]
            if s1 and s2:
                ov.append(len(s1 & s2) / min(len(s1), len(s2)))
    print(f"\n== donor overlap at q={qmin} ==")
    print(f"  distinct donors per unit: median {int(per_unit.median())}, "
          f"min {int(per_unit.min())}, max {int(per_unit.max())}")
    print(f"  pairwise overlap, compounds sharing a plate: mean "
          f"{np.mean(ov):.3f}, median {np.median(ov):.3f} over {len(ov)} pairs"
          if ov else "  no plate carries two study compounds")
    print("  shared donors correlate results across compounds -- the CIs are "
          "narrower than independent sampling would give. For the error bars.")

    # ---- n_real per unit at every q ---------------------------------------- #
    print(f"\n== n_real_treated per unit ==")
    tab = []
    for q in QS:
        per_u = (cell.assign(r=np.floor(q * cell["n"]).astype(int))
                 .groupby("unit_id")["r"].sum())
        tab.append({"q": q, "n_total": N, "n_real": int(per_u.sum()),
                    "n_injected": N - int(per_u.sum()),
                    "real_per_unit_min": int(per_u.min()),
                    "real_per_unit_median": int(per_u.median()),
                    "real_per_unit_max": int(per_u.max())})
    T = pd.DataFrame(tab)
    print(T.to_string(index=False))
    print(f"  at q={qmin} that is ~{T.iloc[0]['real_per_unit_median']} real crops "
          f"per unit; fid_resp will be noisy there and the number belongs next "
          f"to the curve.")

    # ---- no crop is both source and target --------------------------------- #
    bad = sum(len(set(donor_pool[p]) & set(source_pool[p])) for p in donor_pool)
    inj_all = set().union(*draw.values()) if draw else set()
    src_all = set().union(*[set(v) for v in source_pool.values()])
    print(f"\n== assertions ==")
    print(f"  donor pool ∩ source pool, all plates: {bad}  (must be 0)")
    print(f"  injected crops ∩ flow source pool:    {len(inj_all & src_all)}  (must be 0)")
    assert bad == 0 and not (inj_all & src_all), \
        "a crop can be both source and target -- rule 2 violated"
    print("  so no training pair can have source == target, by construction.")

    used.to_csv(os.path.join(out_dir, "plan_per_plate.csv"))
    T.to_csv(os.path.join(out_dir, "plan_per_q.csv"), index=False)
    cell.to_csv(os.path.join(out_dir, "plan_cells.csv"), index=False)
    json.dump({"thr": a.thr, "donor_frac": a.donor_frac, "seed": a.seed,
               "qs": list(QS), "N_fixed": N,
               "n_units": int(cell["unit_id"].nunique()),
               "n_plates": int(len(used)),
               "donors_per_unit_median": int(per_unit.median()),
               "overlap_mean": float(np.mean(ov)) if ov else None,
               "plates_short_sources": n_short},
              open(os.path.join(out_dir, "plan.json"), "w"), indent=2)
    print(f"\n-> {out_dir}/plan.json, plan_per_plate.csv, plan_per_q.csv, "
          f"plan_cells.csv")
    print("\nSTOP. Nothing was built and nothing trains. Read the per-plate "
          "table and the fixed n above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
