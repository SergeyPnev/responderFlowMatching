"""Materialise the dilution sets, one per q, plus the donor score report.

Writes, under ``<out>/q{q}/``, the three parquets the trainer reads
(``flow_index.parquet``, ``crop_scores.parquet``, ``crop_universe.parquet``)
and ``members.parquet`` for scoring the sets as built.

* Injected rows get a synthetic ``crop_id`` ``{orig}__{unit_id}`` (one control
  crop is a target for several units); ``latent_key`` carries the original id.
* Injected wells are renamed ``inj|{orig_well}`` so fold grouping by well keeps
  donor and source crops of one physical well apart.
* Donor scores come from the scorer run's per-contrast ``scores/*.parquet``:
  a donor injected into unit u carries the out-of-fold score it got in u's own
  contrast. Nothing is rescored.

    python dilution_study/dilution_build.py --run_dir $RUN
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilution_study.dilution_core import (DEFAULT_P4, QS, controls, draw_cell, fixed_n,
                           load, select_units, split_control_wells)

S_LO, S_HI = 1e-3, 1 - 1e-3      # cache_scores.py's bounds, kept identical


def donor_scores(run_dir: str, sc: pd.DataFrame,
                 both: bool = False) -> pd.DataFrame:
    """(unit_id, crop_id) -> s, for control crops, from the per-contrast files.

    Each file holds one contrast, both classes. The unit is recovered from the
    treated rows via the score cache.
    """
    owner = dict(zip(sc["crop_id"].astype(str), sc["unit_id"].astype(str)))
    out = []
    files = sorted(glob.glob(os.path.join(run_dir, "scores", "*.parquet")))
    if not files:
        raise SystemExit(f"no per-contrast scores under {run_dir}/scores/. "
                         f"Job A needs them for the injected donors.")
    for f in files:
        d = pd.read_parquet(f)
        d["crop_id"] = d["crop_id"].astype(str)
        units = d.loc[d.y == 1, "crop_id"].map(owner).dropna()
        if not len(units):
            continue
        uid = units.mode().iloc[0]
        c = (d if both else d[d.y == 0])[["crop_id", "y", "s", "h"]].copy()
        c["unit_id"] = uid
        out.append(c)
    ds = pd.concat(out, ignore_index=True)
    ds = ds.drop_duplicates(["unit_id", "crop_id"])
    print(f"[jobA] donor scores: {len(ds)} (unit, control) pairs from "
          f"{len(files)} contrast file(s)")
    return ds


def w_share(M: pd.DataFrame) -> float:
    """Weight the injected crops carry at gamma=1, under weight_norm unit_plate.

    Weights are normalised to mean 1 within each (unit, plate) cell first, as
    the loss sees them. Compare against the injected crop share.
    """
    w = np.clip(M["s"].to_numpy(float), 1e-3, 1 - 1e-3)
    w = pd.Series(w, index=M.index)
    w = w / w.groupby([M["unit_id"], M["plate"]]).transform("mean")
    return float(w[M.role == "injected"].sum() / w.sum())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--p4", default=DEFAULT_P4)
    p.add_argument("--run_dir", required=True,
                   help="the frozen v4 scorer run, for the per-contrast scores")
    p.add_argument("--index", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--thr", type=float, default=0.95)
    p.add_argument("--donor_frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pi_cap", type=float, default=1.0,
                   help="cap pi when recomputing s from the cached posterior: "
                        "s = clip(1 - (1-min(pi,cap))/rho, 0, 1). 1.0 keeps the "
                        "cache as-is. The weight loses its rho dependence as "
                        "pi -> 1 and at pi_hat == 1.0 every crop in the "
                        "contrast scores exactly 1, controls included -- so on "
                        "a pi>=0.95 population the tilt barely separates "
                        "injected donors from real treated crops. Capping is "
                        "shrinkage away from a boundary estimate, applied "
                        "identically to both classes.")
    a = p.parse_args()

    out_dir = a.out or os.path.join(a.p4, "dilution", "sets")
    os.makedirs(out_dir, exist_ok=True)
    idx_path = a.index or os.path.join(a.p4, "flow_index_bbbc021.parquet")
    full = pd.read_parquet(idx_path)
    full["crop_id"] = full["crop_id"].astype(str)
    full["plate"] = full["plate"].astype(str)
    full["well"] = full["well"].astype(str)

    us, sc, un, fi = load(a.p4, idx_path)
    sc["crop_id"] = sc["crop_id"].astype(str)
    tr = select_units(us, sc, un, fi, a.thr)
    ctl = controls(un, fi)
    donor_pool, source_pool, P = split_control_wells(ctl, a.donor_frac, a.seed)
    cell = fixed_n(tr, P)
    # both classes from the SAME contrast, so a capped s is computed the same
    # way for a real treated crop and for an injected donor
    dsc = donor_scores(a.run_dir, sc, both=a.pi_cap < 1.0)
    if a.pi_cap < 1.0:
        pi_of = dict(zip(us["unit_id"].astype(str), us["pi_hat"].astype(float)))
        h = np.clip(dsc["h"].to_numpy(float), 1e-6, 1 - 1e-6)
        pe = np.array([min(pi_of.get(str(u), 1.0), a.pi_cap)
                       for u in dsc["unit_id"]])
        # same [S_LO, S_HI] clip as cache_scores.py: an unclipped zero would
        # delete a crop from training (0 ** gamma == 0) instead of down-weighting it
        dsc["s"] = np.clip(1 - (1 - pe) / (h / (1 - h)), S_LO, S_HI)
        print(f"[cap] s recomputed at pi_cap={a.pi_cap}: "
              f"treated mean {dsc.loc[dsc.y == 1, 's'].mean():.3f}, "
              f"control mean {dsc.loc[dsc.y == 0, 's'].mean():.3f}")
        smap = {c: v for c, v in zip(dsc.loc[dsc.y == 1, "crop_id"],
                                     dsc.loc[dsc.y == 1, "s"])}
        dmap = {(u, c): v for u, c, v, yy in
                zip(dsc.unit_id, dsc.crop_id, dsc.s, dsc.y) if yy == 0}
    else:
        smap = dict(zip(sc["crop_id"], sc["s"]))
        dmap = {(u, c): v for u, c, v in zip(dsc.unit_id, dsc.crop_id, dsc.s)}
    row_of = full.set_index("crop_id")

    # per-unit label fields, taken from a real treated row of that unit
    lab = (full[full.crop_id.isin(tr.crop_id)]
           .merge(tr[["crop_id", "unit_id"]], on="crop_id")
           .groupby("unit_id").first())

    report = []
    for q in QS:
        rows, members = [], []
        for _, r in cell.iterrows():
            ids = tr.loc[(tr.unit_id == r.unit_id) & (tr.plate == r.plate),
                         "crop_id"].astype(str).to_numpy()
            real, inj = draw_cell(r.unit_id, r.plate, int(r.n), q, ids,
                                  donor_pool[r.plate], a.seed)
            for c in real:
                members.append({"crop_id": c, "src_crop_id": c,
                                "unit_id": r.unit_id, "plate": r.plate,
                                "well": row_of.at[c, "well"], "role": "real",
                                "s": smap.get(c, np.nan)})
            for c in inj:
                members.append({"crop_id": f"{c}__{r.unit_id}", "src_crop_id": c,
                                "unit_id": r.unit_id, "plate": r.plate,
                                "well": f"inj|{row_of.at[c, 'well']}",
                                "role": "injected",
                                "s": dmap.get((r.unit_id, c), np.nan)})
        M = pd.DataFrame(members)

        # ---- flow index: treated (real + injected) + the source controls --- #
        treated = M.merge(lab[["compound", "dose", "moa", "text_idx"]]
                          if "text_idx" in lab.columns
                          else lab[["compound", "dose", "moa"]],
                          left_on="unit_id", right_index=True, how="left")
        t_idx = pd.DataFrame({
            "crop_id": treated["crop_id"],
            "latent_key": treated["src_crop_id"],
            "path": row_of.loc[treated["src_crop_id"], "path"].to_numpy(),
            "y": 1, "split": "train",
            "compound": treated["compound"], "dose": treated["dose"],
            "moa": treated["moa"], "plate": treated["plate"],
            "well": treated["well"]})
        if "text_idx" in treated.columns:
            t_idx["text_idx"] = treated["text_idx"].to_numpy()
        src = np.concatenate([source_pool[p_] for p_ in sorted(source_pool)])
        src = [c for c in src if c in row_of.index]
        c_idx = row_of.loc[src].reset_index()[
            [c for c in ("crop_id", "path", "split", "compound", "dose", "moa",
                         "plate", "well", "text_idx") if c in full.columns]].copy()
        c_idx["latent_key"] = c_idx["crop_id"]
        c_idx["y"] = 0
        c_idx["split"] = "train"
        di = pd.concat([t_idx, c_idx], ignore_index=True)
        assert not di["crop_id"].duplicated().any(), "duplicate crop_id in the set"

        qd = os.path.join(out_dir, f"q{q}")
        os.makedirs(qd, exist_ok=True)
        di.to_parquet(os.path.join(qd, "flow_index.parquet"), index=False)
        M.to_parquet(os.path.join(qd, "members.parquet"), index=False)
        M[["crop_id", "unit_id", "s"]].assign(
            plate=M["plate"], y=1).to_parquet(
            os.path.join(qd, "crop_scores.parquet"), index=False)
        di[["crop_id", "y"]].assign(kept=True).to_parquet(
            os.path.join(qd, "crop_universe.parquet"), index=False)

        M["s"] = M["s"].clip(S_LO, S_HI)     # covers the pi_cap=1.0 path too
        rl, ij = M[M.role == "real"], M[M.role == "injected"]
        report.append({"q": q, "n_total": len(M), "n_real": len(rl),
                       "n_injected": len(ij), "n_source_controls": len(c_idx),
                       "s_real_mean": float(rl["s"].mean()),
                       "s_inj_mean": float(ij["s"].mean()) if len(ij) else np.nan,
                       "s_inj_median": float(ij["s"].median()) if len(ij) else np.nan,
                       "s_inj_missing": int(ij["s"].isna().sum()),
                       "w_inj_share_gamma1": w_share(M),
                       "crop_share_inj": float((M.role == "injected").mean()),
                       "inj_at_s_lo": float((ij["s"] <= S_LO).mean()),
                       "real_at_s_hi": float((rl["s"] >= S_HI).mean())})
        print(f"[q={q}] {len(M)} treated ({len(rl)} real + {len(ij)} injected) "
              f"+ {len(c_idx)} source controls -> {qd}")

    pd.DataFrame({"unit_id": sorted(cell["unit_id"].unique())}).to_csv(
        os.path.join(out_dir, "study_units.csv"), index=False)
    R = pd.DataFrame(report)
    R.to_csv(os.path.join(out_dir, "jobA_scores.csv"), index=False)
    print("\n== Job A: cached score of injected donors vs real treated ==")
    print(R.to_string(index=False))
    print("\n  s_inj_mean near 0 means gamma=1 down-weights the injected mass "
          "almost out of the loss -- the intended mechanism, now measured "
          "rather than assumed. w_inj_share_gamma1 is the fraction of total "
          "weight the injected crops still carry at gamma=1.")
    if R["s_inj_missing"].sum():
        print(f"  ! {int(R['s_inj_missing'].sum())} injected crop(s) had no "
              f"score in their unit's contrast. They are NOT silently "
              f"defaulted -- fix before training.")
    json.dump({"thr": a.thr, "donor_frac": a.donor_frac, "seed": a.seed,
               "qs": list(QS), "run_dir": a.run_dir, "pi_cap": a.pi_cap},
              open(os.path.join(out_dir, "build.json"), "w"), indent=2)
    print(f"\n-> {out_dir}/q*/, jobA_scores.csv, study_units.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
