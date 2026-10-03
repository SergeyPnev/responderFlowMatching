"""Shared planning for the dilution study: loading, well splitting, the fixed n.

Used by both ``dilution_plan.py`` (report) and ``dilution_build.py``
(materialise). Pure pandas/numpy -- no image loading, no GPU.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

DEFAULT_P4 = "/path/to/workdir/results/iclr/phase4"
QS = (0.1, 0.25, 0.5, 0.75, 0.9)


def load(p4: str, index: str | None = None, quiet: bool = False):
    """unit_scores, crop_scores, crop_universe, flow_index. Plate from ONE file.

    ``crop_scores`` also carries a plate column whose labels may differ, so
    it is dropped here and the flow index supplies plate and well.
    """
    import pyarrow.parquet as pq

    idx = index or os.path.join(p4, "flow_index_bbbc021.parquet")
    us = pd.read_parquet(os.path.join(p4, "cache", "unit_scores.parquet"))
    sc = pd.read_parquet(os.path.join(p4, "cache", "crop_scores.parquet"))
    un = pd.read_parquet(os.path.join(p4, "cache", "crop_universe.parquet"))
    have = set(pq.read_schema(idx).names)
    keep = [c for c in ("crop_id", "plate", "well", "well_id", "split") if c in have]
    fi = pd.read_parquet(idx, columns=keep).copy()
    fi["plate"] = fi["plate"].astype(str)
    if "well" not in fi.columns and "well_id" in fi.columns:
        fi = fi.rename(columns={"well_id": "well"})
    fi["well"] = fi["well"].astype(str)

    lab_sc = set(sc["plate"].astype(str)) if "plate" in sc.columns else set()
    if not quiet:
        print(f"[labels] plate: crop_scores {len(lab_sc)}, flow_index "
              f"{fi['plate'].nunique()}, shared {len(lab_sc & set(fi['plate']))}")
        if lab_sc and lab_sc != set(fi["plate"]):
            only_sc = sorted(lab_sc - set(fi["plate"]))
            only_fi = sorted(set(fi["plate"]) - lab_sc)
            print(f"  ! differ -- {len(only_sc)} only in crop_scores {only_sc[:3]}, "
                  f"{len(only_fi)} only in flow_index {only_fi[:3]}")
            print("    (plate/split come from flow_index throughout)")
    sc = sc.drop(columns=[c for c in ("plate", "well", "well_id") if c in sc.columns])
    return us, sc, un, fi


def select_units(us, sc, un, fi, thr: float) -> pd.DataFrame:
    """Treated crops of the train units clearing ``pi_hat >= thr``, QC-kept."""
    keep = us[us["pi_hat"] >= thr]
    if "has_plateau" in us.columns:
        keep = keep[keep["has_plateau"].fillna(False)]
    tr = sc.merge(fi, on="crop_id")
    tr = tr[(tr["split"] == "train") & (tr["unit_id"].isin(keep["unit_id"]))]
    kept = set(un.loc[un["kept"], "crop_id"].astype(str))
    tr = tr[tr["crop_id"].astype(str).isin(kept)]
    if not len(tr):
        raise SystemExit(f"no train unit clears pi_hat >= {thr}")
    return tr.reset_index(drop=True)


def controls(un, fi) -> pd.DataFrame:
    c = un.merge(fi, on="crop_id")
    return c[(c["kept"]) & (c["y"] == 0)].reset_index(drop=True)


def split_control_wells(ctl: pd.DataFrame, donor_frac: float, seed: int
                        ) -> Tuple[Dict, Dict, pd.DataFrame]:
    """Whole control wells -> donor pool / flow source pool, per plate.

    Whole wells because scoring groups folds by well: a well spanning both
    sides would leak.
    """
    rng = np.random.default_rng(seed)
    rows, donor, source = [], {}, {}
    for plate, g in ctl.groupby("plate"):
        wells = np.sort(g["well"].unique())
        wells = wells[rng.permutation(len(wells))]
        n_don = int(np.floor(len(wells) * donor_frac))
        n_don = max(1, min(n_don, len(wells) - 1))     # never starve either side
        dw, sw = set(wells[:n_don]), set(wells[n_don:])
        donor[plate] = g[g["well"].isin(dw)]["crop_id"].astype(str).to_numpy()
        source[plate] = g[g["well"].isin(sw)]["crop_id"].astype(str).to_numpy()
        rows.append({"plate": plate, "n_control_wells": len(wells),
                     "n_controls": len(g), "donor_wells": n_don,
                     "source_wells": len(wells) - n_don,
                     "donor_slots": len(donor[plate]),
                     "source_crops": len(source[plate])})
    return donor, source, pd.DataFrame(rows).set_index("plate")


def fixed_n(tr: pd.DataFrame, P: pd.DataFrame, qs=QS) -> pd.DataFrame:
    """One n per (unit, plate), identical at every q.

    n must satisfy ``q*n <= T`` and ``(1-q)*n <= D`` for every q, so
    ``n <= min(T / max(q), D / (1 - min(q)))``. Fixed across the grid so a
    change in FID with q is q moving and not n moving.
    """
    cell = tr.groupby(["unit_id", "plate"]).size().rename("T").reset_index()
    cell["D"] = P["donor_slots"].reindex(cell["plate"]).fillna(0).to_numpy().astype(int)
    cell["n"] = np.floor(np.minimum(cell["T"] / max(qs),
                                    cell["D"] / (1 - min(qs)))).astype(int)
    cell = cell[cell["n"] > 0].reset_index(drop=True)
    for q in qs:
        n_real = np.floor(q * cell["n"]).astype(int)
        assert (n_real <= cell["T"]).all(), f"q={q}: real demand exceeds supply"
        assert (cell["n"] - n_real <= cell["D"]).all(), \
            f"q={q}: donor demand exceeds a plate pool"
    return cell


def draw_cell(unit_id: str, plate: str, n: int, q: float,
              real_ids: np.ndarray, donor_ids: np.ndarray, seed: int):
    """Uniform real crops + donors, both without replacement within the cell.

    Seeded on ``(seed, unit, plate, q)``, so the gamma arms at one q get
    identical sets. Real crops are sampled uniformly, never by score.
    """
    n_real = int(np.floor(q * n))
    n_inj = int(n - n_real)
    g = np.random.default_rng((seed, abs(hash(unit_id)) % (2 ** 31),
                               abs(hash(plate)) % (2 ** 31), int(round(q * 1000))))
    real = g.choice(real_ids, n_real, replace=False) if n_real else np.empty(0, object)
    inj = g.choice(donor_ids, n_inj, replace=False) if n_inj else np.empty(0, object)
    return real, inj
