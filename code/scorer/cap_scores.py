"""Cap pi in the responder weight, for the gamma sweep's score cache.

    python scorer/cap_scores.py --cache $P4/cache --cap 0.95
    #   -> $P4/cache/crop_scores_cap0.95.parquet   (the cache itself is untouched)

``s = clip(1 - (1-pi)/rho, 0, 1)`` loses its rho dependence as pi -> 1, so pi
is capped for the weight only; ``pi_hat`` itself is not changed. Before
writing, s is recomputed from the cached posterior ``h`` with the uncapped pi
and must reproduce the cached ``s_raw``. The output keeps the cache's
provenance stamp, adds ``pi_cap``, and keeps the uncapped weight as
``s_uncapped``.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

S_LO, S_HI = 1e-3, 1 - 1e-3      # cache_scores.py's bounds, kept identical
EPS = 1e-6                       # core.py's clip on h


def responder_weight(h: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """core.responder_weight with prior_ratio 1 (the scorer balances)."""
    hh = np.clip(h, EPS, 1 - EPS)
    return np.clip(1.0 - (1.0 - pi) * (1.0 - hh) / hh, 0.0, 1.0)


def cap_scores(crop: pd.DataFrame, units: pd.DataFrame, cap: float,
               tol: float = 1e-6) -> pd.DataFrame:
    """``crop`` with s / s_raw recomputed at ``min(pi_hat, cap)``."""
    pi = crop["unit_id"].astype(str).map(
        dict(zip(units["unit_id"].astype(str), units["pi_hat"].astype(float))))
    if pi.isna().any():
        raise SystemExit(f"{int(pi.isna().sum())} crop(s) belong to a unit "
                         f"with no pi_hat in unit_scores.parquet")
    pi = pi.to_numpy(float)
    h = crop["h"].to_numpy(float)
    err = np.abs(responder_weight(h, pi) - crop["s_raw"].to_numpy(float))
    if err.max() > tol:
        worst = (crop.assign(err=err, pi_hat=pi).groupby("unit_id")
                 .agg(max_diff=("err", "max"), pi_hat=("pi_hat", "first"))
                 .sort_values("max_diff", ascending=False).head(5))
        raise SystemExit(
            f"recomputing s from h with the uncapped pi_hat does not reproduce "
            f"the cached s_raw (max |diff| {err.max():.2e} > {tol:g}), so the "
            f"scorer used a different pi or rho than this script assumes. "
            f"Nothing written. Worst units:\n{worst.to_string()}")
    s_raw = responder_weight(h, np.minimum(pi, cap))
    return crop.assign(s_uncapped=crop["s"], s_raw=s_raw,
                       s=np.clip(s_raw, S_LO, S_HI))


def write_capped(df: pd.DataFrame, src: str, path: str, cap: float) -> None:
    """Parquet carrying ``src``'s provenance stamp plus ``pi_cap``."""
    md = pq.read_schema(src).metadata or {}
    kv = {k: v for k, v in md.items() if not k.startswith(b"pandas")}
    kv[b"pi_cap"] = str(cap).encode()
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(tbl.replace_schema_metadata(
        {**(tbl.schema.metadata or {}), **kv}), path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="cache_scores.py's out dir")
    p.add_argument("--cap", type=float, default=0.95)
    p.add_argument("--out", default=None,
                   help="default <cache>/crop_scores_cap<cap>.parquet")
    a = p.parse_args()

    src = os.path.join(a.cache, "crop_scores.parquet")
    crop = pd.read_parquet(src)
    units = pd.read_parquet(os.path.join(a.cache, "unit_scores.parquet"))
    out = cap_scores(crop, units, a.cap)
    print(f"[cap] self-check pass: s recomputed from h reproduces the cached "
          f"s_raw on all {len(crop)} crops")
    n_cap = int((units["pi_hat"] > a.cap).sum())
    print(f"[cap] pi_cap {a.cap}: {n_cap} of {len(units)} units have pi_hat "
          f"above it and get new weights; the rest are unchanged")
    print(f"[cap] s at the upper clip: {(crop['s'] >= S_HI).mean():.2%} -> "
          f"{(out['s'] >= S_HI).mean():.2%};  at the lower clip: "
          f"{(crop['s'] <= S_LO).mean():.2%} -> {(out['s'] <= S_LO).mean():.2%};"
          f"  mean s {crop['s'].mean():.3f} -> {out['s'].mean():.3f}")

    path = a.out or os.path.join(a.cache, f"crop_scores_cap{a.cap:g}.parquet")
    write_capped(out, src, path, a.cap)
    print(f"-> {path}\n   train on it with --crop_scores {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
