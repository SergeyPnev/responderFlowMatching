"""Per-crop covariates the embedding h5 does not carry. One pass over the .npy.

    python features/covariates.py --dataset bbbc021 \
        --img_dir $D/bbbc021_all --fold_path $D/bbbc021_all/metadata/split_iclr \
        --out $F/bbbc021/crop_covariates.parquet --workers 16

Writes one row per SAMPLE_KEY:

    focus_ch{i}      Laplacian variance / channel variance (contrast-invariant;
                     the raw value is lapvar_ch{i}). LOW is bad.
    satfrac_ch{i}    fraction of pixels at the top of the uint8 range. HIGH is bad.
    illum_r2_ch{i}   R^2 of a least-squares plane fit to the channel. HIGH is bad.
    mean_ch{i}       per-channel mean (thresholded later, relative to the plate's
                     controls).
    foreground_frac  fraction of pixels above --thresh in any channel. A
                     diagnostic, not a filter: it is a phenotype.
    dna_integrated   sum of the DNA channel inside its own threshold mask
                     (cell-cycle proxy); dna_area is the mask fraction.
"""
from __future__ import annotations

import argparse
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.extract_features import DATASETS, load_folds          # noqa: E402


_PLANE: dict = {}


def _plane_design(h: int, w: int) -> np.ndarray:
    """[1, x, y] design matrix for a h x w grid, cached per shape."""
    if (h, w) not in _PLANE:
        yy, xx = np.mgrid[0:h, 0:w]
        A = np.stack([np.ones(h * w), xx.ravel() / max(w - 1, 1),
                      yy.ravel() / max(h - 1, 1)], axis=1).astype(np.float32)
        _PLANE[(h, w)] = (A, np.linalg.pinv(A))
    return _PLANE[(h, w)]


def _defects(a: np.ndarray, sat: float) -> dict:
    """Per-channel focus / saturation / illumination-ramp metrics."""
    h, w, C = a.shape
    A, Ainv = _plane_design(h, w)
    out = {}
    for i in range(C):
        ch = a[..., i]
        lap = (4 * ch[1:-1, 1:-1] - ch[:-2, 1:-1] - ch[2:, 1:-1]
               - ch[1:-1, :-2] - ch[1:-1, 2:])
        v = float(ch.var())
        lv = float(lap.var())
        z = ch.ravel().astype(np.float32)
        resid = z - A @ (Ainv @ z)
        sst = float(((z - z.mean()) ** 2).sum())
        out[f"lapvar_ch{i}"] = lv
        out[f"focus_ch{i}"] = lv / (v + 1e-8)
        out[f"satfrac_ch{i}"] = float((ch >= sat).mean())
        out[f"illum_r2_ch{i}"] = (1.0 - float((resid ** 2).sum()) / sst
                                  if sst > 0 else 0.0)
    return out


def one(key: str, img_dir: str, dataset: str, thresh: float, dna: int,
        sat: float = 254 / 255):
    spec = DATASETS[dataset]
    try:
        a = np.load(spec.path_of(img_dir, key))
    except Exception:                                       # noqa: BLE001
        return {"SAMPLE_KEY": key}
    C = spec.n_channels
    if a.ndim == 3 and a.shape[-1] != C and a.shape[0] == C:
        a = a.transpose(1, 2, 0)
    a = a.astype(np.float32) / 255.0
    m = (a.max(axis=-1) > thresh)
    d = a[..., dna]
    dm = d > thresh
    out = {"SAMPLE_KEY": key,
           "foreground_frac": float(m.mean()),
           "dna_integrated": float(d[dm].sum()),
           "dna_area": float(dm.mean())}
    out.update({f"mean_ch{i}": float(a[..., i].mean()) for i in range(C)})
    out.update(_defects(a, sat))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    p.add_argument("--img_dir", required=True)
    p.add_argument("--fold_path", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--thresh", type=float, default=20 / 255,
                   help="foreground cut on the /255 image")
    p.add_argument("--dna_channel", type=int, default=0)
    p.add_argument("--sat", type=float, default=254 / 255,
                   help="saturation cut on the /255 image")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    spec = DATASETS[args.dataset]
    keys = load_folds(spec, args.fold_path, args.fold, None)["SAMPLE_KEY"]
    keys = keys.drop_duplicates().tolist()
    if args.limit:
        keys = keys[: args.limit]
    print(f"[cov] {len(keys)} unique crops")

    fn = partial(one, img_dir=args.img_dir, dataset=args.dataset,
                 thresh=args.thresh, dna=args.dna_channel, sat=args.sat)
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            rows = list(tqdm(pool.imap(fn, keys, chunksize=256), total=len(keys)))
    else:
        rows = [fn(k) for k in tqdm(keys)]

    df = pd.DataFrame(rows)
    bad = int(df["foreground_frac"].isna().sum()) if "foreground_frac" in df else len(df)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[cov] {bad} unreadable -> {args.out}")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
