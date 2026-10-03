"""Distribution metrics re-estimated from stored features. CPU, no generation.

    python evaluation/dist_metrics.py table  --dir F [--gen a,b] [--subset C] --out T.csv
    python evaluation/dist_metrics.py latent --dir L --out E.csv
    python evaluation/dist_metrics.py fidc   --real R --gen F/feat_gen__x.npy --out P.csv
    python evaluation/dist_metrics.py spread --csv e1.csv e2.csv [--table T.csv --match re]

table:  FID, FID_inf, KID (subset and unbiased) and energy distance of every
        feat_gen__*.npy against the real crops it was generated for, with
        well-level bootstrap intervals. KID values are raw (not x100).
latent: energy distance on eval_flow --dump_latents.
fidc:   per-perturbation FID and its real-vs-real n-bias.
spread: mean / sd / range over eval_flow rows and over ``table`` rows.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# FID
def _sqrt_psd(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


def _tr_sqrt_prod(S1: np.ndarray, C2: np.ndarray) -> float:
    """tr sqrt(C1 C2) with S1 = C1^(1/2): the eigenvalues of S1 C2 S1."""
    w = np.linalg.eigvalsh(S1 @ C2 @ S1)
    return float(np.sqrt(np.clip(w, 0.0, None)).sum())


def fid(X: np.ndarray, Y: np.ndarray) -> float:
    """Frechet distance between two feature sets, (n-1) covariances.

    With n + m < d, tr sqrt(Cx Cy) is the nuclear norm of
    Ax Ay^T / sqrt((n-1)(m-1)) (A = centred features): an n x m SVD."""
    n, m, d = len(X), len(Y), X.shape[1]
    if n < 2 or m < 2:
        return float("nan")
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    mx, my = X.mean(0), Y.mean(0)
    A, B = X - mx, Y - my
    trx = float((A * A).sum() / (n - 1)); try_ = float((B * B).sum() / (m - 1))
    if n + m < d:
        s = float(np.linalg.svd(A @ B.T, compute_uv=False).sum()
                  / np.sqrt((n - 1) * (m - 1)))
    else:
        s = _tr_sqrt_prod(_sqrt_psd(A.T @ A / (n - 1)), B.T @ B / (m - 1))
    return float(((mx - my) ** 2).sum() + trx + try_ - 2.0 * s)


def fid_inf(X: np.ndarray, Y: np.ndarray, n_points: int = 8, reps: int = 3,
            seed: int = 0) -> Tuple[float, float]:
    """(FID_inf, slope k) from FID_N = FID_inf + k / N, real side fixed."""
    X = np.asarray(X, np.float64)
    mx = X.mean(0); A = X - mx
    Cx = A.T @ A / (len(X) - 1)
    Sx, trx = _sqrt_psd(Cx), float(np.trace(Cx))
    m = len(Y)
    Ns = np.unique(np.linspace(max(50, m // 5), m, n_points).astype(int))
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for N in Ns:
        for _ in range(reps if N < m else 1):
            G = np.asarray(Y[rng.choice(m, N, replace=False)], np.float64)
            my = G.mean(0); Bc = G - my
            Cy = Bc.T @ Bc / (N - 1)
            ys.append(float(((mx - my) ** 2).sum() + trx + np.trace(Cy)
                            - 2.0 * _tr_sqrt_prod(Sx, Cy)))
            xs.append(1.0 / N)
    k, b = np.polyfit(xs, ys, 1)
    return float(b), float(k)


# --------------------------------------------------------------------------- #
# kernel / distance sums, by well
def _poly(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """torchmetrics' KID kernel: (a.b / d + 1)^3."""
    return (A @ B.T / A.shape[1] + 1.0) ** 3


def _dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2.0 * A @ B.T
    return np.sqrt(np.maximum(d2, 0.0))


def _onehot(g: np.ndarray, W: int) -> np.ndarray:
    H = np.zeros((len(g), W))
    H[np.arange(len(g)), g] = 1.0
    return H


def block_sums(A, B, ga, gb, W, kern, same: bool, chunk: int = 1024):
    """[W, W]: sum over a in well w, b in well v of kern(a, b); ``same`` zeroes
    the i == j entries (A is B), so no crop is ever paired with itself."""
    A = np.asarray(A, np.float32); B = np.asarray(B, np.float32)
    Hb = _onehot(gb, W)
    out = np.zeros((W, W))
    for i0 in range(0, len(A), chunk):
        i1 = min(i0 + chunk, len(A))
        K = kern(A[i0:i1], B).astype(np.float64)
        if same:
            r = np.arange(i1 - i0)
            K[r, i0 + r] = 0.0
        out += _onehot(ga[i0:i1], W).T @ (K @ Hb)
    return out


def _u_stat(Bxx, Byy, Bxy, c, nw_x, nw_y, sign: int) -> float:
    """sign=+1: MMD^2 (xx + yy - 2xy); sign=-1: energy (2xy - xx - yy).
    c = well multiplicities (all ones at the point estimate)."""
    Nx, Ny = float(c @ nw_x), float(c @ nw_y)
    pxx = Nx * Nx - float((c * c) @ nw_x)
    pyy = Ny * Ny - float((c * c) @ nw_y)
    xx = c @ Bxx @ c / pxx
    yy = c @ Byy @ c / pyy
    xy = c @ Bxy @ c / (Nx * Ny)
    return float(sign * (xx + yy) - sign * 2.0 * xy)


def kid_subsets(X, Y, subsets: int = 100, size: int = 100, seed: int = 0):
    """eval_flow's / CellFlux's KID: torchmetrics' estimator on random subsets."""
    rng = np.random.default_rng(seed)
    m = min(size, len(X), len(Y))
    v = []
    for _ in range(subsets):
        a = np.asarray(X[rng.permutation(len(X))[:m]], np.float64)
        b = np.asarray(Y[rng.permutation(len(Y))[:m]], np.float64)
        kxx, kyy, kxy = _poly(a, a), _poly(b, b), _poly(a, b)
        v.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                 + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                 - 2.0 * kxy.sum() / (m * m))
    return float(np.mean(v)), float(np.std(v))


# --------------------------------------------------------------------------- #
def well_key(idx: pd.DataFrame) -> np.ndarray:
    if "well_id" in idx.columns:
        return idx["well_id"].astype(str).to_numpy()
    if {"plate", "well"} <= set(idx.columns):
        return (idx["plate"].astype(str) + "|" + idx["well"].astype(str)).to_numpy()
    print("  ! no well column: the bootstrap resamples crops, and its interval "
          "is too narrow")
    return idx["crop_id"].astype(str).to_numpy()


def draw_counts(well_unit: np.ndarray, rng) -> np.ndarray:
    """Well multiplicities for one bootstrap draw, wells resampled within unit."""
    c = np.zeros(len(well_unit))
    for u in np.unique(well_unit):
        ws = np.flatnonzero(well_unit == u)
        np.add.at(c, rng.choice(ws, len(ws)), 1.0)
    return c


class Paired:
    """Real and generated features for the same crops, grouped by well."""

    def __init__(self, idx: pd.DataFrame, X: np.ndarray, Y: np.ndarray):
        self.idx, self.X, self.Y = idx.reset_index(drop=True), X, Y
        wk = well_key(self.idx)
        self.wells, self.g = np.unique(wk, return_inverse=True)
        self.W = len(self.wells)
        unit = (self.idx["unit_id"].astype(str).to_numpy()
                if "unit_id" in self.idx.columns else np.zeros(len(wk), str))
        self.well_unit = pd.Series(unit).groupby(self.g).first().to_numpy()
        self.nw = np.bincount(self.g, minlength=self.W).astype(float)
        self.rows = [np.flatnonzero(self.g == w) for w in range(self.W)]
        self._blocks = {}

    def blocks(self, kind: str):
        if kind not in self._blocks:
            kern = _poly if kind == "kid" else _dist
            self._blocks[kind] = tuple(
                block_sums(a, b, self.g, self.g, self.W, kern, same)
                for a, b, same in ((self.X, self.X, True), (self.Y, self.Y, True),
                                   (self.X, self.Y, False)))
        return self._blocks[kind]

    def u_stat(self, kind: str, c=None) -> float:
        c = np.ones(self.W) if c is None else c
        Bxx, Byy, Bxy = self.blocks(kind)
        return _u_stat(Bxx, Byy, Bxy, c, self.nw, self.nw,
                       +1 if kind == "kid" else -1)

    def rows_of(self, c) -> np.ndarray:
        return np.concatenate([np.repeat(self.rows[w][None], int(k), 0).ravel()
                               for w, k in enumerate(c) if k > 0])


def load_gen(path: str):
    """feat_gen__<name>.npy -> (name, index, features); the index is
    index__<name>.parquet beside it (inception_feats pngs) or the dir's."""
    d, f = os.path.split(path)
    name = f[len("feat_gen__"):-len(".npy")]
    own = os.path.join(d, f"index__{name}.parquet")
    idx = pd.read_parquet(own if os.path.exists(own)
                          else os.path.join(d, "index.parquet"))
    F = np.load(path)
    if len(idx) != len(F):
        raise SystemExit(f"{path}: {len(F)} rows but its index has {len(idx)}")
    idx["crop_id"] = idx["crop_id"].astype(str)
    return name, idx, F


def join(real_idx, XR, gen_idx, XG):
    """Rows present on both sides, in the real index's order."""
    pos = pd.Series(np.arange(len(gen_idx)), index=gen_idx["crop_id"])
    pos = pos[~pos.index.duplicated()]
    keep = real_idx["crop_id"].isin(pos.index).to_numpy()
    ri = np.flatnonzero(keep)
    gi = pos.loc[real_idx["crop_id"].to_numpy()[keep]].to_numpy()
    return real_idx.iloc[ri].reset_index(drop=True), XR[ri], XG[gi]


def _ci(v) -> Tuple[float, float]:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return ((float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            if len(v) else (np.nan, np.nan))


# --------------------------------------------------------------------------- #
def cmd_table(a) -> int:
    real_idx = pd.read_parquet(os.path.join(a.dir, "index.parquet"))
    real_idx["crop_id"] = real_idx["crop_id"].astype(str)
    XR = np.load(os.path.join(a.dir, "feat_real.npy"))
    paths = sorted(glob.glob(os.path.join(a.dir, "feat_gen__*.npy")))
    if a.gen:
        want = set(a.gen.split(","))
        paths = [p for p in paths if os.path.basename(p)[10:-4] in want]
    if not paths:
        raise SystemExit(f"no feat_gen__*.npy in {a.dir} (matching --gen)")
    sub = (set(pd.read_csv(a.subset)["crop_id"].astype(str))
           if a.subset else None)
    rows = []
    for path in paths:
        name, gidx, XG = load_gen(path)
        idx, X, Y = join(real_idx, XR, gidx, XG)
        P = Paired(idx, X, Y)
        r = {"name": name, "n": len(idx), "n_wells": P.W,
             "n_units": idx["unit_id"].nunique() if "unit_id" in idx else np.nan,
             "fid": fid(X, Y)}
        if len(idx) < len(real_idx):
            print(f"  ! {name}: {len(real_idx) - len(idx)} real crop(s) have no "
                  f"generated partner")
        if sub is not None:
            s = idx["crop_id"].isin(sub).to_numpy()
            r.update(n_sub=int(s.sum()), fid_sub=fid(X[s], Y[s]))
        r["fid_inf"], r["fid_inf_slope"] = fid_inf(X, Y, seed=a.seed)
        r["kid_tm"], r["kid_tm_std"] = kid_subsets(X, Y, seed=a.seed)
        r["kid_u"] = P.u_stat("kid")
        r["energy"] = P.u_stat("energy")
        rng = np.random.default_rng(a.seed)
        bk, be, bf = [], [], []
        for b in range(a.boot):
            c = draw_counts(P.well_unit, rng)
            bk.append(P.u_stat("kid", c)); be.append(P.u_stat("energy", c))
            if b < a.boot_fid:
                ii = P.rows_of(c)
                bf.append(fid(X[ii], Y[ii]))
        (r["kid_u_lo"], r["kid_u_hi"]), (r["energy_lo"], r["energy_hi"]) = \
            _ci(bk), _ci(be)
        bf = np.asarray(bf, float)
        r["fid_se"] = float(bf.std(ddof=1)) if len(bf) > 1 else np.nan
        r["fid_lo"] = r["fid"] - 1.96 * r["fid_se"]
        r["fid_hi"] = r["fid"] + 1.96 * r["fid_se"]
        r["fid_boot_shift"] = float(bf.mean() - r["fid"]) if len(bf) else np.nan
        rows.append(r)
        print(f"  {name:<40} n {r['n']:>6}  FID {r['fid']:8.3f} "
              f"[{r['fid_lo']:.2f}, {r['fid_hi']:.2f}]  inf {r['fid_inf']:8.3f}"
              + (f"  sub({r['n_sub']}) {r['fid_sub']:8.3f}" if sub else "")
              + f"  KIDu {r['kid_u']:.5f}  KIDtm {r['kid_tm']:.5f}"
              f"  E {r['energy']:.4f}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(a.out, index=False)
    print(f"-> {a.out}")
    return 0


def cmd_latent(a) -> int:
    idx = pd.read_parquet(os.path.join(a.dir, "index.parquet"))
    zr = np.load(os.path.join(a.dir, "z_real.npy")).astype(np.float32)
    zr = zr.reshape(len(zr), -1)
    rows = []
    for path in sorted(glob.glob(os.path.join(a.dir, "z_gen__*.npy"))):
        name = os.path.basename(path)[len("z_gen__"):-4]
        zg = np.load(path).astype(np.float32).reshape(len(zr), -1)
        P = Paired(idx, zr, zg)
        r = {"name": name, "n": len(zr), "n_wells": P.W,
             "energy": P.u_stat("energy")}
        # eval_flow.energy_distance, replayed: 2,000 per side, diagonal kept
        rs = np.random.default_rng(0)
        ia = rs.choice(len(zr), min(2000, len(zr)), replace=False)
        ib = rs.choice(len(zg), min(2000, len(zg)), replace=False)
        A, B = zr[ia].astype(np.float64), zg[ib].astype(np.float64)
        r["energy_evalflow"] = float(2 * _dist(A, B).mean() - _dist(A, A).mean()
                                     - _dist(B, B).mean())
        rng = np.random.default_rng(a.seed)
        r["energy_lo"], r["energy_hi"] = _ci(
            [P.u_stat("energy", draw_counts(P.well_unit, rng))
             for _ in range(a.boot)])
        rows.append(r)
        print(f"  {name:<40} energy {r['energy']:.4f} [{r['energy_lo']:.4f}, "
              f"{r['energy_hi']:.4f}]   eval_flow's estimator "
              f"{r['energy_evalflow']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"-> {a.out}")
    return 0


# --------------------------------------------------------------------------- #
def cmd_fidc(a) -> int:
    ri = pd.read_parquet(os.path.join(a.real, "index.parquet"))
    ri["crop_id"] = ri["crop_id"].astype(str)
    XR = np.load(os.path.join(a.real, "feat_real.npy"))
    g = a.group
    ev = ri["split"] == a.eval_split
    gens = [load_gen(p) for p in a.gen]
    rng = np.random.default_rng(a.seed)
    rows = []
    for p, d in ri.groupby(g, sort=True):
        iv = d.index[ev.loc[d.index]].to_numpy()
        it = d.index[~ev.loc[d.index]].to_numpy()
        n = len(iv)
        r = {g: p, "n": n, "n_other": len(it)}
        r["rr_matched"] = (fid(XR[iv], XR[rng.choice(it, n, replace=False)])
                           if 2 <= n <= len(it) else np.nan)
        if n >= 4:
            h = rng.permutation(iv)
            r["rr_half"] = fid(XR[h[: n // 2]], XR[h[n // 2:]])
        else:
            r["rr_half"] = np.nan
        for name, gidx, XG in gens:
            pos = pd.Series(np.arange(len(gidx)), index=gidx["crop_id"])
            pos = pos[~pos.index.duplicated()]
            cid = ri.loc[iv, "crop_id"]
            k = cid.isin(pos.index).to_numpy()
            r[f"gen_{name}"] = fid(XR[iv[k]], XG[pos.loc[cid[k]].to_numpy()])
            r[f"n_gen_{name}"] = int(k.sum())
        rows.append(r)
    P = pd.DataFrame(rows)
    cols = ["rr_matched", "rr_half"] + [f"gen_{nm}" for nm, _, _ in gens]
    pd.set_option("display.width", 200)
    q = P["n"].describe(percentiles=[.1, .5, .9])
    print(f"== {len(P)} perturbations: eval crops per perturbation min "
          f"{int(q['min'])} / p10 {q['10%']:.0f} / median {q['50%']:.0f} / "
          f"p90 {q['90%']:.0f} / max {int(q['max'])}; {int((P.n < 50).sum())} "
          f"under 50 ==")
    S = []
    for c in cols:
        v = P[c].dropna()
        s = {"column": c, "fidc": float(v.mean()), "n_perts": len(v),
             "spearman_vs_n": float(pd.Series(v).corr(P.loc[v.index, "n"],
                                                      method="spearman"))}
        if a.n_sub and len(v) >= a.n_sub:
            dr = np.array([v.to_numpy()[rng.choice(len(v), a.n_sub, replace=False)]
                           .mean() for _ in range(a.draws)])
            s.update(draw_mean=dr.mean(), draw_sd=dr.std(ddof=1),
                     draw_lo=np.percentile(dr, 2.5), draw_hi=np.percentile(dr, 97.5),
                     draw_min=dr.min(), draw_max=dr.max())
        S.append(s)
    S = pd.DataFrame(S)
    print("\nFIDc = unweighted mean over perturbations (eval_fid.py). rr_* = "
          "real vs real, nothing generated.")
    print(S.round(3).to_string(index=False))
    curve = []
    for ns in a.n_curve:
        v = []
        for _, d in ri.groupby(g):
            if len(d) >= 2 * ns:
                h = rng.permutation(d.index.to_numpy())
                v.append(fid(XR[h[:ns]], XR[h[ns: 2 * ns]]))
        curve.append({"n_per_side": ns, "rr_fid_mean": np.mean(v) if v else np.nan,
                      "n_perts": len(v)})
    C = pd.DataFrame(curve)
    print("\nreal vs real, same perturbation, n crops per side (eval + other "
          "splits pooled):")
    print(C.round(3).to_string(index=False))
    P.to_csv(a.out, index=False)
    S.to_csv(a.out.replace(".csv", "_summary.csv"), index=False)
    C.to_csv(a.out.replace(".csv", "_ncurve.csv"), index=False)
    print(f"-> {a.out} (+ _summary, _ncurve)")
    return 0


def cmd_spread(a) -> int:
    pd.set_option("display.width", 200)
    if a.csv:
        E = pd.concat([pd.read_csv(p).assign(file=os.path.basename(p))
                       for p in a.csv], ignore_index=True)
        if a.arm:
            E = E[E["arm"] == a.arm]
        mets = [m for m in ("fid_all", "kid_all", "energy_latent_all",
                            "precision", "recall", "density", "coverage",
                            "moa_cf_acc_gen", "moa_cf_acc_gen_sub",
                            "moa_cf_f1_macro_gen", "moa_cfood_acc_gen",
                            "clf_top5_gen") if m in E.columns and E[m].notna().any()]
        print(f"== {len(E)} eval_flow rows ({', '.join(E.file)}) ==")
        print(E[mets].agg(["mean", "std", "min", "max"]).T.round(5).to_string())
    if a.table:
        T = pd.read_csv(a.table)
        T = T[T["name"].str.contains(a.match)] if a.match else T
        mets = [m for m in ("fid", "fid_sub", "fid_inf", "kid_u", "kid_tm",
                            "energy") if m in T.columns]
        print(f"\n== {len(T)} feature rows ({', '.join(T.name)}) ==")
        print(T[mets].agg(["mean", "std", "min", "max"]).T.round(5).to_string())
        if "fid_lo" in T.columns:
            print(f"  mean well-bootstrap FID half-width "
                  f"{((T.fid_hi - T.fid_lo) / 2).mean():.3f} (per-row interval)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("table")
    t.add_argument("--dir", required=True)
    t.add_argument("--gen", default=None, help="comma list of names; default all")
    t.add_argument("--subset", default=None, help="csv with crop_id")
    t.add_argument("--boot", type=int, default=1000,
                   help="well-bootstrap draws for KID_u and energy (cheap)")
    t.add_argument("--boot_fid", type=int, default=100,
                   help="how many of them also recompute FID (~3 s each at "
                        "7k crops); 0 = no FID interval")
    t.add_argument("--out", required=True)
    l_ = sp.add_parser("latent")
    l_.add_argument("--dir", required=True)
    l_.add_argument("--boot", type=int, default=1000)
    l_.add_argument("--out", required=True)
    f = sp.add_parser("fidc")
    f.add_argument("--real", required=True, help="inception_feats.py real's dir")
    f.add_argument("--gen", nargs="*", default=[],
                   help="feat_gen__<name>.npy files, joined on crop_id")
    f.add_argument("--eval_split", default="val")
    f.add_argument("--group", default="compound")
    f.add_argument("--n_sub", type=int, default=0,
                   help="CellFlux's RxRx1 FIDc averages a random 50")
    f.add_argument("--draws", type=int, default=1000)
    f.add_argument("--n_curve", type=int, nargs="*",
                   default=[8, 16, 32, 64, 128, 256])
    f.add_argument("--out", required=True)
    s = sp.add_parser("spread")
    s.add_argument("--csv", nargs="*", default=[])
    s.add_argument("--arm", default=None, help="keep this arm's eval rows")
    s.add_argument("--table", default=None)
    s.add_argument("--match", default=None, help="regex on table names")
    for x in (t, l_, f):
        x.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    return {"table": cmd_table, "latent": cmd_latent, "fidc": cmd_fidc,
            "spread": cmd_spread}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
