"""CellProfiler output -> per-crop features -> effect-size metrics.

CPU only, runs on the csvs CellProfiler wrote.

    python cellprofiler_eval/cp_features.py aggregate  --cp_out CPOUT --out percrop.parquet
    python cellprofiler_eval/cp_features.py roundtrip  --features percrop.parquet --out filter.csv
    python cellprofiler_eval/cp_features.py effects    --features percrop.parquet --filter filter.csv

One feature table holds four populations: ``real`` (treated crops), ``control``
(same-plate DMSO), ``recon`` (VAE reconstruction of ``real``) and ``gen``.
Effects are computed against same-plate controls per plate, then averaged
across plates, never pooled.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

OBJECT_TABLES = ("Nuclei", "Cells", "Cytoplasm")
DROP_PREFIX = ("ImageNumber", "ObjectNumber", "Number_Object_Number",
               "Parent_", "Children_", "Location_", "Metadata_", "Group_",
               "Series_", "Frame_", "URL_", "PathName_", "FileName_",
               "ExecutionTime_", "ModuleError_", "MD5Digest_", "Scaling_")
META = ("crop_id", "population", "arm", "unit_id", "compound", "dose",
        "moa", "plate", "well", "well_id", "s", "y")


def _is_feature(c: str) -> bool:
    return not any(c.startswith(p) for p in DROP_PREFIX)


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
def cmd_aggregate(a) -> int:
    dirs = a.cp_out if isinstance(a.cp_out, (list, tuple)) else [a.cp_out]
    dirs = sorted({d for pat in dirs for d in (glob.glob(pat) or [pat])})
    if len(dirs) > 1:
        print(f"[aggregate] {len(dirs)} CellProfiler output dirs", flush=True)
    parts = []
    for d in dirs:
        parts.append(_aggregate_one(a, d) if len(dirs) == 1
                     else _aggregate_child(a, d))
    out = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    if out.duplicated(["crop_id", "population", "arm"]).any():
        n = int(out.duplicated(["crop_id", "population", "arm"]).sum())
        raise SystemExit(
            f"{n} (crop_id, population, arm) rows appear twice across "
            f"{len(dirs)} output dirs. The chunks overlap -- check the -f/-l "
            f"ranges cp_run.sh used, or aggregate one dir at a time.")
    n_feat = sum(1 for c in out.columns if c not in META and c != "ImageNumber")
    print(f"[aggregate] TOTAL {len(out)} crops x {n_feat} features")
    if "population" in out:
        print(out.groupby("population").size().to_string())
    out.to_parquet(a.out, index=False)
    print(f"-> {a.out}")
    return 0


def _aggregate_child(a, cp_out: str) -> pd.DataFrame:
    """_aggregate_one in a child process, one chunk dir per child, so an
    unreadable chunk fails on its own and names its dir."""
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "part.parquet")
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "aggregate",
             "--cp_out", cp_out, "--out", f, "--prefix", a.prefix,
             "--image_csv", a.image_csv, "--pool", a.pool,
             "--crop_size", str(a.crop_size)])
        if r.returncode != 0 or not os.path.exists(f):
            raise SystemExit(
                f"{cp_out} died in its own process (exit {r.returncode}). "
                f"That dir's {a.image_csv} is the bad one -- rerun its "
                f"CellProfiler chunk.")
        return pd.read_parquet(f)


def _aggregate_one(a, cp_out: str) -> pd.DataFrame:
    img_path = os.path.join(cp_out, a.image_csv)
    if not os.path.exists(img_path):
        cand = glob.glob(os.path.join(cp_out, "*Image.csv"))
        if not cand:
            raise SystemExit(f"no {a.image_csv} under {cp_out}. "
                             f"Found: {sorted(os.listdir(cp_out))[:20]}")
        img_path = cand[0]
    img = pd.read_csv(img_path)
    print(f"[aggregate] {os.path.basename(img_path)}: {len(img)} images")

    meta_cols = [c for c in img.columns if c.startswith("Metadata_")]
    if "Metadata_crop_id" not in meta_cols:
        raise SystemExit(
            "Image.csv has no Metadata_crop_id -- ExportToSpreadsheet was run "
            "without 'add image metadata columns', or LoadData did not get the "
            "csv cp_io wrote. Without it nothing can be joined back to a crop.")
    out = img[["ImageNumber"] + meta_cols].copy()
    out.columns = ["ImageNumber"] + [c[len("Metadata_"):] for c in meta_cols]

    # image-level features (granularity, image quality, image colocalization)
    imf = [c for c in img.columns if _is_feature(c)]
    out = pd.concat([out, img[imf].add_prefix("Image_")], axis=1)
    print(f"[aggregate]   image-level features: {len(imf)}")

    for tbl in OBJECT_TABLES:
        p = os.path.join(cp_out, f"{a.prefix}{tbl}.csv")
        if not os.path.exists(p):
            print(f"[aggregate]   {tbl}.csv absent, skipped")
            continue
        d = pd.read_csv(p)
        feats = [c for c in d.columns if _is_feature(c)]
        if a.pool == "center":
            d = _pick_center(d, img, a.crop_size)
            g = d.set_index("ImageNumber")[feats]
        else:
            g = d.groupby("ImageNumber")[feats].mean()
        g = g.add_prefix(f"{tbl}_")
        out = out.merge(g, left_on="ImageNumber", right_index=True, how="left")
        print(f"[aggregate]   {tbl}: {len(d)} objects -> {len(feats)} features "
              f"({a.pool}-pooled)")

    print(f"[aggregate]   -> {len(out)} crops")
    return out


def _pick_center(d: pd.DataFrame, img: pd.DataFrame, crop_size: int):
    """One object per image: the one nearest the crop centre (the cell the
    crop is about, rather than a mean over it and its neighbours)."""
    if not {"Location_Center_X", "Location_Center_Y"} <= set(d.columns):
        raise SystemExit("--pool center needs Location_Center_X/Y in the "
                         "object table (MeasureObjectSizeShape provides them)")
    c = crop_size / 2.0
    d = d.assign(_r=np.hypot(d.Location_Center_X - c, d.Location_Center_Y - c))
    return d.sort_values("_r").drop_duplicates("ImageNumber", keep="first")


# --------------------------------------------------------------------------- #
# helpers shared by the metrics
# --------------------------------------------------------------------------- #
def load_features(path: str):
    df = pd.read_parquet(path)
    # real / recon / control are the same image whichever arm wrote them: keep
    # one row per crop. gen rows are per arm and stay.
    if {"population", "crop_id"} <= set(df.columns):
        shared = (df["population"] != "gen").to_numpy()
        keep = df[shared].drop_duplicates(["population", "crop_id"])
        if len(keep) < shared.sum():
            print(f"[features] {int(shared.sum()) - len(keep)} duplicate "
                  f"real/recon/control row(s) dropped: one per crop")
            df = pd.concat([keep, df[~shared]], ignore_index=True)
    feats = [c for c in df.columns
             if c not in META and c != "ImageNumber"
             and pd.api.types.is_numeric_dtype(df[c])]
    return df, feats


def _pivot(df, feats, pop_a, pop_b):
    """Matched (pop_a, pop_b) frames on shared crop_ids, same row order."""
    a = df[df.population == pop_a].drop_duplicates("crop_id").set_index("crop_id")
    b = df[df.population == pop_b].drop_duplicates("crop_id").set_index("crop_id")
    ids = a.index.intersection(b.index)
    return a.loc[ids, feats], b.loc[ids, feats], ids


# --------------------------------------------------------------------------- #
# channels
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# roundtrip
# --------------------------------------------------------------------------- #
def cmd_roundtrip(a) -> int:
    from scipy.stats import spearmanr
    df, feats = load_features(a.features)
    R, C, ids = _pivot(df, feats, "real", "recon")
    if len(ids) < a.min_n:
        raise SystemExit(f"only {len(ids)} crops have both real and recon; "
                         f"need --min_n {a.min_n}. Lower it only for a smoke "
                         f"test -- a rho from a handful of crops decides which "
                         f"features the whole analysis runs on.")
    print(f"[roundtrip] {len(ids)} crops, {len(feats)} features")

    out = []
    for f in feats:
        x, y = R[f].to_numpy(float), C[f].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < a.min_n or np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0:
            out.append((f, np.nan, int(m.sum())))
            continue
        out.append((f, float(spearmanr(x[m], y[m]).statistic), int(m.sum())))
    t = pd.DataFrame(out, columns=["feature", "rho", "n"])
    t["keep"] = t.rho >= a.rho_min
    t = t.sort_values("rho", ascending=False)
    t.to_csv(a.out, index=False)

    k = int(t.keep.sum())
    print(f"[roundtrip] rho >= {a.rho_min}: {k}/{len(t)} features kept "
          f"({k / len(t):.1%}); median rho {t.rho.median():.3f}")
    for grp in ("AreaShape", "Intensity", "Texture", "Granularity",
                "Correlation", "ImageQuality"):
        s = t[t.feature.str.contains(grp)]
        if len(s):
            print(f"    {grp:14s} {int(s.keep.sum()):4d}/{len(s):4d} kept, "
                  f"median rho {s.rho.median():+.3f}")
    if k == 0:
        print("  ! nothing survived. Either the VAE destroys this feature set, "
              "or real and recon were not matched on crop_id -- check that "
              "both populations are in the table before lowering --rho_min.")
    print(f"-> {a.out}")
    return 0


# --------------------------------------------------------------------------- #
# effects
# --------------------------------------------------------------------------- #
ARM_RE = (
    # dil_q0.1_g1_oracle__a2 -> family g1_oracle__a2, q 0.1
    (r"^dil_q(?P<q>[0-9.]+)_g(?P<g>[0-9.]+)(?P<rest>.*)$", "g{g}{rest}"),
    # dilreal_q0.25 -> family dilreal, q 0.25
    (r"^dilreal_q(?P<q>[0-9.]+)(?P<rest>.*)$", "dilreal{rest}"),
    # gamma0__q1 -> family g0 at q 1: the undiluted arm on the study units
    (r"^gamma(?P<g>[0-9.]+)__q1(?P<rest>.*)$", "g{g}{rest}"),
    # any other gamma arm sits at q 1
    (r"^gamma(?P<g>[0-9.]+)(?P<rest>.*)$", "g{g}{rest}"),
)


def family_q(arm: str):
    import re
    for pat, fam in ARM_RE:
        m = re.match(pat, str(arm))
        if m:
            d = m.groupdict()
            return fam.format(**d), float(d.get("q") or 1.0)
    return str(arm), float("nan")


def _plate_stats(treated: pd.DataFrame, control: pd.DataFrame, cols,
                 offset: Optional[pd.DataFrame] = None,
                 min_ctl: int = 10, min_trt: int = 5):
    """Control-SD z and pooled-SD Cohen's d, per plate, averaged over plates.

    ``offset``: a population whose per-plate mean replaces the control mean in
    the numerator (the decoded controls, i.e. passthrough), so a constant VAE
    decode shift cancels. The denominator stays the raw control SD.
    """
    zs, ds = [], []
    for plate, t in treated.groupby("plate"):
        c = control[control.plate == plate]
        if len(c) < min_ctl or len(t) < min_trt:
            continue
        mc, sc = c[cols].mean(), c[cols].std(ddof=1)
        mt, st = t[cols].mean(), t[cols].std(ddof=1)
        if offset is not None:
            o = offset[offset.plate == plate]
            if len(o) < min_trt:
                continue
            mc = o[cols].mean()
        zs.append((mt - mc) / (sc + 1e-12))
        ds.append((mt - mc) / (np.sqrt((st ** 2 + sc ** 2) / 2) + 1e-12))
    if not zs:
        return None, None
    return pd.concat(zs, axis=1).mean(axis=1), pd.concat(ds, axis=1).mean(axis=1)


def _load_effect_inputs(a):
    import re
    df, feats = load_features(a.features)
    if a.filter and os.path.exists(a.filter):
        keep = pd.read_csv(a.filter)
        keep = set(keep[keep.keep].feature)
        feats = [f for f in feats if f in keep]
        print(f"[effects] round-trip filter: {len(feats)} features")
    if a.regex:
        feats = [f for f in feats if re.search(a.regex, f)]
        print(f"[effects] --regex {a.regex!r}: {len(feats)} features")
    feats = [f for f in feats if df[f].notna().mean() > 0.5
             and df[f].std() > 0]
    if not feats:
        raise SystemExit("no feature left after the filter / regex")
    return df, feats


def _unit_effects(arm_df, real, ctl, feats, min_z, min_crops, offset=None,
                  real_ref=None):
    """Rows (unit, feature, z_real, z_gen, d_real, d_gen) for one arm."""
    rows = []
    for unit, r in real.groupby("unit_id"):
        g = arm_df[arm_df.unit_id == unit]
        if len(g) < min_crops or len(r) < min_crops:
            continue
        # controls from both sides' plates: an arm on other plates than the
        # reference needs its own plates' controls
        c = ctl[ctl.plate.isin(set(r.plate.unique()) | set(g.plate.unique()))]
        rr = real_ref[unit] if real_ref is not None and unit in real_ref else None
        zr, dr = rr if rr is not None else _plate_stats(r, c, feats)
        o = (offset[offset.unit_id == unit] if offset is not None else None)
        zg, dg = _plate_stats(g, c, feats, offset=o)
        if zr is None or zg is None:
            continue
        m = (np.isfinite(zr) & np.isfinite(zg) & (zr.abs() >= min_z))
        for f in zr.index[m]:
            rows.append((unit, f, zr[f], zg[f], dr[f], dg[f]))
    return pd.DataFrame(rows, columns=["unit_id", "feature", "z_real", "z_gen",
                                       "d_real", "d_gen"])


def _summ(t: pd.DataFrame) -> Dict[str, float]:
    """Arm-level numbers from its (unit, feature) cells."""
    if not len(t):
        return {"n_units": 0, "n_cells": 0}
    ratio = t.z_gen / t.z_real
    g = t.assign(zz=t.z_gen * t.z_real, z2=t.z_real ** 2,
                 dd=t.d_gen * t.d_real, d2=t.d_real ** 2).groupby("unit_id")
    s_ = g[["zz", "z2", "dd", "d2"]].sum()
    pu, du = s_.zz / s_.z2, s_.dd / s_.d2
    return {"n_units": int(t.unit_id.nunique()), "n_cells": int(len(t)),
            "z_ratio_median": float(ratio.median()),
            "z_ratio_q25": float(ratio.quantile(.25)),
            "z_ratio_q75": float(ratio.quantile(.75)),
            "z_proj_median": float(pu.median()),
            "d_ratio_median": float((t.d_gen / t.d_real).median()),
            "d_proj_median": float(du.median()),
            "sign_flip": float((ratio < 0).mean())}


def eq61(q: float, d1: np.ndarray, pi: float = 1.0) -> np.ndarray:
    """Cohen's d (pooled SD) of a set diluted to responder fraction q*pi, from
    d1 = the same set's d at q=1. Two components with a common variance s^2
    and a mean shift D = delta*s on the responders: the mixture at responder
    fraction p has mean p*D and variance s^2 + p(1-p)D^2, so
        d(p) = p*delta / sqrt(1 + p(1-p) delta^2 / 2)
    and d1 = d(pi) fixes delta. pi=1 (the default) reads d1 as a pure
    responder set.
    """
    d1 = np.asarray(d1, dtype=float)
    den = pi ** 2 - d1 ** 2 * pi * (1 - pi) / 2
    delta = np.sign(d1) * np.sqrt(np.where(den > 0, d1 ** 2 / np.where(den > 0, den, 1), np.nan))
    p = q * pi
    return p * delta / np.sqrt(1 + p * (1 - p) * delta ** 2 / 2)


def cmd_effects(a) -> int:
    df, feats = _load_effect_inputs(a)
    ctl = df[df.population == "control"]
    real = df[df.population == "real"]
    arms = {}
    gen = df[df.population == "gen"]
    for arm, g in gen.groupby("arm"):
        arms[str(arm)] = g
    if (df.population == "recon").any():
        arms["recon"] = df[df.population == "recon"]
    if a.arms:
        import fnmatch
        pats = a.arms.split(",")
        arms = {k: v for k, v in arms.items()
                if any(fnmatch.fnmatch(k, p) for p in pats) or k == "recon"}
    offset = None
    if a.ctrl_ref == "decoded":
        pt = [k for k in arms if "passthrough" in k]
        if not pt:
            raise SystemExit("--ctrl_ref decoded needs a passthrough arm")
        offset = arms[pt[0]]
        print(f"[effects] decoded-control reference: {pt[0]} replaces the "
              f"control mean for every decoded population")
    # the real effect is the same for every arm; compute it once per unit
    real_ref = {}
    for unit, r in real.groupby("unit_id"):
        c = ctl[ctl.plate.isin(r.plate.unique())]
        zr, dr = _plate_stats(r, c, feats)
        if zr is not None:
            real_ref[unit] = (zr, dr)
    print(f"[effects] {len(real_ref)} units, {len(feats)} features, "
          f"{len(arms)} arms, |real z| >= {a.min_z}")

    cells, summ = [], []
    for arm, g in sorted(arms.items()):
        dec = offset if (offset is not None and arm != "real") else None
        t = _unit_effects(g, real, ctl, feats, a.min_z, a.min_crops,
                          offset=dec, real_ref=real_ref)
        t.insert(0, "arm", arm)
        fam, q = family_q(arm)
        cells.append(t)
        summ.append({"arm": arm, "family": fam, "q": q, **_summ(t)})
    C = pd.concat(cells, ignore_index=True)
    S = pd.DataFrame(summ)

    # eq. (6.1): each family's q=1 member fixes d1, per (unit, feature)
    S["d_eq61_median"] = np.nan
    S["d_meas_over_eq61"] = np.nan
    for fam, fs in S.groupby("family"):
        one = fs[fs.q == 1.0]
        if not len(one) or len(fs) < 2:
            continue
        base = C[C.arm == one.arm.iloc[0]].set_index(["unit_id", "feature"]).d_gen
        for i, r in fs[fs.q < 1.0].iterrows():
            t = C[C.arm == r.arm].set_index(["unit_id", "feature"])
            d1 = base.reindex(t.index)
            ok = d1.notna()
            pred = eq61(r.q, d1[ok].to_numpy(), a.pi_true)
            S.loc[i, "d_eq61_median"] = float(np.nanmedian(pred / t.d_real[ok]))
            S.loc[i, "d_meas_over_eq61"] = float(np.nanmedian(t.d_gen[ok] / pred))

    pd.set_option("display.width", 200)
    show = ["arm", "q", "n_units", "n_cells", "z_ratio_median", "z_ratio_q25",
            "z_ratio_q75", "z_proj_median", "d_ratio_median", "d_eq61_median",
            "sign_flip"]
    print("\n== recovered effect, real = 1 (z = mean shift / control SD, "
          f"cells with |real z| >= {a.min_z}) ==")
    for fam, fs in S.sort_values(["family", "q"]).groupby("family", sort=False):
        print(f"\n  -- {fam}")
        print(fs[show].round(3).to_string(index=False))

    seeds = S.arm.str.extract(r"^(.*?)_s\d+$")[0]
    if seeds.notna().any():
        S["seed_group"] = seeds.fillna(S.arm)
        grp = S[S.seed_group.isin(seeds.dropna())].groupby("seed_group")
        print("\n== seed groups (the unsuffixed arm is seed 0) ==")
        print(grp[["z_proj_median", "z_ratio_median", "d_ratio_median"]]
              .agg(["mean", "std", "count"]).round(3).to_string())

    fits = []
    for fam, fs in S.groupby("family"):
        fs = fs[np.isfinite(fs.q)]
        if fs.q.nunique() < 3:
            continue
        for stat in ("z_ratio_median", "z_proj_median", "d_ratio_median"):
            x, y = fs.q.to_numpy(), fs[stat].to_numpy()
            ok = np.isfinite(y)
            if ok.sum() < 3:
                continue
            b, c0 = np.polyfit(x[ok], y[ok], 1)
            b0 = float((x[ok] * y[ok]).sum() / (x[ok] ** 2).sum())
            fits.append({"family": fam, "stat": stat, "n_q": int(ok.sum()),
                         "slope": float(b), "intercept": float(c0),
                         "slope_through_origin": b0})
    if fits:
        F = pd.DataFrame(fits)
        print("\n== recovered effect vs q, per family (OLS over the q grid) ==")
        print(F.round(3).to_string(index=False))
        print("\n  Proposition 3: g0 tracks dilreal (slope ~1, intercept ~0); "
              "g1 is flatter. dilreal is linear in q by construction, so its "
              "fit is the pipeline check.")
        F.to_csv(a.out.replace(".csv", "_fits.csv"), index=False)
    S.to_csv(a.out, index=False)
    C.to_csv(a.out.replace(".csv", "_cells.csv.gz"), index=False)
    print(f"\n-> {a.out}  (+ _cells.csv.gz, _fits.csv)")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("aggregate")
    q.add_argument("--cp_out", required=True, nargs="+",
                   help="CellProfiler output dir(s); globs are "
                        "expanded, so cp_run.sh's chunk_* dirs "
                        "aggregate in one call. Chunks are joined "
                        "on crop_id, never on ImageNumber, which "
                        "restarts per chunk.")
    q.add_argument("--out", default="percrop_features.parquet")
    q.add_argument("--prefix", default="", help="ExportToSpreadsheet prefix")
    q.add_argument("--image_csv", default="Image.csv")
    q.add_argument("--pool", default="mean", choices=("mean", "center"),
                   help="mean = the protocol spec (mean over cells in the "
                        "crop); center = the object nearest the crop centre, "
                        "i.e. the cell the crop is about")
    q.add_argument("--crop_size", type=int, default=96)
    q.set_defaults(fn=cmd_aggregate)

    q = sub.add_parser("roundtrip")
    q.add_argument("--features", required=True)
    q.add_argument("--out", default="feature_filter.csv")
    q.add_argument("--rho_min", type=float, default=0.5)
    q.add_argument("--min_n", type=int, default=30,
                   help="minimum matched real/recon crops. Lower only for a "
                        "smoke test.")
    q.set_defaults(fn=cmd_roundtrip)

    q = sub.add_parser("effects")
    q.add_argument("--features", required=True)
    q.add_argument("--filter", default=None,
                   help="roundtrip's csv: only features the VAE preserves")
    q.add_argument("--regex", default=None,
                   help="feature-name regex, e.g. '^Image_' (whole crop)")
    q.add_argument("--min_z", type=float, default=1.0,
                   help="keep (unit, feature) cells with |real z| >= this")
    q.add_argument("--min_crops", type=int, default=20)
    q.add_argument("--out", default="effects.csv")
    q.add_argument("--arms", default=None,
                   help="comma-separated fnmatch patterns; default all")
    q.add_argument("--ctrl_ref", default="raw", choices=("raw", "decoded"),
                   help="decoded: the passthrough arm's mean replaces the "
                        "control mean for decoded populations")
    q.add_argument("--pi_true", type=float, default=1.0,
                   help="responder fraction of the q=1 set")
    q.set_defaults(fn=cmd_effects)

    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
