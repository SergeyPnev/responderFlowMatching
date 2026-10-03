"""The 2048-d Inception pool features FID / KID are computed from, on disk, so
``dist_metrics.py`` can re-estimate every distribution metric on CPU.

    eval_flow.py ... --dump_feats DIR          # gen per arm/setting + real
    python evaluation/inception_feats.py real --config C --out DIR [--splits val,train]
    python evaluation/inception_feats.py pngs --png_root R --index I --out DIR --name N

``DIR/index.parquet`` holds one row per feature row (crop_id + metadata),
``DIR/feat_real.npy`` the real crops, ``DIR/feat_gen__<name>.npy`` one
generated set; ``pngs`` writes its own ``DIR/index__<name>.parquet``. Images
are preprocessed exactly as eval_flow's FID sees them.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def inception_net(device):
    """torchmetrics' FID Inception (pool3, 2048-d), the one eval_flow's FID uses."""
    os.environ.setdefault("USE_TF", "0")
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        return FrechetInceptionDistance(normalize=True).to(device).inception
    except Exception as e:                                        # noqa: BLE001
        raise SystemExit(f"Inception features need torchmetrics + "
                         f"torch-fidelity ({type(e).__name__}: {str(e)[:160]})")


@torch.no_grad()
def features(net, imgs01: torch.Tensor) -> np.ndarray:
    """[B,3,H,W] in [0,1] -> [B,2048] float32, converted exactly as
    FrechetInceptionDistance.update(normalize=True) converts."""
    return net((imgs01 * 255).byte()).reshape(len(imgs01), -1).float().cpu().numpy()


def index_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("crop_id", "unit_id", "compound", "dose", "moa",
                        "plate", "well", "well_id", "split", "s")
            if c in df.columns]
    out = df[cols].copy()
    out["crop_id"] = out["crop_id"].astype(str)
    return out.reset_index(drop=True)


def write_feats(out: str, name: str, idx: pd.DataFrame,
                f_real: Optional[np.ndarray], f_gen: np.ndarray) -> None:
    """eval_flow's writer: index + real once per eval set, gen per name."""
    os.makedirs(out, exist_ok=True)
    ip = os.path.join(out, "index.parquet")
    if not os.path.exists(ip):
        index_frame(idx).to_parquet(ip, index=False)
    else:
        have = pd.read_parquet(ip, columns=["crop_id"])["crop_id"].astype(str)
        if not have.equals(idx["crop_id"].astype(str).reset_index(drop=True)):
            raise SystemExit(f"{ip} holds a different eval set; point "
                             f"--dump_feats somewhere else")
    rp = os.path.join(out, "feat_real.npy")
    if f_real is not None and not os.path.exists(rp):
        np.save(rp, f_real.astype(np.float32))
    np.save(os.path.join(out, f"feat_gen__{name}.npy"), f_gen.astype(np.float32))
    print(f"[dump_feats] {out}  feat_gen__{name}.npy {f_gen.shape}")


# --------------------------------------------------------------------------- #
class _Crops(torch.utils.data.Dataset):
    def __init__(self, paths, tf, n_ch):
        self.paths, self.tf, self.n_ch = list(paths), tf, n_ch

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from evaluation.eval_flow import fid_view
        x = torch.clamp(self.tf(image=np.load(self.paths[i]))["image"].float(),
                        0.0, 1.0)
        return fid_view(x[None])[0] if self.n_ch != 3 else x


class _Pngs(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from evaluation.cellflux_moa import read_png
        return read_png(self.paths[i]).float() / 255.0


def _run(net, ds, device, batch_size, num_workers) -> np.ndarray:
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers)
    out, n = [], 0
    for x in dl:
        out.append(features(net, x.to(device)))
        n += len(x)
        if (n // batch_size) % 50 == 0:
            print(f"  {n}/{len(ds)}")
    return np.concatenate(out)


def cmd_real(a) -> None:
    """Real treated crops of the eval perturbations, from several splits
    (a second real sample per perturbation, for real-vs-real FID). QC and
    scoring rules are eval_flow's own."""
    import yaml
    from flow.crop_flow_dataset import load_training_index
    from datasets.bbbc021_dataset import default_bbbc_transform

    cfg = yaml.safe_load(open(a.config))
    fx = cfg.get("eval_flow_index") or cfg["flow_index"]
    cs = cfg.get("eval_crop_scores") or cfg["crop_scores"]
    cu = cfg.get("eval_crop_universe") or cfg.get("crop_universe")
    ev = cfg.get("eval_split", "val")
    splits = [ev] + [s for s in a.splits.split(",") if s and s != ev]
    trt, _, _ = load_training_index(
        fx, cs, train_splits=splits,
        control_splits=cfg.get("eval_control_splits", ["train", "test"]),
        crop_universe=cu, require_all_scored=False, warn_val=False)
    if "split" not in trt.columns:
        raise SystemExit(f"{fx} has no split column")
    g = a.group
    keep = set(trt.loc[trt["split"] == ev, g].astype(str))
    trt = trt[trt[g].astype(str).isin(keep)]
    rng = np.random.default_rng(a.seed)
    parts = []
    for (sp, _), d in trt.groupby(["split", g], sort=True):
        if sp != ev and a.max_per_unit and len(d) > a.max_per_unit:
            d = d.iloc[np.sort(rng.choice(len(d), a.max_per_unit, replace=False))]
        parts.append(d)
    trt = pd.concat(parts).reset_index(drop=True)
    print(f"[real] {len(keep)} {g}(s) with {ev} crops; "
          + ", ".join(f"{s} {int((trt.split == s).sum())}" for s in splits))
    device = torch.device(a.device)
    tf = default_bbbc_transform(cfg["img_size"], cfg.get("n_channels", 3))
    F = _run(inception_net(device),
             _Crops(trt["path"], tf, int(cfg.get("n_channels", 3))),
             device, a.batch_size, a.num_workers)
    os.makedirs(a.out, exist_ok=True)
    index_frame(trt).to_parquet(os.path.join(a.out, "index.parquet"), index=False)
    np.save(os.path.join(a.out, "feat_real.npy"), F.astype(np.float32))
    print(f"-> {a.out}  feat_real.npy {F.shape}")


def cmd_pngs(a) -> None:
    """Generated PNGs, <png_root>/.../<crop_id>.png (CellFlux's layout),
    joined to an index. The file stem is the crop_id; crops the index does not
    hold are reported and left out."""
    import glob
    files = sorted(glob.glob(os.path.join(a.png_root, "**", "*.png"),
                             recursive=True))
    if not files:
        raise SystemExit(f"no PNGs under {a.png_root}")
    stem = pd.Series([os.path.splitext(os.path.basename(f))[0] for f in files])
    if stem.duplicated().any():
        raise SystemExit(f"{int(stem.duplicated().sum())} duplicate file stems "
                         f"under {a.png_root}: the stem must be the crop_id")
    ix = (pd.read_parquet(a.index) if a.index.endswith(".parquet")
          else pd.read_csv(a.index))
    key = "crop_id" if "crop_id" in ix.columns else "SAMPLE_KEY"
    ix = ix.rename(columns={key: "crop_id"})
    ix["crop_id"] = ix["crop_id"].astype(str)
    ix = ix.drop_duplicates("crop_id").set_index("crop_id")
    hit = stem.isin(ix.index).to_numpy()
    print(f"[pngs] {len(files)} PNGs, {int(hit.sum())} in the index, "
          f"{int((~hit).sum())} not (left out)")
    if not hit.any():
        raise SystemExit(f"no PNG stem matches a crop_id of {a.index}, e.g. "
                         f"{stem.iloc[0]!r} vs {ix.index[0]!r}")
    files = [f for f, h in zip(files, hit) if h]
    idx = ix.loc[stem[hit].to_numpy()].reset_index()
    device = torch.device(a.device)
    F = _run(inception_net(device), _Pngs(files), device, a.batch_size,
             a.num_workers)
    os.makedirs(a.out, exist_ok=True)
    index_frame(idx).to_parquet(os.path.join(a.out, f"index__{a.name}.parquet"),
                                index=False)
    np.save(os.path.join(a.out, f"feat_gen__{a.name}.npy"), F.astype(np.float32))
    print(f"-> {a.out}  feat_gen__{a.name}.npy {F.shape}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("real", help="real crops of the eval perturbations")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--splits", default="train",
                   help="splits besides the eval split (always included)")
    r.add_argument("--group", default="compound",
                   help="the perturbation column (FIDc's grouping)")
    r.add_argument("--max_per_unit", type=int, default=600,
                   help="cap per (split, perturbation) outside the eval "
                        "split; 0 = no cap")
    q = sp.add_parser("pngs", help="generated PNGs named <crop_id>.png")
    q.add_argument("--png_root", required=True)
    q.add_argument("--index", required=True,
                   help="parquet/csv with crop_id (or SAMPLE_KEY) + metadata")
    q.add_argument("--out", required=True)
    q.add_argument("--name", required=True)
    for s in (r, q):
        s.add_argument("--device", default="cuda" if torch.cuda.is_available()
                       else "cpu")
        s.add_argument("--batch_size", type=int, default=128)
        s.add_argument("--num_workers", type=int, default=8)
        s.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"real": cmd_real, "pngs": cmd_pngs}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
