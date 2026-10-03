"""Evaluate a trained VAE: reconstruction L1/L2/LPIPS, reconstruction FID,
and save sample reconstruction grids.

Usage:
    python autoencoder/eval.py --config configs/vae_bbbc.yaml --ckpt runs/bbbc/ckpt/last.pt \
        --use_ema --out_dir runs/bbbc/eval
"""
from __future__ import annotations

import argparse
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoencoder.vae import AutoencoderKL, VAEConfig
from autoencoder.losses import MultiChannelLPIPS
from autoencoder.utils import to_minus_one_one, to_zero_one, make_grid_multi_channel


# --------------------------------------------------------------------------- #
def build_datasets(cfg: Dict[str, Any]):
    ds_cfg = cfg["data"]
    ns = Namespace(**ds_cfg["args"])
    if ds_cfg["name"] == "bbbc":
        from datasets.bbbc021_dataset import build_bbbc
        train_set, val_set, test_set = build_bbbc(ns, _all=True)
    elif ds_cfg["name"] == "cpg":
        from datasets.cpg_dataset import build_cpg
        train_set, val_set, test_set = build_cpg(ns, _all=True)
    elif ds_cfg["name"] == "rxrx1":
        from datasets.rxrx1_dataset import build_rxrx1
        train_set, val_set, test_set = build_rxrx1(ns, _all=True)
    else:
        raise ValueError(ds_cfg["name"])
    return train_set, val_set, test_set


def collate_image_only(batch):
    return torch.stack([b[0] for b in batch], dim=0)


# --------------------------------------------------------------------------- #
# Reconstruction FID
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_recon_fid(vae, loader, device, in_range, max_samples=10_000):
    """Reconstruction FID: compare InceptionV3 features of real vs reconstructed."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as e:
        raise ImportError("pip install torchmetrics") from e

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    seen = 0
    for imgs in loader:
        imgs = imgs.to(device, non_blocking=True).float()
        x = to_minus_one_one(imgs, in_range=in_range)
        x_rec, _ = vae(x, sample_posterior=False)

        real_01 = to_zero_one(x)
        rec_01  = to_zero_one(x_rec)

        # InceptionV3 expects 3 channels: for C != 3 use the first three
        if real_01.shape[1] != 3:
            real_rgb = real_01[:, :3]
            rec_rgb  = rec_01[:, :3]
        else:
            real_rgb, rec_rgb = real_01, rec_01

        fid.update(real_rgb, real=True)
        fid.update(rec_rgb, real=False)

        seen += imgs.size(0)
        if seen >= max_samples:
            break

    return float(fid.compute())


# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(cfg: Dict[str, Any], ckpt_path: str, use_ema: bool, out_dir: str,
             split: str = "test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")

    # ---- data ---- #
    train_set, val_set, test_set = build_datasets(cfg)
    ds = {"train": train_set, "val": val_set, "test": test_set}[split]
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_image_only,
    )

    # ---- model ---- #
    vae_cfg = VAEConfig(**cfg["model"])
    vae = AutoencoderKL(vae_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["ema"] if use_ema else ckpt["vae"]
    vae.load_state_dict(state)
    vae.eval()
    print(f"Loaded {'EMA' if use_ema else 'online'} weights from {ckpt_path}")

    in_range = cfg.get("input_range", "01")

    # ---- metrics ---- #
    lpips = MultiChannelLPIPS(net="vgg").to(device)

    tot_l1, tot_l2, tot_lpips, n = 0.0, 0.0, 0.0, 0
    sample_real, sample_rec = None, None

    for imgs in loader:
        imgs = imgs.to(device, non_blocking=True).float()
        x = to_minus_one_one(imgs, in_range=in_range)
        x_rec, _ = vae(x, sample_posterior=False)

        l1 = (x - x_rec).abs().mean()
        l2 = ((x - x_rec) ** 2).mean()
        p  = lpips(x, x_rec)

        bs = x.size(0)
        tot_l1    += float(l1) * bs
        tot_l2    += float(l2) * bs
        tot_lpips += float(p)  * bs
        n += bs

        if sample_real is None:
            sample_real = x[:16].cpu()
            sample_rec  = x_rec[:16].cpu()

    print(f"[{split}] L1 {tot_l1/n:.4f} | L2 {tot_l2/n:.4f} | LPIPS {tot_lpips/n:.4f}")

    # FID over reasonable budget
    fid_score = compute_recon_fid(vae, loader, device, in_range,
                                  max_samples=cfg.get("fid_max_samples", 10_000))
    print(f"[{split}] reconstruction FID: {fid_score:.3f}")

    # save samples and metrics
    writer.add_image("eval/real", make_grid_multi_channel(to_zero_one(sample_real), max_imgs=16), 0)
    writer.add_image("eval/rec",  make_grid_multi_channel(to_zero_one(sample_rec),  max_imgs=16), 0)
    writer.add_scalar(f"eval/{split}_l1",     tot_l1 / n, 0)
    writer.add_scalar(f"eval/{split}_l2",     tot_l2 / n, 0)
    writer.add_scalar(f"eval/{split}_lpips",  tot_lpips / n, 0)
    writer.add_scalar(f"eval/{split}_fid",    fid_score, 0)
    writer.close()

    metrics = {
        "split": split,
        "l1":    tot_l1 / n,
        "l2":    tot_l2 / n,
        "lpips": tot_lpips / n,
        "fid":   fid_score,
        "n":     n,
        "ckpt":  ckpt_path,
        "ema":   use_ema,
    }
    with open(out_dir / f"metrics_{split}.yaml", "w") as f:
        yaml.dump(metrics, f)
    print(f"Saved metrics to {out_dir / f'metrics_{split}.yaml'}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg, args.ckpt, args.use_ema, args.out_dir, split=args.split)