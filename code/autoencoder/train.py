"""Train the AutoencoderKL on BBBC021 (3ch), CPG (5ch) or RxRx1 (6ch).

Usage:
    python autoencoder/train.py --config configs/vae_bbbc.yaml
    python autoencoder/train.py --config configs/vae_cpg.yaml

Resume:
    python autoencoder/train.py --config configs/vae_bbbc.yaml --resume runs/bbbc/ckpt/last.pt
    python autoencoder/train.py --config configs/vae_bbbc.yaml --resume auto   # <out_dir>/ckpt/last.pt

Logs go to TensorBoard at <out_dir>/tb, checkpoints to <out_dir>/ckpt. A
checkpoint carries the config, both optimiser and LR-scheduler states, the EMA
weights and the torch RNG state.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoencoder.vae import AutoencoderKL, VAEConfig
from autoencoder.losses import LPIPSWithDiscriminator
from autoencoder.utils import (ModelEMA, to_minus_one_one, to_zero_one, make_grid_multi_channel,
                   count_parameters, load_ckpt)


# --------------------------------------------------------------------------- #
# Dataset wiring
# --------------------------------------------------------------------------- #
def build_datasets(cfg: Dict[str, Any]):
    """Return (train, val) from build_bbbc / build_cpg / build_rxrx1.

    Uses the val fold, not test: the BBBC021 test fold is the OOD split.
    """
    ds_cfg = cfg["data"]
    ns = Namespace(**ds_cfg["args"])

    if ds_cfg["name"] == "bbbc":
        from datasets.bbbc021_dataset import build_bbbc
        train_set, val_set, _ = build_bbbc(ns, _all=True)
    elif ds_cfg["name"] == "cpg":
        from datasets.cpg_dataset import build_cpg
        train_set, val_set, _ = build_cpg(ns, _all=True)
    elif ds_cfg["name"] == "rxrx1":
        from datasets.rxrx1_dataset import build_rxrx1
        train_set, val_set, _ = build_rxrx1(ns, _all=True)
    else:
        raise ValueError(ds_cfg["name"])
    return train_set, val_set


def collate_image_only(batch):
    """Datasets return (img, labels_dict); we only need img for VAE training."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    return imgs


# --------------------------------------------------------------------------- #
# LR schedule
# --------------------------------------------------------------------------- #
def make_lr_lambda(name: str, warmup_steps: int, total_steps: int,
                   min_lr_scale: float):
    """Per-global-step LR multiplier. `constant` with no warmup is a no-op."""
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if name == "constant":
            return 1.0
        if name == "cosine":
            p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            p = min(max(p, 0.0), 1.0)
            return min_lr_scale + (1.0 - min_lr_scale) * 0.5 * (1.0 + math.cos(math.pi * p))
        raise ValueError(f"unknown scheduler.name: {name!r}")
    return fn


def build_schedulers(cfg: Dict[str, Any], opt_g, opt_d, steps_per_epoch: int):
    """Schedulers for both optimisers. Absent `scheduler:` block -> constant LR."""
    sch = cfg.get("scheduler") or {}
    name = sch.get("name", "constant")
    warmup = int(sch.get("warmup_steps", 0))
    min_scale = float(sch.get("min_lr_scale", 0.1))
    total = steps_per_epoch * int(cfg["epochs"])
    fn = make_lr_lambda(name, warmup, total, min_scale)
    print(f"LR schedule: {name} (warmup {warmup}, total {total} steps)")
    return (optim.lr_scheduler.LambdaLR(opt_g, fn),
            optim.lr_scheduler.LambdaLR(opt_d, fn))


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(cfg: Dict[str, Any], resume: str | None = None,
          allow_cfg_mismatch: bool = False):
    out_dir = Path(cfg["out_dir"])
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (out_dir / "tb").mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 0))

    # ---------------- data ---------------- #
    train_set, val_set = build_datasets(cfg)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_image_only,
        persistent_workers=cfg.get("num_workers", 8) > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_image_only,
    )

    # ---------------- model ---------------- #
    vae_cfg = VAEConfig(**cfg["model"])
    vae = AutoencoderKL(vae_cfg).to(device)
    print(f"VAE params: {count_parameters(vae) / 1e6:.2f} M")

    loss_fn = LPIPSWithDiscriminator(
        disc_in_channels=vae_cfg.in_channels,
        **cfg["loss"],
    ).to(device)
    print(f"Disc params: {count_parameters(loss_fn.discriminator) / 1e6:.2f} M")

    ema = ModelEMA(vae, decay=cfg.get("ema_decay", 0.999))

    # ---------------- optimisers ---------------- #
    lr = cfg["lr"]
    opt_g = optim.AdamW(
        list(vae.encoder.parameters())
        + list(vae.decoder.parameters())
        + list(vae.quant_conv.parameters())
        + list(vae.post_quant_conv.parameters()),
        lr=lr, betas=(0.5, 0.9),
    )
    opt_d = optim.AdamW(
        loss_fn.discriminator.parameters(),
        lr=lr, betas=(0.5, 0.9),
    )

    sched_g, sched_d = build_schedulers(cfg, opt_g, opt_d, len(train_loader))

    global_step = 0
    start_epoch = 0
    if resume is not None:
        if resume == "auto":
            resume = str(out_dir / "ckpt" / "last.pt")
        if not os.path.exists(resume):
            raise FileNotFoundError(f"--resume {resume} does not exist")

        ckpt = load_ckpt(resume)
        check_resume_cfg(ckpt, cfg, allow_cfg_mismatch)

        vae.load_state_dict(ckpt["vae"])
        loss_fn.discriminator.load_state_dict(ckpt["disc"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        ema.ema.load_state_dict(ckpt["ema"])

        # checkpoints without scheduler state restart the schedule
        if "sched_g" in ckpt:
            sched_g.load_state_dict(ckpt["sched_g"])
            sched_d.load_state_dict(ckpt["sched_d"])
        else:
            print("[resume] no scheduler state in checkpoint - schedulers start at step 0")

        if ckpt.get("rng", {}).get("torch") is not None:
            torch.set_rng_state(ckpt["rng"]["torch"])
            if torch.cuda.is_available() and ckpt["rng"].get("cuda") is not None:
                try:
                    torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
                except (RuntimeError, ValueError) as e:
                    print(f"[resume] could not restore CUDA RNG state ({e}); continuing")

        global_step = ckpt["global_step"]
        # `epoch_done` distinguishes an end-of-epoch save from a mid-epoch
        # `ckpt_every` save; a missing key means the epoch finished.
        start_epoch = ckpt["epoch"] + int(ckpt.get("epoch_done", True))
        print(f"Resumed from {resume}: step {global_step}, starting at epoch {start_epoch} "
              f"(lr {opt_g.param_groups[0]['lr']:.3e})")

    in_range = cfg.get("input_range", "01")  # what the dataset hands us
    log_every = cfg.get("log_every", 50)
    img_every = cfg.get("img_every", 500)
    val_every = cfg.get("val_every", 2000)
    ckpt_every = cfg.get("ckpt_every", 5000)

    # ---------------- loop ---------------- #
    for epoch in range(start_epoch, cfg["epochs"]):
        vae.train()
        loss_fn.train()
        ep_start = time.time()

        for it, imgs in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True).float()
            x = to_minus_one_one(imgs, in_range=in_range)

            # ----- generator step ----- #
            opt_g.zero_grad(set_to_none=True)
            x_rec, posterior = vae(x, sample_posterior=True)
            g_loss, g_log = loss_fn(
                inputs=x,
                reconstructions=x_rec,
                posterior=posterior,
                optimizer_idx=0,
                global_step=global_step,
                last_layer=vae.last_layer,
            )
            g_loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            opt_g.step()

            # ----- discriminator step ----- #
            if global_step >= cfg["loss"]["disc_start"]:
                opt_d.zero_grad(set_to_none=True)
                # need fresh forward for fair disc update
                with torch.no_grad():
                    x_rec_det, posterior_det = vae(x, sample_posterior=True)
                d_loss, d_log = loss_fn(
                    inputs=x,
                    reconstructions=x_rec_det,
                    posterior=posterior_det,
                    optimizer_idx=1,
                    global_step=global_step,
                    last_layer=None,
                )
                d_loss.backward()
                opt_d.step()
            else:
                d_log = {}

            ema.update(vae)

            # ----- logging ----- #
            if global_step % log_every == 0:
                writer.add_scalar("train/lr", opt_g.param_groups[0]["lr"], global_step)
                for k, v in g_log.items():
                    writer.add_scalar(f"train/{k}", float(v), global_step)
                for k, v in d_log.items():
                    writer.add_scalar(f"train/{k}", float(v), global_step)
                print(
                    f"ep {epoch} it {it} step {global_step} "
                    f"nll {float(g_log['loss/nll']):.4f} "
                    f"kl {float(g_log['loss/kl']):.2f} "
                    f"g {float(g_log['loss/g']):.4f}"
                )

            if global_step % img_every == 0:
                with torch.no_grad():
                    vae.eval()
                    x_rec_eval, _ = vae(x[:8], sample_posterior=False)
                    vae.train()
                    real_grid = make_grid_multi_channel(to_zero_one(x[:8]))
                    rec_grid = make_grid_multi_channel(to_zero_one(x_rec_eval))
                    writer.add_image("train/real", real_grid, global_step)
                    writer.add_image("train/rec",  rec_grid,  global_step)

            if global_step > 0 and global_step % val_every == 0:
                validate(vae, ema, val_loader, device, in_range, writer, global_step, loss_fn)

            # Step the schedulers before checkpointing so the saved LR state
            # matches the step the resume will start on (global_step + 1).
            sched_g.step()
            sched_d.step()

            if global_step > 0 and global_step % ckpt_every == 0:
                save_ckpt(out_dir / "ckpt" / "last.pt", cfg, vae, loss_fn,
                          opt_g, opt_d, sched_g, sched_d, ema,
                          global_step + 1, epoch, epoch_done=False)

            global_step += 1

        print(f"[epoch {epoch} done in {time.time() - ep_start:.0f}s]")
        save_ckpt(out_dir / "ckpt" / f"epoch_{epoch:04d}.pt", cfg, vae, loss_fn,
                  opt_g, opt_d, sched_g, sched_d, ema,
                  global_step, epoch, epoch_done=True)
        save_ckpt(out_dir / "ckpt" / "last.pt", cfg, vae, loss_fn,
                  opt_g, opt_d, sched_g, sched_d, ema,
                  global_step, epoch, epoch_done=True)

    writer.close()


# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(vae, ema, val_loader, device, in_range, writer, global_step, loss_fn):
    vae.eval()
    ema.ema.eval()

    tot_l1, tot_lpips, tot_kl, n = 0.0, 0.0, 0.0, 0
    tot_l1_ema = 0.0
    sample_real, sample_rec, sample_rec_ema = None, None, None

    for imgs in val_loader:
        imgs = imgs.to(device, non_blocking=True).float()
        x = to_minus_one_one(imgs, in_range=in_range)

        x_rec, posterior = vae(x, sample_posterior=False)
        l1 = (x - x_rec).abs().mean()
        if loss_fn.perceptual_loss is not None:
            p = loss_fn.perceptual_loss(x, x_rec)
        else:
            p = torch.tensor(0.0, device=device)
        kl = posterior.kl()

        # EMA forward
        x_rec_ema, _ = ema.ema(x, sample_posterior=False)
        l1_ema = (x - x_rec_ema).abs().mean()

        bs = x.size(0)
        tot_l1     += float(l1) * bs
        tot_lpips  += float(p) * bs
        tot_kl     += float(kl) * bs
        tot_l1_ema += float(l1_ema) * bs
        n += bs

        if sample_real is None:
            sample_real    = x[:8].cpu()
            sample_rec     = x_rec[:8].cpu()
            sample_rec_ema = x_rec_ema[:8].cpu()

    writer.add_scalar("val/l1",      tot_l1     / n, global_step)
    writer.add_scalar("val/lpips",   tot_lpips  / n, global_step)
    writer.add_scalar("val/kl",      tot_kl     / n, global_step)
    writer.add_scalar("val/l1_ema",  tot_l1_ema / n, global_step)
    print(f"[val @ {global_step}] l1 {tot_l1/n:.4f} | l1_ema {tot_l1_ema/n:.4f} | lpips {tot_lpips/n:.4f}")

    writer.add_image("val/real",    make_grid_multi_channel(to_zero_one(sample_real)),    global_step)
    writer.add_image("val/rec",     make_grid_multi_channel(to_zero_one(sample_rec)),     global_step)
    writer.add_image("val/rec_ema", make_grid_multi_channel(to_zero_one(sample_rec_ema)), global_step)
    vae.train()


CKPT_FORMAT_VERSION = 2


def save_ckpt(path, cfg, vae, loss_fn, opt_g, opt_d, sched_g, sched_d, ema,
              global_step, epoch, epoch_done):
    """Write the full training state.

    Only tensors, plain containers and scalars, so the file loads under
    `torch.load(..., weights_only=True)`.
    """
    torch.save(
        {
            "format_version": CKPT_FORMAT_VERSION,
            "cfg":          cfg,          # lr, model, loss, scheduler, data
            "vae":          vae.state_dict(),
            "disc":         loss_fn.discriminator.state_dict(),
            "opt_g":        opt_g.state_dict(),
            "opt_d":        opt_d.state_dict(),
            "sched_g":      sched_g.state_dict(),
            "sched_d":      sched_d.state_dict(),
            "ema":          ema.ema.state_dict(),
            "ema_decay":    ema.decay,
            "global_step":  global_step,
            "epoch":        epoch,
            "epoch_done":   epoch_done,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda":  torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        path,
    )


def check_resume_cfg(ckpt, cfg, allow_mismatch: bool):
    """A checkpoint only resumes into an identically shaped model/loss."""
    old = ckpt.get("cfg")
    if old is None:
        print("[resume] checkpoint predates config saving - taking model/loss "
              "hyper-parameters from --config, unverified")
        return
    diffs = [
        f"  {block}.{k}: ckpt={old.get(block, {}).get(k)!r} "
        f"config={cfg.get(block, {}).get(k)!r}"
        for block in ("model", "loss")
        for k in sorted(set(old.get(block, {})) | set(cfg.get(block, {})))
        if old.get(block, {}).get(k) != cfg.get(block, {}).get(k)
    ]
    if not diffs:
        return
    msg = "config differs from the checkpoint:\n" + "\n".join(diffs)
    if allow_mismatch:
        print(f"[resume] WARNING: {msg}")
    else:
        raise ValueError(msg + "\n(pass --allow_cfg_mismatch to override)")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None,
                    help='checkpoint path, or "auto" for <out_dir>/ckpt/last.pt')
    ap.add_argument("--allow_cfg_mismatch", action="store_true",
                    help="resume even if model/loss hyper-parameters changed")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, resume=args.resume, allow_cfg_mismatch=args.allow_cfg_mismatch)