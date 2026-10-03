"""Latent CellFlux, per crop, with the responder tilt.

One control crop -> one treated crop in the frozen VAE's latent space, with the
treated crop's loss scaled by its responder weight::

    l_i  = mean over latent dims of (v_pred - v_true)**2      # [B]
    loss = sum_i w_i l_i / sum_i w_i
    w_i  = s_i**gamma / mean_{j in same unit}(s_j**gamma)

gamma = 0 is the baseline: every weight is exactly 1.0 and the run is identical
to an unweighted one (asserted at startup). The effective sample size
``(sum w)^2 / sum w^2`` is logged per batch to ess.csv.

    python flow/train_cellflux_percrop.py --config configs/cellflux_percrop_bbbc_fp.yaml
    python flow/train_cellflux_percrop.py --config ... --gamma 0.5 --out_suffix g0.5
    python flow/train_cellflux_percrop.py --config ... --dry_run     # no GPU needed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.crop_flow_dataset import (CropFlowDataset, assign_cond_idx,    # noqa: E402
                               collate, latent_keys, load_latents,
                               load_cond_table, load_training_index,
                               resolve_scaling_factor, responder_weights)
from flow.dit import DiTConfig, DiTVelocity                            # noqa: E402
from flow.flow_matching import FlowConfig, RectifiedFlowBag            # noqa: E402
from autoencoder.utils import (make_grid_rgb, rgb_scale,                      # noqa: E402
                   to_minus_one_one, to_zero_one)
from autoencoder.vae_wrapper import load_frozen_vae                           # noqa: E402


def ess(w: torch.Tensor) -> float:
    """(sum w)^2 / sum w^2 -- the batch's effective sample size."""
    return float(w.sum() ** 2 / (w ** 2).sum())


class EMA:
    """Exponential moving average of the velocity field's weights (shadow copy)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.buffers = {k: v for k, v in model.state_dict().items()
                        if not v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1 - self.decay)

    def state_dict(self):
        return {**self.buffers, **{k: v.clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.float())


class _NullWriter:
    """Stand-in for SummaryWriter. ess.csv and previews/ carry the same content."""

    def add_scalar(self, *a, **k): pass

    def add_image(self, *a, **k): pass

    def close(self): pass


def _writer(path):
    """TensorBoard if it imports, a no-op if it does not."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(path)
    except Exception as e:                                       # noqa: BLE001
        print(f"[tb] disabled ({type(e).__name__}: {str(e)[:80]}); "
              f"metrics still go to ess.csv")
        return _NullWriter()


# --------------------------------------------------------------------------- #
def preview_batch(ds, n: int, device):
    """A fixed batch of crops, spread across units, for the image previews.

    Indices are chosen once and the epoch is pinned, so the control partner
    does not change between logs.
    """
    order = np.argsort(ds.t["unit_id"].to_numpy(), kind="stable")
    take = order[np.linspace(0, len(order) - 1, min(n, len(order))).astype(int)]
    saved, ds.epoch = ds.epoch, 0
    items = [ds[int(i)] for i in take]
    ds.epoch = saved
    ctrl = torch.stack([b[0] for b in items]).to(device).float()
    trt = torch.stack([b[1] for b in items]).to(device).float()
    text = torch.tensor([b[2] for b in items], dtype=torch.long, device=device)
    label = ds.t.iloc[take]["compound"].astype(str).tolist()
    return ctrl, trt, text, label


@torch.no_grad()
def log_images(writer, step, model, flow, vae, batch, cfg, text_table, device,
               png_dir=None):
    """Four rows: control source, real treated, VAE reconstruction, generated."""
    ctrl, trt, text, _ = batch
    in_range = cfg.get("input_range", "01")
    c = (text_table[text] if text_table is not None
         else torch.zeros(len(text), cfg["dit"]["text_dim"], device=device))
    assert c.shape == (len(text), cfg["dit"]["text_dim"]), tuple(c.shape)
    z0 = vae.encode_flat(to_minus_one_one(ctrl, in_range))
    z1 = vae.encode_flat(to_minus_one_one(trt, in_range))
    z1_hat = flow.sample(model, z0, c, num_steps=cfg.get("sample_steps", 50),
                         cfg_scale=cfg.get("cfg_scale", 1.2),
                         method=cfg.get("ode_method", "heun2"),
                         edm_schedule=cfg.get("edm_schedule", True))
    dec = lambda z: to_zero_one(vae.decode(z)).clamp(0, 1).cpu()

    rows = [ctrl.cpu(), trt.cpu(), dec(z1), dec(z1_hat)]
    ch = tuple(cfg.get("rgb_channels", [0, 1, 2]))
    pct = cfg.get("img_percentile", 99.5)
    # one scale, taken from the real crops, applied to every row -- otherwise a
    # dim generation is stretched to look as bright as its target
    scale = rgb_scale(trt.cpu(), ch, pct)
    grid = make_grid_rgb(torch.cat(rows, 0), ch, max_imgs=4 * len(ctrl),
                         scale=scale, nrow=len(ctrl))
    writer.add_image("preview/ctrl_real_recon_generated", grid, step)
    # also to disk: TensorBoard is optional
    if png_dir is not None:
        from torchvision.utils import save_image
        os.makedirs(png_dir, exist_ok=True)
        save_image(grid, os.path.join(png_dir, f"step_{step:08d}.png"))
    return grid


def make_loader(cfg: Dict[str, Any], gamma: float, allow_unscored: bool):
    from datasets.bbbc021_dataset import default_bbbc_transform

    trt, ctl, info = load_training_index(
        cfg["flow_index"], cfg["crop_scores"],
        train_splits=cfg.get("train_splits", ["train"]),
        control_splits=cfg.get("control_splits", ["train", "test"]),
        crop_universe=cfg.get("crop_universe"),
        require_all_scored=not allow_unscored)

    text_table = None
    if cfg.get("text_emb"):
        text_table, cpd_row = load_cond_table(cfg["text_emb"],
                                              int(cfg["dit"]["text_dim"]))
        if cpd_row is not None:
            trt = assign_cond_idx(trt, cpd_row)
        elif "text_idx" not in trt.columns:
            raise SystemExit("the flow index has no text_idx column; rebuild "
                             "it with flow_index.py --text_emb")

    lat = row_of = lmeta = None
    if cfg.get("latents"):
        lat, row_of, lmeta = load_latents(
            cfg["latents"], list(latent_keys(trt)) + list(latent_keys(ctl)))

    tf = default_bbbc_transform(cfg["img_size"], cfg.get("n_channels", 3))
    ds = CropFlowDataset(trt, ctl, gamma=gamma, transform=tf,
                         seed=cfg.get("train_seed", cfg.get("seed", 0)),
                         text_table=text_table,
                         latents=lat, latent_row=row_of,
                         scaling_factor=cfg["vae"].get("scaling_factor", 1.0),
                         weight_norm=cfg.get("weight_norm", "unit"))
    rep = ds.weight_report()
    print("[weights] " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float)
                                   else f"{k}={v}" for k, v in rep.items()))
    if abs(rep["unit_mean_max_dev"]) > 1e-9:
        raise SystemExit(f"per-unit mean weight is off by "
                         f"{rep['unit_mean_max_dev']:.2e}; normalisation broken")
    ds.latent_meta = lmeta
    # Drawing crop i with p ~ w_i and training on an unweighted mean targets
    # the same tilted distribution as sum(w*l)/sum(w) on a uniform draw, at
    # ESS = batch size.
    sampler = None
    if cfg.get("sampler"):
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(torch.as_tensor(ds.w,
                                                        dtype=torch.double),
                                        num_samples=len(ds), replacement=True)
        print(f"[sampler] drawing {len(ds)} crops per epoch with replacement, "
              f"p ~ w; the loss reduction is unweighted")
    loader = DataLoader(ds, batch_size=cfg["batch_size"],
                        shuffle=sampler is None, sampler=sampler,
                        num_workers=cfg.get("num_workers", 8), pin_memory=True,
                        drop_last=True, collate_fn=collate,
                        persistent_workers=cfg.get("num_workers", 8) > 0)
    return ds, loader, rep, info


def check_gamma0(ds: CropFlowDataset) -> None:
    """gamma=0 must reproduce the unweighted baseline exactly.

    Checks that every weight is 1.0 and that ``sum(w*l)/sum(w)`` is
    bit-identical to ``l.mean()`` when every w is 1.
    """
    w = torch.as_tensor(ds.w, dtype=torch.float64)
    if not torch.equal(w, torch.ones_like(w)):
        raise AssertionError(f"gamma=0 weights are not all 1.0: "
                             f"min {w.min()} max {w.max()}")
    g = torch.Generator().manual_seed(0)
    for n in (8, 64, 256):
        l = torch.rand(n, generator=g, dtype=torch.float64)
        ww = torch.ones(n, dtype=torch.float64)
        if not torch.equal((ww * l).sum() / ww.sum(), l.mean()):
            raise AssertionError(
                f"weighted reduction != unweighted mean at n={n}; the gamma=0 "
                f"run would not reproduce the baseline")
    print(f"[gamma0] weights all exactly 1.0 over {len(w)} crops, and the "
          f"weighted reduction is bit-identical to an unweighted mean")


# --------------------------------------------------------------------------- #
def train(cfg: Dict[str, Any], gamma: float, out_dir: Path, resume: str | None,
          allow_unscored: bool, dry_run: bool, force: bool = False) -> None:
    # never overwrite an arm dir that already holds a checkpoint
    ck = out_dir / "ckpt" / "last.pt"
    if ck.exists() and not resume and not dry_run and not force:
        import json as _j
        pv = out_dir / "provenance.json"
        was = _j.load(open(pv)).get("gamma") if pv.exists() else "?"
        raise SystemExit(
            f"{ck} already exists (gamma={was}). Refusing to overwrite an arm.\n"
            f"  --resume {ck}   to continue it\n"
            f"  --force         to start it over\n"
            f"  --out_suffix X  to write somewhere else\n"
            f"If you did not mean to land here, check that every argument "
            f"reached the process -- a wrapped paste silently drops them and "
            f"the run falls back to the config's own gamma and index.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpt").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("train_seed", cfg.get("seed", 0)))

    ds, loader, wrep, info = make_loader(cfg, gamma, allow_unscored)
    sampled = bool(cfg.get("sampler"))
    if gamma == 0.0:
        check_gamma0(ds)

    prov = {"gamma": gamma, "sampler": sampled, "weights": wrep, "index": info,
            "flow_index": cfg["flow_index"], "crop_scores": cfg["crop_scores"],
            # which conditioning this arm was trained on
            "cond_emb": cfg.get("text_emb"),
            "cond_dim": cfg["dit"]["text_dim"],
            "p_cond_drop_per_batch": cfg["flow"].get("p_cond_drop_per_batch",
                                                     False),
            "latents": cfg.get("latents"), "latent_meta": ds.latent_meta,
            "scaling_factor": cfg["vae"].get("scaling_factor", 1.0),
            "train_seed": cfg.get("train_seed", cfg.get("seed", 0))}
    try:
        import pyarrow.parquet as pq
        md = pq.read_schema(cfg["crop_scores"]).metadata or {}
        prov["scores_stamp"] = {k.decode(): v.decode() for k, v in md.items()
                                if not k.decode().startswith("pandas")}
    except Exception as e:                                       # noqa: BLE001
        prov["scores_stamp"] = f"unreadable: {e}"
    json.dump(prov, open(out_dir / "provenance.json", "w"), indent=2, default=str)
    print(f"[prov] scores {prov['scores_stamp']}")

    if dry_run:
        print("\n[dry_run] index, join, weights and the gamma=0 identity are "
              "checked; no model built, no GPU touched.")
        return

    writer = _writer(out_dir / "tb")

    img_every = int(cfg.get("img_every", 0))
    vae = None
    if ds.latents is not None and img_every:
        # cached latents remove the encoder from the training loop, but the
        # decoder is still needed to look at anything
        vae = load_frozen_vae(cfg["vae"]["ckpt"], cfg["vae"]["model"],
                              use_ema=cfg["vae"].get("use_ema", True),
                              scaling_factor=cfg["vae"].get("scaling_factor", 1.0),
                              device=device)
        C_l, H_l, W_l = vae.latent_shape
        print(f"[vae] loaded for image previews only; training still reads the "
              f"latent cache ({C_l}, {H_l}, {W_l})")
    elif ds.latents is not None:
        C_l, H_l, W_l = ds.latent_meta["latent_shape"]
        print(f"[vae] not loaded: training off the latent cache "
              f"({C_l}, {H_l}, {W_l}), scaling_factor "
              f"{cfg['vae'].get('scaling_factor', 1.0)}")
    else:
        vae = load_frozen_vae(ckpt_path=cfg["vae"]["ckpt"],
                              vae_cfg_dict=cfg["vae"]["model"],
                              use_ema=cfg["vae"].get("use_ema", True),
                              scaling_factor=cfg["vae"].get("scaling_factor", 1.0),
                              device=device)
        C_l, H_l, W_l = vae.latent_shape
        print(f"[vae] encoding on the fly, latent ({C_l}, {H_l}, {W_l}) -- "
              f"precompute_latents.py removes this from the training loop")

    model = DiTVelocity(DiTConfig(latent_channels=C_l, latent_size=H_l,
                                  **cfg["dit"])).to(device)
    flow = RectifiedFlowBag(FlowConfig(**cfg["flow"]))
    print(f"[dit] {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params")

    # CellFlux's default is betas (0.9, 0.95), not torch's (0.9, 0.999)
    betas = tuple(cfg.get("optimizer_betas", [0.9, 0.95]))
    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], betas=betas,
                      weight_decay=cfg.get("weight_decay", 0.0))
    ema = EMA(model, cfg.get("ema_decay", 0.9999)) if cfg.get("use_ema", True) else None
    print(f"[opt] AdamW lr={cfg['lr']} betas={betas}  "
          f"ema={'decay ' + str(cfg.get('ema_decay', 0.9999)) if ema else 'off'}")
    text_table = ds.text_table.to(device) if ds.text_table is not None else None

    step, start_epoch = 0, 0
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if ema is not None and ck.get("ema"):
            ema.load_state_dict(ck["ema"])
        step, start_epoch = ck["global_step"], ck["epoch"] + 1
        if abs(ck.get("gamma", gamma) - gamma) > 1e-12:
            raise SystemExit(f"checkpoint was trained at gamma "
                             f"{ck.get('gamma')}, not {gamma}")
        print(f"[resume] step {step}, epoch {start_epoch}")

    prev = None
    if img_every:
        # previews come from a dataset that always returns IMAGES, so the real
        # crops can be shown next to the generated ones
        from datasets.bbbc021_dataset import default_bbbc_transform
        pv = CropFlowDataset(
            ds.t, ds.c, gamma=0.0,
            transform=default_bbbc_transform(cfg["img_size"],
                                             cfg.get("n_channels", 3)),
            seed=cfg.get("seed", 0))
        prev = preview_batch(pv, int(cfg.get("n_preview", 8)), device)
        print(f"[preview] {len(prev[0])} crop(s) every {img_every} steps: "
              + ", ".join(prev[3]))

    in_range = cfg.get("input_range", "01")
    s1 = cfg["stage1_epochs"]
    total = s1 + cfg["stage2_epochs"]
    log_every, ckpt_every = cfg.get("log_every", 50), cfg.get("ckpt_every", 2000)
    # 0 = only last.pt
    keep_every = int(cfg.get("keep_every", 0) or 0)
    # in epochs, when steps/epoch is only known once the index is loaded
    if cfg.get("keep_every_epochs"):
        keep_every = int(cfg["keep_every_epochs"]) * len(loader)
        print(f"[ckpt] snapshot every {cfg['keep_every_epochs']} epochs = "
              f"{keep_every} steps ({len(loader)} steps/epoch)")
    ess_log = []

    for epoch in range(start_epoch, total):
        stage = 1 if epoch < s1 else 2
        ds.set_epoch(epoch)
        model.train()
        t0, run_loss, run_ess, nb = time.time(), 0.0, 0.0, 0

        for ctrl, trt, text, w in loader:
            ctrl, trt = ctrl.to(device).float(), trt.to(device).float()
            if ds.latents is None:   # cached latents were encoded from [-1,1]
                ctrl = to_minus_one_one(ctrl, in_range)
                trt = to_minus_one_one(trt, in_range)
            w = w.to(device)
            c = (text_table[text.to(device)] if text_table is not None
                 else torch.zeros(len(w), cfg["dit"]["text_dim"], device=device))
            if c.dim() != 2 or c.shape != (len(w), cfg["dit"]["text_dim"]):
                # the DiT does not reject a 3-D c; it broadcasts it into
                # [B, B, hidden] and fails several frames later in adaLN
                raise SystemExit(f"conditioning must be [B, text_dim] = "
                                 f"{(len(w), cfg['dit']['text_dim'])}, got "
                                 f"{tuple(c.shape)}")

            if ds.latents is not None:
                z0, z1 = ctrl, trt              # already latents, already scaled
            else:
                z0, z1 = vae.encode_flat(ctrl), vae.encode_flat(trt)
            li, log = flow.training_loss_per_sample(model, z0, z1, c, stage)
            # the sampler already applied the tilt; weighting here too would
            # apply it twice
            if sampled:
                w = torch.ones_like(w)
            # sum(w*l)/sum(w), never mean(w*l): the latter changes the
            # effective learning rate with gamma.
            loss = (w * li).sum() / w.sum()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            if ema is not None:
                ema.update(model)

            e = ess(w)
            run_loss += float(loss); run_ess += e; nb += 1
            if step % log_every == 0:
                writer.add_scalar("train/loss", float(loss), step)
                writer.add_scalar("train/ess", e, step)
                writer.add_scalar("train/ess_frac", e / len(w), step)
                writer.add_scalar("train/w_max", float(w.max()), step)
                writer.add_scalar("train/stage", stage, step)
                writer.add_scalar("train/disp_norm",
                                  float(log["flow/disp_norm"]), step)
                ess_log.append({"step": step, "epoch": epoch, "gamma": gamma,
                                "loss": float(loss), "ess": e,
                                "ess_frac": e / len(w), "batch": int(len(w)),
                                "w_max": float(w.max())})
                print(f"ep {epoch} stage {stage} step {step} "
                      f"loss {float(loss):.4f} ess {e:.1f}/{len(w)} "
                      f"({100 * e / len(w):.0f}%) w_max {float(w.max()):.2f}")
            if prev is not None and step % img_every == 0:
                model.eval()
                log_images(writer, step, model, flow, vae, prev, cfg,
                           text_table, device, png_dir=str(out_dir / "previews"))
                model.train()
            if step and step % ckpt_every == 0:
                save(out_dir / "ckpt" / "last.pt", model, opt, step, epoch,
                     gamma, ema)
            # stage 2 only: a stage-1 checkpoint maps Gaussian -> target and
            # cannot be evaluated on the control -> target task
            if (keep_every and stage == 2 and step
                    and step % keep_every == 0):
                snapshot(out_dir / "ckpt" / f"step_{step:08d}.pt", model,
                         step, epoch, gamma, ema)
            step += 1

        print(f"[epoch {epoch} stage {stage}] loss {run_loss / max(nb, 1):.4f} "
              f"mean ess {run_ess / max(nb, 1):.1f}  {time.time() - t0:.0f}s")
        save(out_dir / "ckpt" / "last.pt", model, opt, step, epoch, gamma, ema)
        pd.DataFrame(ess_log).to_csv(out_dir / "ess.csv", index=False)
    writer.close()
    pd.DataFrame(ess_log).to_csv(out_dir / "ess.csv", index=False)

    if cfg.get("eval_after_train", False):
        del model, opt, ema, vae
        torch.cuda.empty_cache()
        eval_after_train(cfg, out_dir)


def eval_after_train(cfg: Dict[str, Any], out_dir: Path) -> None:
    """Run the FID/KID job on this arm's final checkpoint, in-process.

    A failure here is caught: the checkpoint is already saved. Each arm writes
    only its own ``eval_metrics.json``; ``eval_flow.py --collect`` gathers the
    per-arm files into one table.
    """
    from types import SimpleNamespace

    from evaluation import eval_flow

    e = cfg.get("eval", {})
    moa_head = e.get("moa_head")
    if moa_head and not Path(moa_head).exists():
        print(f"[eval] no MoA head at {moa_head} -- fit it with moa_eval.py; "
              f"evaluating without MoA")
        moa_head = None
    pheno_dir = e.get("pheno_dir")
    if pheno_dir and not (Path(pheno_dir) / "heads.pt").exists():
        print(f"[eval] no pheno heads in {pheno_dir} -- run pheno_eval.py fit; "
              f"evaluating without top-k / replicate mAP")
        pheno_dir = None
    a = SimpleNamespace(
        limit=e.get("limit"), resp_q=e.get("resp_q", 0.5),
        steps=e.get("steps", cfg.get("sample_steps", 50)),
        cfg_scale=e.get("cfg_scale", cfg.get("cfg_scale", 1.2)),
        ode_method=e.get("ode_method", cfg.get("ode_method", "heun2")),
        no_edm_schedule=not cfg.get("edm_schedule", True),
        batch_size=e.get("batch_size", 64),
        num_workers=e.get("num_workers", cfg.get("num_workers", 8)),
        kid_subset=e.get("kid_subset", 100), weights=e.get("weights", "ema"),
        allow_unscored=False, no_inception=e.get("no_inception", False),
        prdc_k=e.get("prdc_k", 5), dump_images=None, moa_head=moa_head,
        pheno_dir=pheno_dir, pheno_null=e.get("pheno_null", 100_000))
    print(f"\n[eval] {out_dir.name}: FID/KID on the {cfg.get('eval_split', 'val')} "
          f"split, {a.steps} steps, {a.weights} weights")
    try:
        row = eval_flow.compute(cfg, str(out_dir / "ckpt" / "last.pt"), out_dir, a)
    except (Exception, SystemExit) as err:                        # noqa: BLE001
        print(f"[eval] FAILED: {type(err).__name__}: {str(err)[:400]}\n"
              f"[eval] the checkpoint is saved; re-run by hand with\n"
              f"       python evaluation/eval_flow.py --config <cfg> --arms {out_dir.name}")
        return
    json.dump(row, open(out_dir / "eval_metrics.json", "w"), indent=2)
    print("[eval] " + "  ".join(
        f"{k}={row[k]:.3f}" for k in ("fid_all", "fid_resp", "fid_recon",
                                      "energy_latent_all", "energy_latent_resp")
        if k in row and row[k] == row[k]))
    print(f"[eval] -> {out_dir / 'eval_metrics.json'}")


def save(path, model, opt, step, epoch, gamma, ema=None):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "global_step": step, "epoch": epoch, "gamma": gamma}, path)


def snapshot(path, model, step, epoch, gamma, ema=None):
    """A numbered, evaluation-only checkpoint: model + EMA, no optimiser state.

    Written every `keep_every` steps so checkpoints can be selected among;
    `last.pt` alone is overwritten every `ckpt_every`.
    """
    torch.save({"model": model.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "global_step": step, "epoch": epoch, "gamma": gamma,
                "eval_only": True}, path)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--gamma", type=float, default=None,
                    help="overrides the config; 0 is the baseline")
    ap.add_argument("--out_suffix", default=None,
                    help="appended to out_dir; defaults to gamma<value>")
    ap.add_argument("--seed", type=int, default=None,
                    help="training seed (init, data order, training control "
                         "pairs); the arm becomes gamma<value>_s<seed>. Eval "
                         "keeps the config's seed, so every seed is scored on "
                         "the same control pairs.")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--allow_unscored", action="store_true",
                    help="drop treated crops with no cached score instead of "
                         "stopping. Changes the training set, so a run using "
                         "it is not comparable to one that does not.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an arm dir that already has a checkpoint")
    # dilution runs point the same config at a built set
    ap.add_argument("--flow_index", default=None)
    ap.add_argument("--crop_scores", default=None)
    ap.add_argument("--sampler", action="store_true",
                    help="draw crops with replacement at p ~ w and train on an "
                         "unweighted mean, instead of reweighting the loss on a "
                         "uniform draw. Same tilted target (Proposition 5) at "
                         "ESS = batch size, so it separates the tilt from the "
                         "effective-batch cost. Pair it with --out_suffix.")
    ap.add_argument("--crop_universe", default=None)
    ap.add_argument("--weight_norm", default=None, choices=("unit", "unit_plate"),
                    help="overrides the config's cell for the mean-1 weight "
                         "normalisation. The well_mean arm needs `unit`: under "
                         "unit_plate a cell holding one treated well turns every "
                         "well-mean weight into 1, i.e. gamma=0. Recorded in "
                         "provenance.json under weights.weight_norm.")
    # a dilution set is self-contained and all one split, so its control pool
    # is not the config's [train, test]
    ap.add_argument("--train_splits", default=None, help="comma-separated")
    ap.add_argument("--control_splits", default=None, help="comma-separated")
    ap.add_argument("--eval_units", default=None,
                    help="csv with a unit_id column; restricts the FID/KID "
                         "reference to those units. A model conditioned on 31 "
                         "compounds scored against 65 is penalised for "
                         "compounds it never saw.")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    # The FID/KID reference is always the real undiluted population: stash the
    # config's own paths before overriding.
    for k in ("flow_index", "crop_scores", "crop_universe"):
        cfg[f"eval_{k}"] = cfg.get(k)
    for k in ("flow_index", "crop_scores", "crop_universe"):
        if getattr(a, k):
            cfg[k] = getattr(a, k)
            print(f"[override] {k} = {cfg[k]}  (eval still reads "
                  f"{cfg[f'eval_{k}']})")
    for k in ("train_splits", "control_splits"):
        if getattr(a, k):
            cfg[k] = [x.strip() for x in getattr(a, k).split(",")]
            print(f"[override] {k} = {cfg[k]}")
    if a.eval_units:
        cfg["eval_units"] = a.eval_units
        print(f"[override] eval_units = {a.eval_units}")
    if a.weight_norm:
        cfg["weight_norm"] = a.weight_norm
        print(f"[override] weight_norm = {a.weight_norm}")
    if a.sampler:
        cfg["sampler"] = True
        print("[override] sampler = True (weighted draw, unweighted loss)")
    resolve_scaling_factor(cfg)
    gamma = a.gamma if a.gamma is not None else float(cfg.get("gamma", 0.0))
    if a.seed is not None:
        cfg["train_seed"] = a.seed
    suffix = a.out_suffix or (f"gamma{gamma:g}"
                              + (f"_s{a.seed}" if a.seed is not None else ""))
    out = Path(cfg["out_dir"]) / suffix
    print(f"[run] gamma={gamma}  -> {out}")
    train(cfg, gamma, out, a.resume, a.allow_unscored, a.dry_run, a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
