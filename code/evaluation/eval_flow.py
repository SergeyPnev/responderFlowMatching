"""FID / KID for the per-crop latent flow arms, on the eval split.

Generates a treated crop for every crop in ``eval_split`` by integrating the
flow from a same-plate control, and scores the generated distribution against
the real one. Writes a ``gamma``-keyed csv.

    python evaluation/eval_flow.py --config configs/cellflux_percrop_bbbc_fp.yaml     # every arm
    python evaluation/eval_flow.py --config ... --arms gamma0,gamma1 --limit 1000  # smoke

References, fixed across arms: ``all`` (every real treated crop), ``resp`` /
``resp_h`` (the top ``1 - --resp_q`` of each unit by responder score ``s`` /
posterior ``h``), and ``recon`` (the VAE's reconstructions: the floor).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.crop_flow_dataset import (CropFlowDataset, assign_cond_idx,     # noqa: E402
                               collate, latent_keys, load_latents,
                               load_cond_table, load_training_index,
                               resolve_scaling_factor)
from flow.dit import DiTConfig, DiTVelocity                             # noqa: E402
from flow.flow_matching import FlowConfig, RectifiedFlowBag             # noqa: E402
from autoencoder.utils import to_minus_one_one, to_zero_one                    # noqa: E402
from autoencoder.vae_wrapper import load_frozen_vae                            # noqa: E402


def to_uint8_range(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] float -> [0,1] float quantised to 8 bits, as the real uint8 crops."""
    x = torch.clamp(to_zero_one(x), 0.0, 1.0)
    return torch.floor(x * 255.0).float() / 255.0


# CellFlux's 3-channel view for Inception on 5/6-channel data: CPG keeps its
# first three channels; RxRx1 is a weighted composite taken in [-1, 1].
RXRX1_TO_RGB = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0],
                             [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def fid_view(x: torch.Tensor) -> torch.Tensor:
    """[B,C,H,W] in [0,1] -> [B,3,H,W] in [0,1], quantised like the real crops."""
    C = x.shape[1]
    if C == 3:
        return x
    if C == 5:
        y = x[:, :3]
    elif C == 6:
        y = torch.einsum("bchw,cn->bnhw", x * 2 - 1,
                         RXRX1_TO_RGB.to(x)).clamp(-1, 1) * 0.5 + 0.5
    else:
        raise SystemExit(f"no Inception view defined for {C} channels")
    # the 1e-3 keeps float round-off from pushing an exact k/255 down a level
    return torch.floor(y.clamp(0, 1) * 255.0 + 1e-3) / 255.0


def save_pngs(root: str, imgs: torch.Tensor, compounds, crop_ids) -> None:
    """Write the generated crops as 8-bit RGB PNGs at
    <root>/<compound>/<crop_id>.png (CellFlux's layout), stored channel order."""
    from PIL import Image
    x = (imgs.detach().clamp(0, 1) * 255).round().to(torch.uint8)
    for img, c, k in zip(x.permute(0, 2, 3, 1).cpu().numpy(), compounds, crop_ids):
        d = os.path.join(root, c)            # a compound name with '/' nests
        os.makedirs(d, exist_ok=True)
        Image.fromarray(img, "RGB").save(os.path.join(d, f"{k}.png"))


def responder_reference(trt: pd.DataFrame, resp_q: float,
                        by: str = "s") -> np.ndarray:
    """Boolean mask: the top (1 - resp_q) of each unit by ``by`` (s or h)."""
    thr = trt.groupby("unit_id")[by].transform(lambda v: v.quantile(resp_q))
    return (trt[by] >= thr).to_numpy()


def _verify_pairing(ds, rows, ctrl_batch) -> None:
    """The replayed control rows must be the images the loader actually gave."""
    import torch as _t
    for k, j in enumerate(rows):
        want = ds._load(ds.c_path[int(j)]).float()
        got = ctrl_batch[k].detach().cpu().float()
        if not _t.allclose(want, got, atol=1e-5):
            raise SystemExit(
                "paired_control_rows() disagrees with the batch the loader "
                "produced, so every dumped control crop would carry the wrong "
                "crop_id. CropFlowDataset.__getitem__'s partner draw has "
                "changed; re-sync paired_control_rows with it.")
    print(f"[dump] control pairing replay verified on {len(rows)} crops")


def paired_control_rows(ds) -> np.ndarray:
    """Which control row CropFlowDataset pairs with each treated row.

    Replays ``__getitem__``'s partner draw (seeded by ``(seed, epoch, i)``);
    ``_verify_pairing`` checks the replay against the first batch.
    """
    out = np.empty(len(ds), dtype=np.int64)
    for i in range(len(ds)):
        rng = np.random.default_rng((ds.seed, ds.epoch, i))
        pool = ds.ctrl_by_plate[ds.t_plate[i]]
        out[i] = int(pool[rng.integers(len(pool))])
    return out


def dump_meta(df: pd.DataFrame, idx, y: int) -> pd.DataFrame:
    """The Metadata_* columns CellProfiler carries through to cp_features."""
    d = df.iloc[idx] if idx is not None else df
    cols = ("unit_id", "compound", "dose", "moa", "plate", "well",
            "well_id", "s")
    out = {c: (d[c].to_numpy() if c in d.columns else
               ("DMSO" if c == "compound" and y == 0 else
                "control" if c == "unit_id" and y == 0 else np.nan))
           for c in cols}
    out["y"] = y
    return pd.DataFrame(out, index=range(len(d)))


def prdc(real: np.ndarray, gen: np.ndarray, k: int = 5, n: int = 2000,
         seed: int = 0) -> Dict[str, float]:
    """Precision / recall / density / coverage (Naeem et al. 2020) on up to
    ``n`` samples per side, k-NN radii in the given feature space."""
    rs = np.random.default_rng(seed)
    R = real[rs.choice(len(real), min(n, len(real)), replace=False)]
    G = gen[rs.choice(len(gen), min(n, len(gen)), replace=False)]
    R = R.reshape(len(R), -1).astype(np.float64)
    G = G.reshape(len(G), -1).astype(np.float64)

    def cdist(x, y):
        d = (x * x).sum(1)[:, None] + (y * y).sum(1)[None] - 2.0 * x @ y.T
        return np.sqrt(np.maximum(d, 0.0))

    drr, dgg, drg = cdist(R, R), cdist(G, G), cdist(R, G)
    kk = min(k, len(R) - 1, len(G) - 1)
    # k-th nearest neighbour within each set, self excluded
    rad_r = np.partition(drr, kk, axis=1)[:, kk]
    rad_g = np.partition(dgg, kk, axis=1)[:, kk]
    inside_r = drg <= rad_r[:, None]          # [n_real, n_gen]
    inside_g = drg <= rad_g[None, :]
    return {
        "precision": float(inside_r.any(axis=0).mean()),
        "recall": float(inside_g.any(axis=1).mean()),
        "density": float(inside_r.sum(axis=0).mean() / kk),
        "coverage": float((drg.min(axis=1) <= rad_r).mean()),
        "prdc_k": int(kk), "prdc_n": int(min(len(R), len(G))),
    }


def energy_distance(a: np.ndarray, b: np.ndarray, n: int = 2000,
                    seed: int = 0) -> float:
    """Energy distance between two latent samples (up to ``n`` per side):
    the Inception-free companion to FID, in the VAE latent."""
    rng = np.random.default_rng(seed)

    def take(x):
        idx = rng.choice(len(x), min(n, len(x)), replace=False)
        return torch.from_numpy(
            np.ascontiguousarray(x[idx].reshape(len(idx), -1))).float()

    a, b = take(a), take(b)
    ab = torch.cdist(a, b).mean()
    aa = torch.cdist(a, a).mean()
    bb = torch.cdist(b, b).mean()
    return float(2 * ab - aa - bb)


# --------------------------------------------------------------------------- #
REFS = ("all", "resp", "resp_h", "recon", "recon_resp")


def build_metrics(device, kid_subset: int):
    """One FID/KID pair per reference, or a clear message about what to install.

    torchmetrics only raises for a missing ``torch-fidelity`` when the metric
    is built, so the construction is guarded too.
    """
    os.environ.setdefault("USE_TF", "0")
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance

        def mk():
            return (FrechetInceptionDistance(normalize=True).to(device),
                    KernelInceptionDistance(subset_size=kid_subset,
                                            normalize=True).to(device))

        return {k: mk() for k in REFS}
    except Exception as e:                                        # noqa: BLE001
        raise SystemExit(
            f"FID/KID need torchmetrics + torch-fidelity "
            f"({type(e).__name__}: {str(e)[:160]}).\n"
            f"  pip install torch-fidelity\n"
            f"Or run with --no_inception for the latent energy distance only, "
            f"which needs neither and is the metric that does not assume "
            f"Inception features mean anything on fluorescence microscopy.")


def compute(cfg: Dict, ckpt_path: str, out_dir: Path, a) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_split = cfg.get("eval_split", "val")

    # eval_* is set by the trainer for a dilution set: the reference must be
    # the real undiluted population, identical for every arm.
    fx = cfg.get("eval_flow_index") or cfg["flow_index"]
    cs = cfg.get("eval_crop_scores") or cfg["crop_scores"]
    cu = cfg.get("eval_crop_universe") or cfg.get("crop_universe")
    if fx != cfg["flow_index"]:
        print(f"[eval] reference from the REAL index {fx}, not the training set")
    trt, ctl, info = load_training_index(
        fx, cs, train_splits=[eval_split],
        control_splits=cfg.get("eval_control_splits",
                               ["train", "test"]),
        crop_universe=cu,
        require_all_scored=not a.allow_unscored, warn_val=False)
    # Loaded here, not next to the model, because a compound-indexed table
    # rewrites `text_idx` and the dataset reads that column when it is built.
    text_table = None
    if cfg.get("text_emb"):
        text_table, cpd_row = load_cond_table(cfg["text_emb"],
                                              int(cfg["dit"]["text_dim"]))
        text_table = text_table.to(device)
        if cpd_row is not None:
            trt = assign_cond_idx(trt, cpd_row)

    if cfg.get("eval_units"):
        keep = set(pd.read_csv(cfg["eval_units"])["unit_id"].astype(str))
        trt = trt[trt["unit_id"].astype(str).isin(keep)].reset_index(drop=True)
        print(f"[eval] restricted to {len(keep)} study unit(s): {len(trt)} crops")
    if a.limit:
        trt = trt.head(a.limit).reset_index(drop=True)

    resp = responder_reference(trt, a.resp_q)
    print(f"[eval] split {eval_split!r}: {len(trt)} crops, "
          f"{int(resp.sum())} in the responder reference "
          f"(top {1 - a.resp_q:.0%} of each of {trt.unit_id.nunique()} units); "
          f"mean s {trt.s.mean():.3f} all vs {trt.s[resp].mean():.3f} resp")
    if "h" not in trt.columns:
        raise SystemExit(f"{cs} has no `h` column, and the resp_h reference "
                         f"ranks by the scorer's posterior. Rebuild it with "
                         f"cache_scores.py.")
    # s ties at its upper clip wherever a unit's pi_hat is 1.0, and the `>=`
    # then takes the whole unit; the posterior h has no such boundary.
    resp_h = responder_reference(trt, a.resp_q, by="h")
    print(f"[eval] resp_h: {int(resp_h.sum())} crops by the posterior h, "
          f"{int((resp & resp_h).sum())} of them also in resp")

    lat = row_of = lmeta = None
    if cfg.get("latents"):
        lat, row_of, lmeta = load_latents(
            cfg["latents"], list(latent_keys(trt)) + list(latent_keys(ctl)))

    from datasets.bbbc021_dataset import default_bbbc_transform
    tf = default_bbbc_transform(cfg["img_size"], cfg.get("n_channels", 3))
    # seeds the partner draw and the source noise; default is the config's seed
    seed = (a.eval_seed if getattr(a, "eval_seed", None) is not None
            else cfg.get("seed", 0))
    # gamma=0: weights are not used here, only the pairing and the images.
    ds = CropFlowDataset(trt, ctl, gamma=0.0, transform=tf,
                         seed=seed,
                         scaling_factor=cfg["vae"].get("scaling_factor", 1.0))
    ds_lat = (CropFlowDataset(trt, ctl, gamma=0.0, transform=tf,
                              seed=seed, latents=lat,
                              latent_row=row_of,
                              scaling_factor=cfg["vae"].get("scaling_factor", 1.0))
              if lat is not None else None)

    vae = load_frozen_vae(cfg["vae"]["ckpt"], cfg["vae"]["model"],
                          use_ema=cfg["vae"].get("use_ema", True),
                          scaling_factor=cfg["vae"].get("scaling_factor", 1.0),
                          device=device)
    C_l, H_l, _ = vae.latent_shape
    model = DiTVelocity(DiTConfig(latent_channels=C_l, latent_size=H_l,
                                  **cfg["dit"])).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    which = "model"
    if a.weights == "ema":
        if ck.get("ema"):
            which = "ema"
        else:
            print("  ! --weights ema but the checkpoint has none; using the "
                  "online weights. CellFlux evaluates EMA, so the numbers are "
                  "not comparable to theirs.")
    model.load_state_dict(ck[which])
    model.eval()
    print(f"  weights: {which}")
    gamma = float(ck.get("gamma", float("nan")))
    flow = RectifiedFlowBag(FlowConfig(**cfg["flow"]))
    # --source_noise: start the ODE from control + N(0, source_noise_std^2),
    # as CellFlux's eval does; otherwise from the clean control latent.
    src_std = (float(flow.cfg.source_noise_std)
               if getattr(a, "source_noise", False) else 0.0)
    src_gen = torch.Generator(device=device).manual_seed(seed)
    if src_std:
        print(f"[eval] source: control + N(0, {src_std:g}^2) in latent space, "
              f"as CellFlux's eval")

    M = None if a.no_inception else build_metrics(device, a.kid_subset)
    # --dump_feats: the per-crop features behind FID/KID, for dist_metrics.py.
    # The net is the FID metric's own, or the same torchmetrics Inception when
    # --no_inception skipped the metrics.
    feat_net = feats_r = feats_g = None
    if getattr(a, "dump_feats", None):
        from evaluation import inception_feats
        feat_net = (M["all"][0].inception if M is not None
                    else inception_feats.inception_net(device))
        feats_r, feats_g = [], []
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, collate_fn=collate)
    lat_loader = (DataLoader(ds_lat, batch_size=a.batch_size, shuffle=False,
                             num_workers=a.num_workers, collate_fn=collate)
                  if ds_lat is not None else None)

    dumper = ctrl_rows = None
    if a.dump_images:
        from cellprofiler_eval.cp_io import (CropWriter, shared_manifest,
                                             DEFAULT_CHANNEL_NAMES,
                                             CHANNEL_CONFLICT)
        chans = (a.dump_channels.split(",") if a.dump_channels
                 else list(DEFAULT_CHANNEL_NAMES))
        print(f"[dump] {CHANNEL_CONFLICT}\n")
        shared_manifest(a.dump_images, trt["crop_id"].astype(str).tolist(),
                        cfg["vae"]["ckpt"], seed, chans)
        # a --tag is a different generation setting of the same checkpoint,
        # so it gets its own gen__ dir
        dump_arm = out_dir.name + (f"__{a.tag}" if getattr(a, "tag", None) else "")
        dumper = CropWriter(a.dump_images, chans, arm=dump_arm)
        ctrl_rows = paired_control_rows(ds)
        print(f"[dump] -> {a.dump_images}  populations "
              f"{a.dump_populations}  arm {dump_arm}")

    # MoA, if a head is given: the same frozen classifier reads the same batch
    # FID reads, so MoA and FID are about the same generated images
    moa = None
    if getattr(a, "moa_head", None):
        from evaluation import moa_eval
        sub = getattr(a, "moa_subset", None)
        moa = moa_eval.MoAScorer(a.moa_head, device,
                                 subset=pd.read_csv(sub)["crop_id"] if sub else None)
        print(f"[moa] scoring gen / real / recon with {a.moa_head} "
              f"({len(moa.classes)} classes, keys {moa.prefix}_*)")

    pheno = None
    if getattr(a, "pheno_dir", None):
        from evaluation import pheno_eval
        pheno = pheno_eval.PhenoScorer(a.pheno_dir, trt, device,
                                       null_size=getattr(a, "pheno_null", 100_000))
        print(f"[pheno] top-k + replicate mAP for gen / real / recon from "
              f"{a.pheno_dir} ({len(pheno.classes)} classes, heads for "
              f"{len(pheno.heads)} group(s))")
    n_ch = int(cfg.get("n_channels", 3))

    z_real, z_gen, n, t0 = [], [], 0, time.time()
    it = zip(loader, lat_loader) if lat_loader else ((b, None) for b in loader)
    for (ctrl_i, trt_i, text, _), latb in it:
        b = len(text)
        sl = slice(n, n + b)
        mask = torch.from_numpy(resp[sl])
        mask_h = torch.from_numpy(resp_h[sl])
        ctrl_i = ctrl_i.to(device).float()
        trt_i = trt_i.to(device).float()
        c = (text_table[text.to(device)] if text_table is not None
             else torch.zeros(b, cfg["dit"]["text_dim"], device=device))
        if c.shape != (b, cfg["dit"]["text_dim"]):
            raise SystemExit(f"conditioning must be [B, text_dim] = "
                             f"{(b, cfg['dit']['text_dim'])}, got {tuple(c.shape)}")

        with torch.no_grad():
            if latb is not None:
                z0, z1 = latb[0].to(device).float(), latb[1].to(device).float()
            else:
                z0 = vae.encode_flat(to_minus_one_one(ctrl_i, cfg.get("input_range", "01")))
                z1 = vae.encode_flat(to_minus_one_one(trt_i, cfg.get("input_range", "01")))
            if src_std:
                z0 = z0 + src_std * torch.randn(z0.shape, generator=src_gen,
                                                device=device)
            # the identity model: the control latent, unmoved
            z1_hat = z0 if getattr(a, "passthrough", False) else flow.sample(
                model, z0, c, num_steps=a.steps, cfg_scale=a.cfg_scale,
                method=a.ode_method, edm_schedule=not a.no_edm_schedule)
            d_gen, d_rec = vae.decode(z1_hat), vae.decode(z1)
            gen, recon = to_uint8_range(d_gen), to_uint8_range(d_rec)
        real = torch.clamp(trt_i, 0.0, 1.0)      # already [0,1] off disk
        if n_ch == 3:
            f_gen, f_real, f_rec = gen, real, recon
        else:   # reduce before quantising, as CellFlux does
            f_gen, f_rec = (fid_view(torch.clamp(to_zero_one(d), 0.0, 1.0))
                            for d in (d_gen, d_rec))
            f_real = fid_view(real)
        if feat_net is not None:
            feats_r.append(inception_feats.features(feat_net, f_real))
            feats_g.append(inception_feats.features(feat_net, f_gen))
        if getattr(a, "save_pngs", None):
            # f_gen, not gen: at C != 3 CellFlux's scorers expect the
            # 3-channel reduction
            save_pngs(os.path.join(a.save_pngs, out_dir.name), f_gen,
                      trt["compound"].to_numpy()[sl],
                      trt["crop_id"].astype(str).to_numpy()[sl])
        if moa is not None:
            for pop, imgs in (("gen", gen), ("real", real), ("recon", recon)):
                moa.add(pop, imgs)
        if pheno is not None:
            for pop, imgs in (("gen", gen), ("real", real), ("recon", recon)):
                pheno.add(pop, imgs, sl)

        if M is not None:
            def upd(key, imgs, is_real):
                for m in M[key]:
                    m.update(imgs, real=is_real)

            upd("all", f_real, True);   upd("all", f_gen, False)
            upd("recon", f_real, True); upd("recon", f_rec, False)
            # the generated set goes to every reference unchanged: the model
            # does not get to pick which crops it is judged against
            upd("resp", f_gen, False)
            upd("resp_h", f_gen, False)
            if mask.any():
                mk_ = mask.to(real.device)
                upd("resp", f_real[mk_], True)
                upd("recon_resp", f_real[mk_], True)
                upd("recon_resp", f_rec[mk_], False)
            if mask_h.any():
                upd("resp_h", f_real[mask_h.to(real.device)], True)

        if dumper is not None and (a.dump_n <= 0 or n < a.dump_n):
            want = set(a.dump_populations.split(","))
            cj = ctrl_rows[sl]
            if n == 0:
                _verify_pairing(ds, cj[:min(4, b)], ctrl_i[:min(4, b)])
            ids_t = trt["crop_id"].astype(str).to_numpy()[sl]
            ids_c = ctl["crop_id"].astype(str).to_numpy()[cj]
            for pop, imgs, ids, meta in (
                    ("real", real, ids_t, dump_meta(trt, np.arange(n, n + b), 1)),
                    ("recon", recon, ids_t, dump_meta(trt, np.arange(n, n + b), 1)),
                    ("gen", gen, ids_t, dump_meta(trt, np.arange(n, n + b), 1)),
                    ("control", ctrl_i, ids_c, dump_meta(ctl, cj, 0))):
                if pop in want:
                    dumper.add(imgs.detach().cpu().numpy(), ids, pop, meta)

        z_real.append(z1.cpu().numpy()); z_gen.append(z1_hat.cpu().numpy())
        n += b
        if n % (a.batch_size * 20) < a.batch_size:
            el = time.time() - t0
            print(f"  {n}/{len(trt)}  {el:.0f}s  eta "
                  f"{el / max(n, 1) * (len(trt) - n):.0f}s")

    if dumper is not None:
        dumper.write(provenance={
            "arm": out_dir.name, "ckpt": ckpt_path, "gamma": gamma,
            "eval_split": eval_split, "eval_flow_index": fx,
            "eval_units": cfg.get("eval_units"), "vae_ckpt": cfg["vae"]["ckpt"],
            "seed": seed, "steps": a.steps,
            "cfg_scale": a.cfg_scale, "weights": which,
            "source_noise_std": src_std,
            "populations": a.dump_populations, "n_crops": n})

    if feat_net is not None:
        inception_feats.write_feats(a.dump_feats, out_dir.name + out_tag(a),
                                    trt, np.concatenate(feats_r),
                                    np.concatenate(feats_g))

    z_real = np.concatenate(z_real); z_gen = np.concatenate(z_gen)
    row = {"arm": out_dir.name, "gamma": gamma, "ckpt": ckpt_path,
           "epoch": int(ck.get("epoch", -1)), "step": int(ck.get("global_step", -1)),
           "n_eval": int(n), "n_resp": int(resp.sum()),
           "n_resp_h": int(resp_h.sum()), "resp_q": a.resp_q,
           "eval_split": eval_split, "steps": a.steps, "cfg_scale": a.cfg_scale,
           "eval_units": cfg.get("eval_units"), "eval_flow_index": fx,
           "n_eval_units": int(trt["unit_id"].nunique()),
           "passthrough": bool(getattr(a, "passthrough", False)),
           "weights": which, "ode_method": a.ode_method,
           "edm_schedule": not a.no_edm_schedule, "source_noise_std": src_std,
           "eval_seed": int(seed), "tag": getattr(a, "tag", None)}
    for k in REFS:
        if M is None:
            row[f"fid_{k}"] = row[f"kid_{k}"] = row[f"kid_{k}_std"] = float("nan")
            continue
        fid, kid = M[k]
        row[f"fid_{k}"] = float(fid.compute())
        km, ks = kid.compute()
        row[f"kid_{k}"] = float(km)
        row[f"kid_{k}_std"] = float(ks)
    if getattr(a, "dump_latents", None):
        dump_latents(a.dump_latents, out_dir.name, trt, z_real, z_gen, resp)
    row["energy_latent_all"] = energy_distance(z_real, z_gen)
    row["energy_latent_resp"] = energy_distance(z_real[resp], z_gen)
    row["energy_latent_resp_h"] = energy_distance(z_real[resp_h], z_gen)
    row.update(prdc(z_real, z_gen, k=a.prdc_k, seed=cfg.get("seed", 0)))
    row.update({f"{k}_resp": v for k, v in
                prdc(z_real[resp], z_gen, k=a.prdc_k,
                     seed=cfg.get("seed", 0)).items()
                if k in ("precision", "recall", "density", "coverage")})
    if moa is not None:
        row.update(moa.finish(trt, out_dir,
                              name=f"moa_preds{moa.tag}{out_tag(a)}.parquet"))
    if pheno is not None:
        row.update(pheno.finish(out_dir, name=f"pheno{out_tag(a)}"))
    row["seconds"] = round(time.time() - t0, 1)
    return row


# Two rows with these equal describe the same model scored on the same reference.
EVAL_IDENTITY = ("ckpt", "step", "n_eval", "n_resp", "eval_units", "cfg_scale")


def merge_rows(old: Dict, new: Dict) -> Dict:
    """``new``, with every missing / NaN value filled from ``old`` -- if they
    are the same evaluation (EVAL_IDENTITY); otherwise ``new`` untouched.

    Keeps a ``--no_inception`` rerun from erasing an arm's stored FID.
    """
    if any(old[k] != new[k] for k in EVAL_IDENTITY if k in old and k in new):
        return new

    def missing(v):
        return v is None or (isinstance(v, float) and np.isnan(v))

    out = dict(new)
    for k, v in old.items():
        if missing(out.get(k)) and not missing(v):
            out[k] = v
    return out


def dump_latents(out: str, arm: str, trt, z_real, z_gen, resp):
    """z_real once per eval set, z_gen per arm, plus the index they share.
    Stored as float16."""
    os.makedirs(out, exist_ok=True)
    cols = [c for c in ("crop_id", "unit_id", "compound", "plate", "well",
                        "well_id", "s") if c in trt.columns]
    idx = trt[cols].copy()
    idx["resp"] = resp
    ip = os.path.join(out, "index.parquet")
    if not os.path.exists(ip):
        idx.to_parquet(ip, index=False)
    elif len(pd.read_parquet(ip, columns=["crop_id"])) != len(idx):
        raise SystemExit(f"{ip} has a different eval set; point --dump_latents "
                         f"somewhere else")
    rp = os.path.join(out, "z_real.npy")
    if not os.path.exists(rp):
        np.save(rp, z_real.astype(np.float16))
    # named by arm alone: one dump dir holds one generation setting
    np.save(os.path.join(out, f"z_gen__{arm}.npy"), z_gen.astype(np.float16))
    print(f"[dump_latents] {out}  z_gen__{arm}.npy {z_gen.shape} float16")


def out_tag(a) -> str:
    """File-name tag for a --source_noise and/or --tag evaluation, so it never
    overwrites the arm's default eval_metrics.json / moa_preds.parquet."""
    tag = getattr(a, "tag", None)
    return (("_srcnoise" if getattr(a, "source_noise", False) else "")
            + (f"_{tag}" if tag else ""))


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--arms", default=None,
                   help="comma-separated arm dir names; default every gamma*/")
    p.add_argument("--ckpt", default="last.pt")
    p.add_argument("--weights", default="ema", choices=("ema", "model"),
                   help="EMA is what CellFlux evaluates; `model` is the online "
                        "weights.")
    p.add_argument("--out", default=None, help="csv path; default <out_dir>/eval_metrics.csv")
    p.add_argument("--resp_q", type=float, default=0.5,
                   help="per-unit quantile of s defining the responder "
                        "reference; 0.5 = the top half of every unit")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=None,
                   help="default: the config's cfg_scale. Our convention, "
                        "uncond + s*(cond-uncond): CellFlux's 0.2 is s=1.2")
    p.add_argument("--ode_method", default="heun2", choices=("heun2", "euler"))
    p.add_argument("--no_edm_schedule", action="store_true")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--kid_subset", type=int, default=100)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--allow_unscored", action="store_true")
    p.add_argument("--eval_units", default=None,
                   help="csv with a unit_id column, as the trainer's flag. "
                        "Required to re-evaluate a dilution arm by hand: the "
                        "auto-eval restricted the reference to the study "
                        "units, and without this the rerun uses all of them "
                        "and is not comparable with the arms that did not "
                        "need one.")
    p.add_argument("--no_inception", action="store_true",
                   help="skip FID/KID and report only the latent energy "
                        "distance. Needs no torch-fidelity, and is the metric "
                        "that does not assume Inception features transfer to "
                        "fluorescence microscopy.")
    p.add_argument("--prdc_k", type=int, default=5,
                   help="k for precision/recall/density/coverage on the "
                        "latents. Shrinkage is a coverage failure with "
                        "precision intact; FID cannot say that and this can.")
    p.add_argument("--dump_images", default=None, metavar="DIR",
                   help="also write CellProfiler-ready 16-bit TIFFs + a "
                        "LoadData csv for the real / recon / gen / control "
                        "populations, from inside this same generation loop -- "
                        "so CellProfiler measures exactly the images FID saw. "
                        "See cellprofiler_eval/.")
    p.add_argument("--dump_populations", default="real,recon,gen,control",
                   help="real and recon and control are identical across arms "
                        "and written once; only gen is per-arm. Pass `gen` "
                        "alone when adding a second arm to an existing dump.")
    p.add_argument("--dump_n", type=int, default=0,
                   help="0 = every eval crop. A cap is for smoke tests only: "
                        "the shrinkage ratio needs whole units.")
    p.add_argument("--dump_channels", default=None,
                   help="comma-separated channel names, stored order. Default "
                        "Actin,Tubulin,DNA, BBBC021's stored order "
                        "(cp_io.CHANNEL_CONFLICT). Pass it for 5/6-channel data.")
    p.add_argument("--moa_head", default=None,
                   help="moa_eval.py's head: also score MoA accuracy of gen / "
                        "real / recon on the same batches FID scores, and "
                        "write <arm>/moa_preds.parquet. CellFlux/moa/"
                        "checkpoint.pth as shipped also works: keys moa_cf_*, "
                        "file moa_preds_cf.parquet")
    p.add_argument("--moa_subset", default=None,
                   help="csv with a crop_id column; every MoA metric is also "
                        "reported on those crops (*_sub). cellflux_moa_subset.csv "
                        "is the 5,120 crops CellFlux's Table 2a scores")
    p.add_argument("--pheno_dir", default=None,
                   help="pheno_eval.py fit's output: also score perturbation "
                        "top-1/5/10 (leave-one-group-out heads) and replicate "
                        "mAP against real controls for gen / real / recon, "
                        "writing <arm>/pheno*_preds.parquet and _map.parquet")
    p.add_argument("--pheno_null", type=int, default=100_000,
                   help="null draws per (n_pos, n_total) for the mAP p-values")
    p.add_argument("--passthrough", action="store_true",
                   help="do not run the ODE: the 'generated' crop IS the paired "
                        "control crop, decoded through the same VAE. The "
                        "identity model, scored by every metric in the same "
                        "space, so `how much better than doing nothing is this "
                        "arm` has a number. Tags itself `passthrough` unless "
                        "--tag says otherwise.")
    p.add_argument("--dump_latents", default=None, metavar="DIR",
                   help="write z_real / z_gen (the VAE latents the energy and "
                        "PRDC rows are already computed from) plus an index, "
                        "as float16, for dist_metrics.py latent and "
                        "checks/nn_memo.py. z_real is "
                        "arm-independent and written once.")
    p.add_argument("--collect", action="store_true",
                   help="do not evaluate; gather the per-arm eval_metrics.json "
                        "written by `eval_after_train` into one csv.")
    p.add_argument("--source_noise", action="store_true",
                   help="sample from control + N(0, flow.source_noise_std^2), "
                        "as CellFlux's eval does (use_initial=2). Writes "
                        "eval_metrics_srcnoise.json/.csv and "
                        "moa_preds_srcnoise.parquet, so the clean-source files "
                        "are kept; with --collect it gathers those.")
    p.add_argument("--tag", default=None,
                   help="suffix for this run's eval_metrics / moa_preds files. "
                        "Required when the config's eval_split is not val "
                        "(the OOD crops), so it cannot overwrite the arm's val "
                        "results; --collect needs the same --tag.")
    p.add_argument("--save_pngs", default=None, metavar="DIR",
                   help="also write every generated crop as an 8-bit RGB PNG "
                        "at DIR/<arm>/<compound>/<crop_id>.png, CellFlux's "
                        "layout, so their eval_fid.py (FIDo / FIDc) and MoA "
                        "script score our images exactly as they score theirs. "
                        "At C != 3 the PNG is fid_view's 3-channel reduction, "
                        "their own convert_*ch_to_3ch.")
    p.add_argument("--dump_feats", default=None, metavar="DIR",
                   help="also write the per-crop 2048-d Inception features "
                        "FID/KID are computed from: DIR/index.parquet + "
                        "feat_real.npy once per eval set, feat_gen__<arm>"
                        "<suffix>.npy per arm and setting (the suffix is the "
                        "eval_metrics file's). dist_metrics.py re-estimates "
                        "every distribution metric from them on CPU. One DIR "
                        "per eval set (IID, OOD, each dataset).")
    p.add_argument("--eval_seed", type=int, default=None,
                   help="seed for the control-partner draw and the source "
                        "noise. Default: the config's seed, as every earlier "
                        "row. Pair it with --tag, or the arm's json is "
                        "overwritten.")
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.eval_units:
        cfg["eval_units"] = a.eval_units
    resolve_scaling_factor(cfg)
    # the config is the one place the guidance scale is set
    if a.cfg_scale is None:
        a.cfg_scale = float(cfg["cfg_scale"])
    if a.passthrough and not a.tag:
        a.tag = "passthrough"
    # fall back to the config's eval.pheno_dir, as the trainer's auto-eval does
    if a.pheno_dir is None:
        d = (cfg.get("eval") or {}).get("pheno_dir")
        if d and os.path.exists(os.path.join(d, "heads.pt")):
            a.pheno_dir = d
            print(f"[pheno] from the config: {d}")
    if cfg.get("eval_split", "val") != "val" and not a.tag:
        raise SystemExit(f"eval_split is {cfg['eval_split']!r}: pass --tag, or "
                         f"its rows overwrite the arms' val eval_metrics")
    if a.eval_seed is not None and not a.tag:
        raise SystemExit("--eval_seed needs --tag, or its row overwrites the "
                         "arm's config-seed eval_metrics")
    root = Path(cfg["out_dir"])
    arms = ([root / x for x in a.arms.split(",")] if a.arms
            else sorted(root.glob("gamma*")))

    name = f"eval_metrics{out_tag(a)}"
    if a.collect:
        rows = [json.load(open(d / f"{name}.json")) for d in arms
                if (d / f"{name}.json").exists()]
        if not rows:
            raise SystemExit(f"no arm under {root} has a {name}.json")
        print(f"[collect] {len(rows)} arm(s)")
    else:
        arms = [d for d in arms if (d / "ckpt" / a.ckpt).exists()]
        if not arms:
            raise SystemExit(f"no arm under {root} has ckpt/{a.ckpt}")
        print(f"[eval] {len(arms)} arm(s): {', '.join(d.name for d in arms)}")

        rows = []
        for d in arms:
            print(f"\n== {d.name} ==")
            rows.append(compute(cfg, str(d / "ckpt" / a.ckpt), d, a))
            p = d / f"{name}.json"
            if p.exists():          # a --no_inception rerun must not erase FID
                rows[-1] = merge_rows(json.load(open(p)), rows[-1])
            json.dump(rows[-1], open(p, "w"), indent=2)

    t = pd.DataFrame(rows).sort_values("gamma")
    out = a.out or str(root / f"{name}.csv")
    t.to_csv(out, index=False)

    show = ["arm", "gamma", "n_eval", "fid_all", "fid_resp", "fid_recon",
            "kid_all", "kid_resp", "energy_latent_all", "energy_latent_resp"]
    print("\n" + t[[c for c in show if c in t.columns]].to_string(index=False))
    print(f"\n-> {out}")

    if len(t) and not a.no_inception and (t["fid_recon"] > t["fid_all"] * 0.5).any():
        print("\n! the VAE reconstruction floor is over half of fid_all on at "
              "least one arm. That much of the distance is the autoencoder, "
              "not the flow; quote fid_recon beside it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
