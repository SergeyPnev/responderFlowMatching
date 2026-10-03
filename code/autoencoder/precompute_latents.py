"""Encode every crop in the flow index once, with the frozen VAE.

``FrozenVAE`` takes ``posterior.mode()``, so encoding is deterministic and the
cache changes no result. Latents are written unscaled (``scaling_factor = 1``)
and the measured ``1/std`` is reported at the end; the factor is set in the
config and applied at load time. The trainer memory-maps the output.

    python autoencoder/precompute_latents.py --index flow_index.parquet \\
        --vae_config configs/cellflux_percrop_bbbc_fp.yaml --out latents/
    python autoencoder/precompute_latents.py ... --limit 2000        # smoke first
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoencoder.utils import to_minus_one_one                                # noqa: E402
from autoencoder.vae_wrapper import load_frozen_vae                           # noqa: E402


class _Crops(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = np.load(self.paths[i])
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img.float()


def file_sha(path: str, cap: int = 64 << 20) -> str:
    """Digest of the VAE checkpoint, so a latent cache cannot outlive its VAE."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while len(chunk := f.read(1 << 20)) and cap > 0:
            h.update(chunk)
            cap -= len(chunk)
    return h.hexdigest()[:16]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--index", required=True, help="flow_index.parquet")
    p.add_argument("--vae_config", required=True,
                   help="a yaml with a `vae:` block (the training config is fine)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--std_sample", type=int, default=20000,
                   help="crops used to measure 1/std for scaling_factor")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.vae_config))
    os.makedirs(a.out, exist_ok=True)
    lat_path = os.path.join(a.out, "latents.npy")
    if os.path.exists(lat_path) and not a.force:
        raise SystemExit(f"{lat_path} exists; --force to overwrite. Every run "
                         f"trained against it would need re-running.")

    idx = pd.read_parquet(a.index)
    if a.limit:
        idx = idx.head(a.limit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # scaling_factor=1: the latents go to disk raw, and the factor is applied
    # at load time so it can be changed without re-encoding.
    vae = load_frozen_vae(ckpt_path=cfg["vae"]["ckpt"],
                          vae_cfg_dict=cfg["vae"]["model"],
                          use_ema=cfg["vae"].get("use_ema", True),
                          scaling_factor=1.0, device=device)
    C_l, H_l, W_l = vae.latent_shape
    print(f"[vae] {cfg['vae']['ckpt']}\n[vae] latent ({C_l}, {H_l}, {W_l})  "
          f"ema={cfg['vae'].get('use_ema', True)}  device={device}")

    from datasets.bbbc021_dataset import default_bbbc_transform
    tf = default_bbbc_transform(cfg["img_size"], cfg.get("n_channels", 3))
    loader = DataLoader(_Crops(idx["path"].to_numpy(), tf),
                        batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, pin_memory=True)

    dt = np.float32 if a.dtype == "float32" else np.float16
    out = np.lib.format.open_memmap(lat_path, mode="w+", dtype=dt,
                                    shape=(len(idx), C_l, H_l, W_l))
    in_range = cfg.get("input_range", "01")
    n, t0 = 0, time.time()
    for bi, x in enumerate(loader):
        x = to_minus_one_one(x.to(device, non_blocking=True).float(), in_range)
        z = vae.encode_flat(x)
        out[n:n + len(z)] = z.cpu().numpy().astype(dt)
        n += len(z)
        if bi % 50 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(idx)}  {el:.0f}s  eta "
                  f"{el / max(n, 1) * (len(idx) - n):.0f}s")
    out.flush()
    assert n == len(idx), f"encoded {n} of {len(idx)}"

    # ---- scaling factor ---------------------------------------------------- #
    k = min(a.std_sample, len(idx))
    sample = np.asarray(out[np.random.default_rng(0).choice(len(idx), k, False)],
                        dtype=np.float64)
    std = float(sample.std())
    per_ch = sample.std(axis=(0, 2, 3))
    print(f"\n[latents] std {std:.4f} over {k} crops  ->  "
          f"scaling_factor = {1 / std:.6f}")
    print("[latents] per-channel std: "
          + ", ".join(f"{v:.3f}" for v in per_ch))
    if per_ch.max() / max(per_ch.min(), 1e-9) > 5:
        print("  ! per-channel std spans >5x. One global scaling_factor leaves "
              "the quiet channels near zero; note it before reading the flow.")

    meta = {
        "n": int(len(idx)), "latent_shape": [C_l, H_l, W_l], "dtype": a.dtype,
        "input_range": in_range, "img_size": cfg["img_size"],
        "vae_ckpt": cfg["vae"]["ckpt"], "vae_ckpt_sha": file_sha(cfg["vae"]["ckpt"]),
        "vae_use_ema": bool(cfg["vae"].get("use_ema", True)),
        "scaling_factor_applied": 1.0,
        "measured_std": std, "suggested_scaling_factor": 1 / std,
        "per_channel_std": [float(v) for v in per_ch],
        "index": os.path.abspath(a.index),
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(meta, open(os.path.join(a.out, "latents_meta.json"), "w"), indent=2)
    idx[["crop_id"]].assign(row=np.arange(len(idx))).to_parquet(
        os.path.join(a.out, "latents_index.parquet"), index=False)

    sz = os.path.getsize(lat_path) / 1e9
    print(f"\n[latents] {len(idx)} crops, {sz:.2f} GB -> {a.out}")
    print(f"[latents] set in the config:\n"
          f"  latents: {os.path.abspath(a.out)}\n"
          f"  vae:\n    scaling_factor: {1 / std:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
