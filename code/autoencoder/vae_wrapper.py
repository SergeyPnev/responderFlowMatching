"""Frozen pre-trained VAE used as the latent encoder/decoder for the flow.

Encoding folds the bag dimension into the batch:
    [B, M, C, H, W]  ->  [B*M, C, H, W]  ->  encode  ->  [B, M, C_l, H_l, W_l]
"""
from __future__ import annotations

import torch
import torch.nn as nn

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoencoder.vae import AutoencoderKL, VAEConfig  # noqa: E402


class FrozenVAE(nn.Module):
    """Wraps a trained AutoencoderKL; parameters are frozen and never updated.

    `scaling_factor` normalises the latent to roughly unit variance before
    flow matching (1 / std of encoded latents over a sample of data).
    """

    def __init__(self, vae: AutoencoderKL, scaling_factor: float = 1.0,
                 sample_posterior: bool = False):
        super().__init__()
        self.vae = vae
        self.scaling_factor = scaling_factor
        self.sample_posterior = sample_posterior
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

    def train(self, mode: bool = True):
        # keep VAE in eval mode regardless of parent .train()
        super().train(mode)
        self.vae.eval()
        return self

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_bag(self, x: torch.Tensor) -> torch.Tensor:
        """[B, M, C, H, W] (in [-1,1]) -> [B, M, C_l, H_l, W_l]."""
        B, M, C, H, W = x.shape
        x = x.reshape(B * M, C, H, W)
        posterior = self.vae.encode(x)
        z = posterior.sample() if self.sample_posterior else posterior.mode()
        z = z * self.scaling_factor
        Cl, Hl, Wl = z.shape[1:]
        return z.reshape(B, M, Cl, Hl, Wl)

    @torch.no_grad()
    def encode_flat(self, x: torch.Tensor) -> torch.Tensor:
        """[N, C, H, W] -> [N, C_l, H_l, W_l]."""
        posterior = self.vae.encode(x)
        z = posterior.sample() if self.sample_posterior else posterior.mode()
        return z * self.scaling_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[N, C_l, H_l, W_l] -> [N, C, H, W] in [-1,1]."""
        return self.vae.decode(z / self.scaling_factor)

    def decode_grad(self, z: torch.Tensor) -> torch.Tensor:
        """Same as `decode` but keeps the graph: gradients flow back into `z`,
        the VAE parameters stay frozen."""
        return self.vae.decode(z / self.scaling_factor)

    @property
    def latent_shape(self):
        """(C_l, H_l, W_l) for the configured resolution."""
        cfg: VAEConfig = self.vae.cfg
        f = 2 ** (len(cfg.ch_mult) - 1)
        return (cfg.z_channels, cfg.resolution // f, cfg.resolution // f)


def load_frozen_vae(ckpt_path: str, vae_cfg_dict: dict, use_ema: bool = True,
                     scaling_factor: float = 1.0, device="cpu") -> FrozenVAE:
    """Load a trained VAE checkpoint produced by train.py."""
    cfg = VAEConfig(**vae_cfg_dict)
    vae = AutoencoderKL(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["ema"] if (use_ema and "ema" in ckpt) else ckpt["vae"]
    vae.load_state_dict(state)
    frozen = FrozenVAE(vae, scaling_factor=scaling_factor).to(device)
    return frozen
