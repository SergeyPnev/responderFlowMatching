"""DiT velocity field for latent flow matching.

Operates on a latent z: [B, C_l, H_l, W_l] (e.g. [B, 8, 12, 12]), conditioned
on time t and a compound embedding c:
patchify -> DiT blocks with adaLN-zero conditioning -> unpatchify.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of t in [0,1] followed by an MLP."""

    def __init__(self, hidden: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] in [0,1]
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None] * 1000.0
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class CondEmbedder(nn.Module):
    """Projects a compound embedding to the model width, and supplies the
    unconditional branch for classifier-free guidance.

    ``null_mode="learned"``: a learned ``null_emb`` replaces ``c`` and goes
        through the same MLP.
    ``null_mode="zero"``: the condition contributes exactly zero (CellFlux's
        convention). Not the same as embedding a zero vector: ``mlp(0) != 0``.
    """

    def __init__(self, text_dim: int, hidden: int, null_mode: str = "learned"):
        super().__init__()
        if null_mode not in ("learned", "zero"):
            raise ValueError(f"null_mode must be 'learned' or 'zero', "
                             f"got {null_mode!r}")
        self.null_mode = null_mode
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        # Kept in both modes so a checkpoint trained in one loads in the other
        # (state_dict keys must match); unused when null_mode == "zero".
        self.null_emb = nn.Parameter(torch.zeros(text_dim))

    def forward(self, c: torch.Tensor, drop_mask: torch.Tensor | None = None):
        # c: [B, text_dim]; drop_mask: [B] bool (True -> unconditional)
        if drop_mask is not None and self.null_mode == "learned":
            c = torch.where(drop_mask[:, None], self.null_emb[None], c)
        out = self.mlp(c)
        if drop_mask is not None and self.null_mode == "zero":
            out = torch.where(drop_mask[:, None], torch.zeros_like(out), out)
        return out


class DiTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = \
            self.adaLN(cond).chunk(6, dim=1)
        h = modulate(self.norm1(x), shift_a, scale_a)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a.unsqueeze(1) * attn_out
        h = modulate(self.norm2(x), shift_m, scale_m)
        x = x + gate_m.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden: int, patch_size: int, out_ch: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, patch_size * patch_size * out_ch)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, cond):
        shift, scale = self.adaLN(cond).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# --------------------------------------------------------------------------- #
@dataclass
class DiTConfig:
    latent_channels: int = 8
    latent_size: int = 12          # H_l = W_l
    patch_size: int = 2
    hidden: int = 384
    depth: int = 12
    heads: int = 6
    mlp_ratio: float = 4.0
    text_dim: int = 512            # conditioning width: 512 CellCLIP, 1024 emb_fp
    null_mode: str = "learned"     # "zero" = CellFlux's (see CondEmbedder)


class DiTVelocity(nn.Module):
    """v_theta(z_t, t, c) -> velocity, same shape as z_t."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        C, S, P = cfg.latent_channels, cfg.latent_size, cfg.patch_size
        assert S % P == 0, "latent_size must be divisible by patch_size"
        self.grid = S // P
        self.num_tokens = self.grid ** 2

        self.patch_embed = nn.Conv2d(C, cfg.hidden, kernel_size=P, stride=P)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, cfg.hidden))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.t_embed = TimestepEmbedder(cfg.hidden)
        self.c_embed = CondEmbedder(cfg.text_dim, cfg.hidden,
                                    null_mode=cfg.null_mode)

        self.blocks = nn.ModuleList(
            [DiTBlock(cfg.hidden, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.final = FinalLayer(cfg.hidden, P, C)

    # ------------------------------------------------------------------ #
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_tokens, P*P*C] -> [B, C, S, S]
        C, P, g = self.cfg.latent_channels, self.cfg.patch_size, self.grid
        x = x.reshape(x.shape[0], g, g, P, P, C)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(x.shape[0], C, g * P, g * P)

    def forward(self, z: torch.Tensor, t: torch.Tensor, c: torch.Tensor,
                drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        # z: [B, C_l, S, S]; t: [B]; c: [B, text_dim]
        x = self.patch_embed(z).flatten(2).transpose(1, 2)   # [B, N, hidden]
        x = x + self.pos_embed
        cond = self.t_embed(t) + self.c_embed(c, drop_mask)  # [B, hidden]
        for blk in self.blocks:
            x = blk(x, cond)
        x = self.final(x, cond)                              # [B, N, P*P*C]
        return self.unpatchify(x)                            # [B, C_l, S, S]

    @torch.no_grad()
    def forward_with_cfg(self, z, t, c, cfg_scale: float = 1.0):
        """Classifier-free guidance: v = v_uncond + s * (v_cond - v_uncond)."""
        if cfg_scale == 1.0:
            return self.forward(z, t, c)
        B = z.shape[0]
        drop_cond = torch.zeros(B, dtype=torch.bool, device=z.device)
        drop_uncond = torch.ones(B, dtype=torch.bool, device=z.device)
        v_cond = self.forward(z, t, c, drop_cond)
        v_uncond = self.forward(z, t, c, drop_uncond)
        return v_uncond + cfg_scale * (v_cond - v_uncond)