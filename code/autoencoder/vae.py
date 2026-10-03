"""LDM-style KL-regularised autoencoder for cell-painting images.

Shapes for the default 96x96 configuration with f=8:

    input:        B x C_in x 96 x 96
    encoder out:  B x 16  x 12 x 12   (mean | logvar split into 8 + 8)
    sample z:     B x 8   x 12 x 12
    decoder out:  B x C_in x 96 x 96

Set ``in_channels`` to 3 for BBBC021, 5 for CPG, 6 for RxRx1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Basic blocks
# --------------------------------------------------------------------------- #
def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def Normalize(in_channels: int, num_groups: int = 32) -> nn.Module:
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int | None = None, dropout: float = 0.0):
        super().__init__()
        out_ch = out_ch or in_ch
        self.in_ch, self.out_ch = in_ch, out_ch

        self.norm1 = Normalize(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm2 = Normalize(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, 1, 0) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    """Single-head self-attention over spatial tokens."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.norm = Normalize(in_ch)
        self.q = nn.Conv2d(in_ch, in_ch, 1)
        self.k = nn.Conv2d(in_ch, in_ch, 1)
        self.v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        B, C, H, W = q.shape
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # B, N, C
        k = k.reshape(B, C, H * W)                    # B, C, N
        attn = torch.bmm(q, k) * (C ** -0.5)          # B, N, N
        attn = torch.softmax(attn, dim=-1)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)   # B, N, C
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# --------------------------------------------------------------------------- #
# Encoder / Decoder
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        z_channels: int = 8,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (12,),
        dropout: float = 0.0,
        double_z: bool = True,
        resolution: int = 96,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # stem
        self.conv_in = nn.Conv2d(in_channels, ch, 3, 1, 1)

        # down stages
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResBlock(block_in, block_out, dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            stage = nn.Module()
            stage.block = block
            stage.attn = attn
            if i_level != self.num_resolutions - 1:
                stage.downsample = Downsample(block_in)
                curr_res //= 2
            self.down.append(stage)

        # mid
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(block_in, block_in, dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResBlock(block_in, block_in, dropout)

        # head
        self.norm_out = Normalize(block_in)
        out_z = 2 * z_channels if double_z else z_channels
        self.conv_out = nn.Conv2d(block_in, out_z, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for i_level, stage in enumerate(self.down):
            for i_block in range(self.num_res_blocks):
                h = stage.block[i_block](h)
                if len(stage.attn) > 0:
                    h = stage.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = stage.downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.conv_out(nonlinearity(self.norm_out(h)))
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        z_channels: int = 8,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 3,
        attn_resolutions: Tuple[int, ...] = (12,),
        dropout: float = 0.0,
        resolution: int = 96,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # start at smallest resolution
        block_in = ch * ch_mult[-1]
        curr_res = resolution // (2 ** (self.num_resolutions - 1))

        # stem from latent
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)

        # mid
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(block_in, block_in, dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResBlock(block_in, block_in, dropout)

        # up stages (reverse)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResBlock(block_in, block_out, dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            stage = nn.Module()
            stage.block = block
            stage.attn = attn
            if i_level != 0:
                stage.upsample = Upsample(block_in)
                curr_res *= 2
            self.up.insert(0, stage)  # keep ascending order in indexing

        # head
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, 3, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            stage = self.up[i_level]
            for i_block in range(self.num_res_blocks):
                h = stage.block[i_block](h)
                if len(stage.attn) > 0:
                    h = stage.attn[i_block](h)
            if i_level != 0:
                h = stage.upsample(h)
        h = self.conv_out(nonlinearity(self.norm_out(h)))
        return h


# --------------------------------------------------------------------------- #
# Posterior
# --------------------------------------------------------------------------- #
class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        # KL( N(mean, var) || N(0, I) ), summed over latent dims, mean over batch
        kl = 0.5 * (self.mean.pow(2) + self.var - 1.0 - self.logvar)
        return kl.sum(dim=[1, 2, 3]).mean()


# --------------------------------------------------------------------------- #
# Top-level AutoencoderKL
# --------------------------------------------------------------------------- #
@dataclass
class VAEConfig:
    in_channels: int = 5
    out_channels: int = 5
    z_channels: int = 8
    resolution: int = 96
    ch: int = 128
    ch_mult: Tuple[int, ...] = (1, 2, 4, 4)
    enc_num_res_blocks: int = 2
    dec_num_res_blocks: int = 3
    attn_resolutions: Tuple[int, ...] = (12,)
    dropout: float = 0.0


class AutoencoderKL(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(
            in_channels=cfg.in_channels,
            z_channels=cfg.z_channels,
            ch=cfg.ch,
            ch_mult=cfg.ch_mult,
            num_res_blocks=cfg.enc_num_res_blocks,
            attn_resolutions=cfg.attn_resolutions,
            dropout=cfg.dropout,
            double_z=True,
            resolution=cfg.resolution,
        )
        self.decoder = Decoder(
            out_channels=cfg.out_channels,
            z_channels=cfg.z_channels,
            ch=cfg.ch,
            ch_mult=cfg.ch_mult,
            num_res_blocks=cfg.dec_num_res_blocks,
            attn_resolutions=cfg.attn_resolutions,
            dropout=cfg.dropout,
            resolution=cfg.resolution,
        )
        self.quant_conv = nn.Conv2d(2 * cfg.z_channels, 2 * cfg.z_channels, 1)
        self.post_quant_conv = nn.Conv2d(cfg.z_channels, cfg.z_channels, 1)

    # ---- pieces ---- #
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor, sample_posterior: bool = True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        x_rec = self.decode(z)
        return x_rec, posterior

    @property
    def last_layer(self) -> nn.Module:
        return self.decoder.conv_out  # used for adaptive d-weight