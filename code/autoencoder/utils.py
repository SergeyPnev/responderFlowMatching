"""EMA + small utilities."""
from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn as nn


class ModelEMA:
    """Exponential moving average of model weights.

    Keep a shadow copy of params; update every step. Use the EMA model for
    validation / final checkpoint.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.ema.eval()

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


def load_ckpt(path: str):
    """Load a ``train.py`` checkpoint (``weights_only=True`` where supported)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_minus_one_one(x: torch.Tensor, in_range: str = "01") -> torch.Tensor:
    """Map images to [-1, 1] for LPIPS / GAN."""
    if in_range == "01":
        return x.mul(2.0).sub(1.0)
    if in_range == "-11":
        return x
    raise ValueError(in_range)


def to_zero_one(x: torch.Tensor) -> torch.Tensor:
    return x.add(1.0).mul(0.5).clamp(0.0, 1.0)


def make_grid_multi_channel(x: torch.Tensor, max_imgs: int = 8) -> torch.Tensor:
    """Stack a B x C x H x W tensor into a viewable grid: rows = images,
    cols = channels.  Returns 1 x H' x W' grayscale grid in [0, 1].
    """
    from torchvision.utils import make_grid

    x = x[:max_imgs].detach().cpu().clamp(0, 1)
    B, C, H, W = x.shape
    tiles = x.permute(0, 1, 2, 3).reshape(B * C, 1, H, W)  # B*C x 1 x H x W
    return make_grid(tiles, nrow=C, padding=2, pad_value=0.0)


def rgb_scale(x: torch.Tensor, rgb_channels=(0, 1, 2), percentile: float = 99.5):
    """Per-RGB-channel (lo, hi) display range, from one reference batch.

    Returned so the same scale can be applied to every panel of a comparison.
    """
    x = x.detach().float().cpu()
    lo, hi = [], []
    for c in rgb_channels:
        v = x[:, c] if c < x.shape[1] else torch.zeros_like(x[:, 0])
        lo.append(float(torch.quantile(v, 1 - percentile / 100.0)))
        hi.append(float(torch.quantile(v, percentile / 100.0)))
    return torch.tensor(lo), torch.tensor(hi)


def make_grid_rgb(x: torch.Tensor, rgb_channels=(0, 1, 2), max_imgs: int = 8,
                  scale=None, nrow: int | None = None,
                  percentile: float = 99.5) -> torch.Tensor:
    """B x C x H x W -> one 3 x H' x W' RGB grid, channels composited.

    ``rgb_channels`` picks which source channel goes to R, G, B; the default
    ``[0, 1, 2]`` stacks them in stored order.
    """
    from torchvision.utils import make_grid

    x = x[:max_imgs].detach().float().cpu()
    idx = [c if c < x.shape[1] else 0 for c in rgb_channels]
    img = x[:, idx]                                       # B x 3 x H x W
    lo, hi = scale if scale is not None else rgb_scale(x, rgb_channels, percentile)
    img = (img - lo.view(1, 3, 1, 1)) / (hi - lo).clamp(min=1e-6).view(1, 3, 1, 1)
    return make_grid(img.clamp(0, 1), nrow=nrow or len(img), padding=2,
                     pad_value=0.0)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)