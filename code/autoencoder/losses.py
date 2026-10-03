"""Losses for VAE training: L1 + LPIPS + KL + hinge PatchGAN with adaptive weighting.

Discriminator is a *shallow* PatchGAN tuned for 96x96 inputs (RF ~22 px) so it
evaluates patches rather than the whole image.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips as _lpips  # pip install lpips
except Exception:  # pragma: no cover
    _lpips = None


# --------------------------------------------------------------------------- #
# PatchGAN discriminator (shallow, for 96x96)
# --------------------------------------------------------------------------- #
class NLayerDiscriminator(nn.Module):
    """Pix2pix PatchGAN. Default depth=2 gives a receptive field ~22 px."""

    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 2,
                 use_actnorm: bool = False):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        kw, padw = 4, 1

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]
        self.main = nn.Sequential(*sequence)

        # weight init (DCGAN)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.main(x)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = F.relu(1.0 - logits_real).mean()
    loss_fake = F.relu(1.0 + logits_fake).mean()
    return 0.5 * (loss_real + loss_fake)


def hinge_g_loss(logits_fake):
    return -logits_fake.mean()


# --------------------------------------------------------------------------- #
# LPIPS for arbitrary channel counts
# --------------------------------------------------------------------------- #
class MultiChannelLPIPS(nn.Module):
    """Computes LPIPS for inputs with arbitrary channel counts by replicating
    each channel as a 3-channel grayscale image and averaging the result.

    Inputs are expected in [-1, 1] which is what LPIPS expects.
    """

    def __init__(self, net: str = "vgg"):
        super().__init__()
        if _lpips is None:
            raise ImportError("Install lpips: `pip install lpips`")
        self.lpips = _lpips.LPIPS(net=net)
        for p in self.lpips.parameters():
            p.requires_grad_(False)
        self.lpips.eval()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: B x C x H x W in [-1, 1]
        C = x.shape[1]
        losses = []
        for c in range(C):
            xc = x[:, c:c + 1].expand(-1, 3, -1, -1)
            yc = y[:, c:c + 1].expand(-1, 3, -1, -1)
            losses.append(self.lpips(xc, yc))
        return torch.stack(losses, dim=0).mean()


# --------------------------------------------------------------------------- #
# Combined loss
# --------------------------------------------------------------------------- #
def adopt_weight(weight: float, global_step: int, threshold: int) -> float:
    return weight if global_step >= threshold else 0.0


class LPIPSWithDiscriminator(nn.Module):
    """Wraps reconstruction (L1) + LPIPS + KL + adversarial loss with an
    adaptively weighted discriminator term.
    """

    def __init__(
        self,
        disc_in_channels: int,
        disc_num_layers: int = 2,
        disc_ndf: int = 64,
        disc_weight: float = 0.5,
        disc_start: int = 50_000,
        kl_weight: float = 1e-6,
        perceptual_weight: float = 1.0,
        pixel_weight: float = 1.0,
        lpips_net: str = "vgg",
    ):
        super().__init__()
        self.disc_start = disc_start
        self.disc_weight = disc_weight
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.pixel_weight = pixel_weight

        self.perceptual_loss = MultiChannelLPIPS(net=lpips_net) if perceptual_weight > 0 else None
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels, ndf=disc_ndf, n_layers=disc_num_layers
        )

    # ------------------------------------------------------------------ #
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer.weight, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer.weight, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.disc_weight

    # ------------------------------------------------------------------ #
    def forward(
        self,
        inputs: torch.Tensor,         # ground truth, in [-1, 1]
        reconstructions: torch.Tensor,
        posterior,                    # DiagonalGaussianDistribution
        optimizer_idx: int,           # 0 = generator/VAE, 1 = discriminator
        global_step: int,
        last_layer: Optional[nn.Module] = None,
    ):
        rec = reconstructions.contiguous()
        gt = inputs.contiguous()

        # L1 + LPIPS
        rec_l1 = torch.abs(gt - rec).mean()
        if self.perceptual_loss is not None and self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(gt, rec)
        else:
            p_loss = torch.tensor(0.0, device=gt.device)

        nll_loss = self.pixel_weight * rec_l1 + self.perceptual_weight * p_loss
        kl_loss = posterior.kl()

        if optimizer_idx == 0:
            # ---- generator update ----
            logits_fake = self.discriminator(rec)
            g_loss = hinge_g_loss(logits_fake)

            if global_step >= self.disc_start and last_layer is not None:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=gt.device)
            else:
                d_weight = torch.tensor(0.0, device=gt.device)

            disc_factor = adopt_weight(1.0, global_step, self.disc_start)
            loss = nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {
                "loss/total":      loss.detach(),
                "loss/nll":        nll_loss.detach(),
                "loss/rec_l1":     rec_l1.detach(),
                "loss/lpips":      p_loss.detach() if torch.is_tensor(p_loss) else torch.tensor(0.0),
                "loss/kl":         kl_loss.detach(),
                "loss/g":          g_loss.detach(),
                "loss/d_weight":   d_weight.detach() if torch.is_tensor(d_weight) else torch.tensor(d_weight),
                "loss/disc_factor": torch.tensor(disc_factor),
            }
            return loss, log

        # ---- discriminator update ----
        logits_real = self.discriminator(gt.detach())
        logits_fake = self.discriminator(rec.detach())
        disc_factor = adopt_weight(1.0, global_step, self.disc_start)
        d_loss = disc_factor * hinge_d_loss(logits_real, logits_fake)

        log = {
            "loss/disc":         d_loss.detach(),
            "loss/logits_real":  logits_real.detach().mean(),
            "loss/logits_fake":  logits_fake.detach().mean(),
        }
        return d_loss, log