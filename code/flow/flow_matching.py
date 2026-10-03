"""Rectified flow matching on latents.

  * path "cond_ot":  z_t = (1-t)z0 + t z1,                    v = z1 - z0
  * path "noisy":    z_t = (1-t)z0 + t z1 + sin^2(pi t) eps,  v = z1 - z0 + pi sin(2 pi t) eps
  * source-noise injection with probability p_noise_source
  * classifier-free guidance dropout with probability p_cond_drop
  * stage 1 maps Gaussian noise -> z1, stage 2 maps control -> perturbed
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FlowConfig:
    sigma_noise: float = 1.0        # std of interpolant noise eps ("noisy" path)
    p_noise_source: float = 0.5     # prob of adding noise to source z0 (stage 2)
    p_cond_drop: float = 0.2        # CFG dropout prob
    # True = one Bernoulli for the whole batch (CellFlux's convention);
    # False = per sample
    p_cond_drop_per_batch: bool = False
    source_noise_std: float = 1.0   # std of the source-noise injection
    # "cond_ot": the straight conditional-OT interpolant (CellFlux's
    # CondOTProbPath), no interpolant noise. "noisy": adds sin^2(pi t) eps.
    path: str = "cond_ot"
    # EDM timestep sampling (CellFlux --skewed_timesteps, on for bbbc021)
    skewed_timesteps: bool = True


def cond_drop_mask(cfg: "FlowConfig", B: int, device) -> torch.Tensor:
    """``[B]`` bool: which samples see the null condition this step.

    Per sample by default; ``p_cond_drop_per_batch`` drops the condition for
    the entire batch at once, as CellFlux does.
    """
    if getattr(cfg, "p_cond_drop_per_batch", False):
        return (torch.rand((), device=device) < cfg.p_cond_drop).expand(B)
    return torch.rand(B, device=device) < cfg.p_cond_drop


def skewed_timestep_sample(n: int, device) -> torch.Tensor:
    """EDM's skewed t schedule, as CellFlux training/train_loop.py."""
    P_mean, P_std = -1.2, 1.2
    sigma = (torch.randn((n,), device=device) * P_std + P_mean).exp()
    return torch.clip(1 / (1 + sigma), min=0.0001, max=1.0)


def get_time_discretization(nfes: int, rho: float = 7.0) -> torch.Tensor:
    """EDM sampling grid, as CellFlux training/edm_time_discretization.py."""
    i = torch.arange(nfes, dtype=torch.float64)
    s_min, s_max = 0.002, 80.0
    sig = (s_max ** (1 / rho) + i / (nfes - 1)
           * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho
    sig = torch.cat([sig, torch.zeros_like(sig[:1])])
    return 1.0 - torch.clip((sig / (1 + sig)).squeeze(), min=0.0, max=1.0)


class RectifiedFlowBag:
    def __init__(self, cfg: FlowConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    def training_loss(
        self,
        model,                 # DiTVelocity
        z0: torch.Tensor,      # control latent   [B, C, H, W]
        z1: torch.Tensor,      # perturbed latent [B, C, H, W]
        c: torch.Tensor,       # compound emb     [B, D_text]
        stage: int,            # 1 = noise->target, 2 = control->perturbed
    ):
        B = z1.shape[0]
        device = z1.device
        cfg = self.cfg

        # ---- source distribution ----
        if stage == 1:
            z0 = torch.randn_like(z1)                     # Gaussian -> target
        else:
            if cfg.p_noise_source > 0:
                add = (torch.rand(B, device=device) < cfg.p_noise_source).float()
                add = add.view(B, 1, 1, 1)
                z0 = z0 + add * cfg.source_noise_std * torch.randn_like(z0)

        # ---- noisy rectified interpolant ----
        t = torch.rand(B, device=device)                  # U[0,1]
        tt = t.view(B, 1, 1, 1)
        eps = cfg.sigma_noise * torch.randn_like(z1)
        s = torch.sin(math.pi * tt)
        z_t = (1 - tt) * z0 + tt * z1 + (s ** 2) * eps
        v_true = (z1 - z0) + math.pi * torch.sin(2 * math.pi * tt) * eps

        # ---- CFG dropout ----
        drop_mask = cond_drop_mask(cfg, B, device)

        # ---- predict + loss ----
        v_pred = model(z_t, t, c, drop_mask=drop_mask)
        loss = F.mse_loss(v_pred, v_true)
        return loss, {"flow/loss": loss.detach(),
                      "flow/disp_norm": (z1 - z0).flatten(1).norm(dim=1).mean().detach()}

    # ------------------------------------------------------------------ #
    def _interpolant(self, z0, z1, stage: int):
        """Interpolant + true velocity: (z_t, v_true, t, drop_mask).

        ``cfg.path == "cond_ot"``: ``z_t = (1-t)z0 + t z1``, ``v = z1 - z0``
        (CellFlux's ``CondOTProbPath``). ``"noisy"`` adds interpolant noise;
        its ``v_true`` is the exact time derivative of its ``z_t``.
        """
        B, device, cfg = z1.shape[0], z1.device, self.cfg
        if stage == 1:
            z0 = torch.randn_like(z1)
        elif cfg.p_noise_source > 0:
            add = (torch.rand(B, device=device) < cfg.p_noise_source).float()
            z0 = z0 + add.view(B, 1, 1, 1) * cfg.source_noise_std * torch.randn_like(z0)

        t = (skewed_timestep_sample(B, device) if cfg.skewed_timesteps
             else torch.rand(B, device=device))
        tt = t.view(B, 1, 1, 1)
        if cfg.path == "cond_ot":
            z_t = (1 - tt) * z0 + tt * z1
            v_true = z1 - z0
        elif cfg.path == "noisy":
            eps = cfg.sigma_noise * torch.randn_like(z1)
            z_t = (1 - tt) * z0 + tt * z1 + (torch.sin(math.pi * tt) ** 2) * eps
            v_true = (z1 - z0) + math.pi * torch.sin(2 * math.pi * tt) * eps
        else:
            raise ValueError(f"flow.path must be cond_ot or noisy, got {cfg.path!r}")
        drop_mask = cond_drop_mask(cfg, B, device)
        return z_t, v_true, t, drop_mask

    def training_loss_per_sample(self, model, z0, z1, c, stage: int):
        """Flow-matching loss per batch element, **not** reduced: ``[B]``.

        The caller owns the reduction, which is where the per-crop responder
        weights enter.
        """
        z_t, v_true, t, drop_mask = self._interpolant(z0, z1, stage)
        v_pred = model(z_t, t, c, drop_mask=drop_mask)
        loss = ((v_pred - v_true) ** 2).flatten(1).mean(dim=1)
        return loss, {"flow/disp_norm": (z1 - z0).flatten(1).norm(dim=1).mean().detach()}

    # ------------------------------------------------------------------ #
    def training_loss_detailed(self, model, z0, z1, c, stage: int):
        """Same as training_loss but also returns the pieces needed to build
        an image-space endpoint loss:

            loss     : latent flow-matching MSE
            t        : [B] sampled times
            z_t      : [B,C,H,W] noisy interpolant
            v_pred   : [B,C,H,W] predicted velocity
            z1_hat   : [B,C,H,W] one-step endpoint estimate
                       z1_hat = z_t + (1 - t) * v_pred
            z1       : [B,C,H,W] the (real) target latent used
        """
        B = z1.shape[0]
        device = z1.device
        cfg = self.cfg

        if stage == 1:
            z0 = torch.randn_like(z1)
        else:
            if cfg.p_noise_source > 0:
                add = (torch.rand(B, device=device) < cfg.p_noise_source).float()
                add = add.view(B, 1, 1, 1)
                z0 = z0 + add * cfg.source_noise_std * torch.randn_like(z0)

        t = torch.rand(B, device=device)
        tt = t.view(B, 1, 1, 1)
        eps = cfg.sigma_noise * torch.randn_like(z1)
        s = torch.sin(math.pi * tt)
        z_t = (1 - tt) * z0 + tt * z1 + (s ** 2) * eps
        v_true = (z1 - z0) + math.pi * torch.sin(2 * math.pi * tt) * eps

        drop_mask = cond_drop_mask(cfg, B, device)
        v_pred = model(z_t, t, c, drop_mask=drop_mask)
        loss = F.mse_loss(v_pred, v_true)

        # one-step endpoint estimate (exact for a straight path, good approx
        # for the noisy interpolant, and most reliable at larger t)
        z1_hat = z_t + (1 - tt) * v_pred

        info = {
            "flow/loss": loss.detach(),
            "flow/disp_norm": (z1 - z0).flatten(1).norm(dim=1).mean().detach(),
        }
        return loss, t, z1_hat, z1, info

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        model,
        z0: torch.Tensor,      # source latent [B, C, H, W]
        c: torch.Tensor,       # compound emb  [B, D_text]
        num_steps: int = 50,
        cfg_scale: float = 1.0,
        method: str = "heun2",
        edm_schedule: bool = True,
    ) -> torch.Tensor:
        """Integrate dz/dt = v(z,t,c) from t=0 to t=1.

        Defaults match CellFlux's BBBC021 script: ``heun2`` over the EDM time
        grid at nfe=50. ``method="euler"`` and ``edm_schedule=False`` give
        plain uniform-step Euler.
        """
        if edm_schedule:
            grid = get_time_discretization(num_steps).to(z0.device).float()
        else:
            grid = torch.linspace(0.0, 1.0, num_steps + 1, device=z0.device)
        z = z0.clone()
        for i in range(len(grid) - 1):
            t0, t1 = grid[i], grid[i + 1]
            dt = (t1 - t0)
            tv = torch.full((z.shape[0],), float(t0), device=z.device)
            v0 = model.forward_with_cfg(z, tv, c, cfg_scale=cfg_scale)
            if method == "euler":
                z = z + dt * v0
            elif method == "heun2":
                z_e = z + dt * v0
                tv1 = torch.full((z.shape[0],), float(t1), device=z.device)
                v1 = model.forward_with_cfg(z_e, tv1, c, cfg_scale=cfg_scale)
                z = z + dt * 0.5 * (v0 + v1)
            else:
                raise ValueError(f"method must be euler or heun2, got {method!r}")
        return z

    @torch.no_grad()
    def sample_trajectory(self, model, z0, c, num_steps=50, cfg_scale=1.0):
        """Same as `sample` but returns all intermediate states (for the
        cell-state-interpolation capability)."""
        z = z0.clone()
        dt = 1.0 / num_steps
        traj = [z.clone()]
        for i in range(num_steps):
            t = torch.full((z.shape[0],), i * dt, device=z.device)
            v = model.forward_with_cfg(z, t, c, cfg_scale=cfg_scale)
            z = z + dt * v
            traj.append(z.clone())
        return torch.stack(traj, dim=1)   # [B, num_steps+1, C, H, W]
