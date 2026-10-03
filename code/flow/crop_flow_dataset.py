"""Per-crop control -> treated pairs, and the responder weights.

One dataset item is one treated crop, paired with a control crop drawn from the
same plate. Invariants:

1. Control crops get weight 1: they are the source, the tilt applies to the
   target only.
2. Weights are normalised within unit, ``w_i = s_i**gamma / mean_unit(s**gamma)``,
   so every unit keeps the same total gradient mass.
3. gamma=0 gives exactly 1.0 (short-circuited and asserted).
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
def responder_weights(s: np.ndarray, unit_id: np.ndarray, gamma: float,
                      plate: Optional[np.ndarray] = None) -> np.ndarray:
    """``w_i = s_i**gamma / mean_cell(s**gamma)``, mean 1 within every cell.

    The cell is the unit by default. Pass ``plate`` to normalise within
    ``(unit, plate)`` instead, which keeps every plate's share of gradient
    mass (controls are drawn within plate, so per-unit normalisation also
    reweights the source pool across plates).
    """
    if gamma == 0.0:
        return np.ones(len(s), dtype=np.float64)
    if not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError("s must be finite and > 0; cache_scores.py clips it "
                         "to [1e-3, 1-1e-3] for exactly this reason")
    p = np.power(s.astype(np.float64), gamma)
    cell = (pd.Series(unit_id) if plate is None
            else pd.Series(unit_id).astype(str) + "|" + pd.Series(plate).astype(str))
    den = pd.Series(p).groupby(cell).transform("mean").to_numpy()
    return p / den


def load_training_index(index_path: str, scores_path: str,
                        train_splits: Sequence[str] = ("train",),
                        control_splits: Sequence[str] = ("train", "test"),
                        require_all_scored: bool = True,
                        warn_val: bool = True,
                        crop_universe: Optional[str] = None,
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Join the flow index to the score cache. Returns (treated, control, info).

    A treated crop with no score is either dropped with a logged count or
    fatal, never given a default weight.

    ``train_splits`` selects the treated crops (the target); it must exclude
    ``val`` (the FID/KID reference) and ``test`` (the OOD compounds).
    ``control_splits`` selects the control source pool and is wider, because
    the split partitions compounds, not controls, and some train plates have
    no control in the train split. ``val`` controls stay out.
    """
    idx = pd.read_parquet(index_path)
    sc = pd.read_parquet(scores_path)
    if "split" not in idx.columns:
        raise SystemExit(
            f"{index_path} has no `split` column. It was built before "
            f"flow_index.py carried one, and a split-blind index trains on the "
            f"FID/KID val set and on the OOD test compounds. Rebuild it.")
    # crops the scorer's QC dropped have no score; exclude them here too
    if crop_universe:
        cu = pd.read_parquet(crop_universe)
        keep = set(cu.loc[cu["kept"].astype(bool), "crop_id"].astype(str))
        before = len(idx)
        idx = idx[idx["crop_id"].astype(str).isin(keep)]
        print(f"[index] QC: {before - len(idx)} crop(s) dropped by the scorer's "
              f"QC are excluded from training too ({len(idx)} remain)")
    ts, cs = set(train_splits), set(control_splits)
    unknown = (ts | cs) - set(idx["split"].unique())
    if unknown:
        raise SystemExit(f"no such split(s) in the index: {sorted(unknown)}; "
                         f"it has {sorted(idx['split'].unique())}")
    if ts & {"val"} and warn_val:
        print("[index] ! train_splits includes 'val', which is the FID/KID "
              "reference. Any metric computed on it afterwards is measured on "
              "training data.")
    if sc["crop_id"].duplicated().any():
        raise ValueError(f"{int(sc['crop_id'].duplicated().sum())} duplicate "
                         f"crop_id in {scores_path}")

    n_all_t = int((idx["y"] == 1).sum())
    # h, the scorer's posterior, rides along for eval; training reads only s
    cols = ["crop_id", "s", "unit_id"] + (["h"] if "h" in sc.columns else [])
    trt = idx[(idx["y"] == 1) & idx["split"].isin(ts)].merge(
        sc[cols], on="crop_id", how="left", validate="1:1")
    unscored = trt["s"].isna()
    info = {"train_splits": sorted(ts), "control_splits": sorted(cs),
            "n_treated_universe": n_all_t, "n_treated_index": len(trt),
            "n_unscored": int(unscored.sum())}
    if unscored.any():
        by = trt[unscored].groupby("compound").size().sort_values(ascending=False)
        msg = (f"{int(unscored.sum())} of {len(trt)} treated crop(s) have no "
               f"cached score, in {len(by)} unit(s): "
               + ", ".join(f"{u} ({n})" for u, n in by.head(5).items()))
        if require_all_scored:
            raise SystemExit(
                msg + "\nEither re-run the scorer over these units or pass "
                "--allow_unscored to drop them. Do not train on them with a "
                "default weight: at gamma>0 that silently treats them as "
                "average responders and tilts only part of the target.")
        print(f"[index] dropping {msg}")
        trt = trt[~unscored]
    trt = trt.reset_index(drop=True)

    ctl = idx[(idx["y"] == 0) & idx["split"].isin(cs)].reset_index(drop=True)
    have = set(ctl["plate"])
    keep = trt["plate"].isin(have)
    info["n_unpairable"] = int((~keep).sum())
    if not keep.all():
        print(f"[index] dropping {(~keep).sum()} treated crop(s) on plates "
              f"with no control: {sorted(set(trt.loc[~keep, 'plate']))[:5]}")
        trt = trt[keep].reset_index(drop=True)
    info["n_treated"] = len(trt)
    info["n_control"] = len(ctl)
    info["n_units"] = int(trt["unit_id"].nunique())
    info["n_compounds"] = int(trt["compound"].nunique())
    print(f"[index] treated {len(trt)}/{n_all_t} from splits {sorted(ts)} "
          f"({info['n_units']} units, {info['n_compounds']} compounds); "
          f"controls {len(ctl)} from splits {sorted(cs)} on "
          f"{ctl['plate'].nunique()} plates")
    return trt, ctl, info


def latent_keys(df: pd.DataFrame) -> pd.Series:
    """Which id addresses the latent cache: ``latent_key`` if the index carries
    one, else ``crop_id``.

    Dilution sets inject one control crop into several units, so their rows have
    a synthetic ``crop_id`` and the original id lives in ``latent_key``.
    """
    col = "latent_key" if "latent_key" in df.columns else "crop_id"
    return df[col].astype(str)


def load_latents(latent_dir: str, crop_ids) -> Tuple[np.ndarray, Dict[str, int], Dict]:
    """Memory-map the precomputed latents and bind them to crop ids.

    The cache is addressed by ``crop_id``, never by row order. A crop the
    cache does not hold is fatal.
    """
    import json

    meta = json.load(open(os.path.join(latent_dir, "latents_meta.json")))
    li = pd.read_parquet(os.path.join(latent_dir, "latents_index.parquet"))
    z = np.load(os.path.join(latent_dir, "latents.npy"), mmap_mode="r")
    if len(z) != len(li):
        raise SystemExit(f"latents.npy has {len(z)} rows, latents_index has "
                         f"{len(li)}")
    row_of = dict(zip(li["crop_id"].astype(str), li["row"].astype(int)))
    missing = [c for c in map(str, crop_ids) if c not in row_of]
    if missing:
        raise SystemExit(
            f"{len(missing)} crop(s) in the training index have no precomputed "
            f"latent, e.g. {missing[:3]}. The latent cache was built from a "
            f"different flow index -- rebuild it with the current one.")
    print(f"[latents] {len(z)} x {tuple(z.shape[1:])} {meta['dtype']} from "
          f"{latent_dir}  (vae {meta['vae_ckpt_sha']}, "
          f"measured std {meta['measured_std']:.4f})")
    return z, row_of, meta


def assign_cond_idx(df: pd.DataFrame, cpd_row: Dict[str, int]) -> pd.DataFrame:
    """``text_idx`` rewritten from ``compound``, for a compound-indexed table.

    Fatal on a compound the table does not carry.
    """
    miss = sorted(set(df["compound"].astype(str)) - set(cpd_row))
    if miss:
        raise SystemExit(f"{len(miss)} compound(s) are not in the conditioning "
                         f"table: {', '.join(miss[:8])}"
                         + (" ..." if len(miss) > 8 else ""))
    out = df.copy()
    out["text_idx"] = [cpd_row[str(c)] for c in out["compound"]]
    return out


def load_cond_table(path: str, cond_dim: int
                    ) -> Tuple[torch.Tensor, Optional[Dict[str, int]]]:
    """The conditioning table as a strict ``[N, cond_dim]`` matrix.

    Two formats, chosen by extension:

    ``.pt``   CellCLIP prompt embeddings, one row per prompt, addressed by the
              flow index's ``text_idx``. Second return value is ``None``.
    ``.csv``  a compound-indexed table (e.g. IMPA's Morgan fingerprints
              ``emb_fp.csv``). The second return value maps compound -> row
              and the caller rewrites ``text_idx`` with ``assign_cond_idx``.

    ``cond_dim`` is ``dit.text_dim``, the width of whatever is conditioned on.
    """
    if str(path).endswith(".csv"):
        tab = pd.read_csv(path, index_col=0)
        emb = torch.as_tensor(tab.to_numpy(), dtype=torch.float32)
        if emb.shape[1] != cond_dim:
            raise SystemExit(f"{path}: {emb.shape[1]} columns but the config "
                             f"says dit.text_dim={cond_dim}")
        rows = {str(c): i for i, c in enumerate(tab.index)}
        if len(rows) != len(tab):
            raise SystemExit(f"{path}: duplicate compound(s) in the index")
        print(f"[cond] fingerprint table {tuple(emb.shape)} from {path}, "
              f"{len(rows)} compounds")
        return emb.contiguous(), rows
    return _load_text_table(path, cond_dim), None


def _load_text_table(path: str, text_dim: int) -> torch.Tensor:
    """The CellCLIP prompt embeddings as a strict ``[N, text_dim]`` matrix.

    The stored table is ``[N, 1, D]``; the singleton axis is dropped here,
    because a ``[B, 1, D]`` condition silently broadcasts against the timestep
    embedding inside the DiT.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    emb = blob["emb"] if isinstance(blob, dict) else blob
    emb = torch.as_tensor(emb).float()
    if emb.dim() == 3 and emb.shape[1] == 1:
        emb = emb[:, 0]
    if emb.dim() != 2:
        raise SystemExit(f"{path}: expected a 2-D [N, D] embedding table, got "
                         f"{tuple(emb.shape)}")
    if emb.shape[1] != text_dim:
        raise SystemExit(f"{path}: embedding dim {emb.shape[1]} but the config "
                         f"says dit.text_dim={text_dim}")
    print(f"[text] table {tuple(emb.shape)}")
    return emb.contiguous()


# --------------------------------------------------------------------------- #
class CropFlowDataset(Dataset):
    """One treated crop + one same-plate control crop + its weight.

    ``__getitem__`` returns ``(ctrl, trt, text_idx, w)``. The control partner is
    redrawn every epoch from a generator seeded by ``(seed, epoch, i)``, so the
    pairing is reproducible and does not depend on the weights: runs at
    different gamma see the same pairs in the same order.
    """

    def __init__(self, treated: pd.DataFrame, control: pd.DataFrame,
                 gamma: float, transform=None, seed: int = 0,
                 text_table: Optional[torch.Tensor] = None,
                 latents: Optional[np.ndarray] = None,
                 latent_row: Optional[Dict[str, int]] = None,
                 scaling_factor: float = 1.0,
                 weight_norm: str = "unit"):
        self.t = treated.reset_index(drop=True)
        self.c = control.reset_index(drop=True)
        self.transform = transform
        # with a latent cache the VAE never runs: the encode is deterministic
        # (posterior mode), so the cached tensor equals the encoder's output
        self.latents = latents
        self.latent_row = latent_row
        self.scaling_factor = float(scaling_factor)
        self.seed = seed
        self.epoch = 0
        self.text_table = text_table
        self.gamma = float(gamma)

        if weight_norm not in ("unit", "unit_plate"):
            raise SystemExit(f"weight_norm must be 'unit' or 'unit_plate', "
                             f"got {weight_norm!r}")
        self.weight_norm = weight_norm
        self.w = responder_weights(
            self.t["s"].to_numpy(float), self.t["unit_id"].to_numpy(), self.gamma,
            plate=self.t["plate"].to_numpy() if weight_norm == "unit_plate" else None)
        if self.gamma == 0.0 and not np.array_equal(self.w, np.ones(len(self.w))):
            raise AssertionError("gamma=0 did not give exactly unit weights")
        # positional row ids of the control crops on each plate
        self.ctrl_by_plate = {p: np.asarray(i) for p, i
                              in self.c.groupby("plate").indices.items()}
        self.t_plate = self.t["plate"].to_numpy()
        self.t_path = self.t["path"].to_numpy()
        self.c_path = self.c["path"].to_numpy()
        self.t_text = (self.t["text_idx"].to_numpy()
                       if "text_idx" in self.t.columns else None)
        if self.latents is not None:
            self.t_row = np.array([self.latent_row[str(c)]
                                   for c in latent_keys(self.t)], dtype=np.int64)
            self.c_row = np.array([self.latent_row[str(c)]
                                   for c in latent_keys(self.c)], dtype=np.int64)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.t)

    def _load(self, path: str) -> torch.Tensor:
        img = np.load(path)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img.float()

    def _latent(self, row: int) -> torch.Tensor:
        z = torch.from_numpy(np.asarray(self.latents[row], dtype=np.float32))
        return z * self.scaling_factor if self.scaling_factor != 1.0 else z

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        pool = self.ctrl_by_plate[self.t_plate[i]]
        j = int(pool[rng.integers(len(pool))])
        if self.latents is not None:
            ctrl = self._latent(int(self.c_row[j]))
            trt = self._latent(int(self.t_row[i]))
        else:
            ctrl = self._load(self.c_path[j])
            trt = self._load(self.t_path[i])
        text = (int(self.t_text[i]) if self.t_text is not None else 0)
        return ctrl, trt, text, float(self.w[i])

    # -------------------------------------------------------------- #
    def weight_report(self) -> Dict[str, float]:
        """Summary statistics of the weights (per unit, per plate, ESS)."""
        w = self.w
        per_unit = pd.Series(w).groupby(pd.Series(self.t["unit_id"].to_numpy())).mean()
        # controls are drawn within plate, so a plate whose mean weight moves
        # away from 1 has its controls reweighted too
        per_plate = pd.Series(w).groupby(pd.Series(self.t["plate"].to_numpy())).mean()
        return {
            "gamma": self.gamma, "weight_norm": self.weight_norm,
            "plate_w_min": float(per_plate.min()),
            "plate_w_max": float(per_plate.max()),
            "plate_mass_spread": float(per_plate.max() / per_plate.min()),
            "w_mean": float(w.mean()), "w_min": float(w.min()),
            "w_max": float(w.max()), "w_p99": float(np.quantile(w, 0.99)),
            # effective sample size if the whole training set were one batch:
            # the ceiling the per-batch ESS is heading towards
            "ess_frac_dataset": float(w.sum() ** 2 / (w ** 2).sum() / len(w)),
            "unit_mean_max_dev": float(np.abs(per_unit - 1.0).max()),
            "n_units": int(len(per_unit)), "n_crops": int(len(w)),
        }


def collate(batch):
    ctrl = torch.stack([b[0] for b in batch], 0)
    trt = torch.stack([b[1] for b in batch], 0)
    text = torch.tensor([b[2] for b in batch], dtype=torch.long)
    w = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    return ctrl, trt, text, w


def resolve_scaling_factor(cfg: Dict) -> float:
    """Resolve ``vae.scaling_factor: auto`` against the latent cache's own metadata.

    ``auto`` reads the value precompute_latents.py measured, so the number
    cannot drift from the cache it belongs to.
    """
    import json

    sf = cfg.get("vae", {}).get("scaling_factor", 1.0)
    if not isinstance(sf, str):
        return float(sf)
    if sf != "auto":
        raise SystemExit(f"vae.scaling_factor must be a number or 'auto', got {sf!r}")
    if not cfg.get("latents"):
        raise SystemExit("vae.scaling_factor: auto needs `latents:` to read the "
                         "measured value from; set the number explicitly when "
                         "encoding on the fly.")
    meta = json.load(open(os.path.join(cfg["latents"], "latents_meta.json")))
    val = float(meta["suggested_scaling_factor"])
    cfg["vae"]["scaling_factor"] = val
    print(f"[vae] scaling_factor auto -> {val:.4f} "
          f"(1/std over the cache in {cfg['latents']})")
    return val
