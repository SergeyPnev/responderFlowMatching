"""Crops -> CellProfiler-ready 16-bit TIFFs + a LoadData csv.

Used by ``eval_flow.py --dump_images`` (real / reconstructed / generated /
control populations, written from inside the generation loop).

1. One fixed intensity scale, never per-image: ``uint16 = round(clip(x, 0, 1)
   * 65535)``. A per-image rescale would normalise away the intensity
   differences being measured.
2. One csv for every population (``Metadata_population`` separates them), so
   all go through one CellProfiler run with one pipeline.
3. Channel index -> name is a config (see ``CHANNEL_CONFLICT``); the pipeline
   refers to images by name.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

CHANNEL_CONFLICT = """\
BBBC021 crops are stored (F-actin, beta-tubulin, DNA): index 2 is the nuclei.
Checked on raw crops and by cp_features.py channels.
`channel_names` is what CellProfiler sees, so every per-channel
result inherits whatever is passed there."""

# stored channel order
DEFAULT_CHANNEL_NAMES = ("Actin", "Tubulin", "DNA")
POPULATIONS = ("real", "recon", "gen", "control")
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_name(s: str) -> str:
    """Filesystem-safe stem. The true id always travels in Metadata_crop_id."""
    return _SAFE.sub("-", str(s))


def to_uint16(x: np.ndarray) -> np.ndarray:
    """[0,1] float -> uint16 on the fixed global scale. See rule 1."""
    return np.rint(np.clip(x, 0.0, 1.0) * 65535.0).astype(np.uint16)


def _writer():
    try:
        import tifffile
        return lambda p, a: tifffile.imwrite(p, a, photometric="minisblack")
    except ImportError:                                           # pragma: no cover
        from PIL import Image
        return lambda p, a: Image.fromarray(a, mode="I;16").save(p)


class CropWriter:
    """Accumulates TIFFs and LoadData rows across batches and populations.

    ``add`` takes one batch as ``[B, C, H, W]`` in [0, 1] plus the crop ids and
    per-crop metadata for those rows, and is safe to call once per population
    per batch from inside a generation loop.
    """

    def __init__(self, out_dir: str, channel_names: Sequence[str],
                 arm: str = "", overwrite: bool = False):
        self.out = os.path.abspath(out_dir)
        self.names = list(channel_names)
        self.arm = arm
        self.overwrite = overwrite
        self.rows: List[Dict] = []
        self._seen: set = set()
        self._imwrite = _writer()
        os.makedirs(self.out, exist_ok=True)

    # ------------------------------------------------------------------ #
    def _dir(self, population: str) -> str:
        # gen depends on the arm; real / recon / control are written once and
        # shared (shared_manifest() guards that they match across arms)
        sub = f"gen__{self.arm}" if population == "gen" and self.arm else population
        d = os.path.join(self.out, sub)
        os.makedirs(d, exist_ok=True)
        return d

    def add(self, imgs: np.ndarray, crop_ids: Sequence[str], population: str,
            meta: Optional[pd.DataFrame] = None) -> int:
        """Write one batch. Returns the number of crops actually written."""
        if population not in POPULATIONS:
            raise ValueError(f"population must be one of {POPULATIONS}")
        imgs = np.asarray(imgs)
        if imgs.ndim != 4:
            raise ValueError(f"expected [B,C,H,W], got {imgs.shape}")
        if imgs.shape[1] != len(self.names):
            raise ValueError(f"{imgs.shape[1]} channels but channel_names has "
                             f"{len(self.names)}: {self.names}")
        if len(crop_ids) != len(imgs):
            raise ValueError(f"{len(crop_ids)} crop_ids for {len(imgs)} images")

        d = self._dir(population)
        written = 0
        for i, cid in enumerate(crop_ids):
            key = (population, str(cid))
            if key in self._seen:
                continue                       # a crop can pair twice; keep one
            self._seen.add(key)
            stem = safe_name(cid)
            row = {"Metadata_crop_id": str(cid),
                   "Metadata_population": population,
                   "Metadata_arm": self.arm or "-"}
            for c, nm in enumerate(self.names):
                fn = f"{stem}__{nm}.tiff"
                p = os.path.join(d, fn)
                if self.overwrite or not os.path.exists(p):
                    self._imwrite(p, to_uint16(imgs[i, c]))
                row[f"Image_FileName_{nm}"] = fn
                row[f"Image_PathName_{nm}"] = d
            if meta is not None:
                for k, v in meta.iloc[i].items():
                    row[f"Metadata_{k}"] = v
            self.rows.append(row)
            written += 1
        return written

    # ------------------------------------------------------------------ #
    def load_data(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        first = [c for nm in self.names
                 for c in (f"Image_FileName_{nm}", f"Image_PathName_{nm}")]
        rest = [c for c in df.columns if c not in first]
        return df[first + rest]

    def write(self, csv_name: str = "load_data.csv",
              provenance: Optional[Dict] = None) -> str:
        """Append to any existing csv for this dir, then dedupe."""
        path = os.path.join(self.out, csv_name)
        df = self.load_data()
        if os.path.exists(path):
            old = pd.read_csv(path)
            df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates(
                subset=["Metadata_crop_id", "Metadata_population",
                        "Metadata_arm"], keep="last")
        # real / recon / control: one row per crop, whichever arm wrote it
        shared = (df["Metadata_population"] != "gen").to_numpy()
        df = pd.concat([df[shared].drop_duplicates(
            ["Metadata_crop_id", "Metadata_population"], keep="first"),
            df[~shared]], ignore_index=True)
        df.to_csv(path, index=False)
        if provenance:
            with open(os.path.join(self.out, "dump_provenance.json"), "w") as f:
                json.dump(provenance, f, indent=2, default=str)
        n = df.groupby("Metadata_population").size().to_dict()
        print(f"[cp_io] {path}: {len(df)} rows  {n}")
        print(f"[cp_io] channel names {self.names} -- see CHANNEL_CONFLICT "
              f"before reading any per-channel number")
        return path


def shared_manifest(out_dir: str, crop_ids: Sequence[str], vae_ckpt: str,
                    seed: int, channel_names: Sequence[str]) -> None:
    """Guard the populations that must be identical across arms.

    ``real`` / ``control`` / ``recon`` are written once and reused by every
    arm, so the eval index, pairing seed, VAE and channel map must match the
    manifest on disk; a mismatch is fatal.
    """
    m = {"n_crops": len(crop_ids),
         "crop_id_sha": hashlib.sha256(
             "|".join(map(str, crop_ids)).encode()).hexdigest()[:16],
         "vae_ckpt": os.path.basename(str(vae_ckpt)),
         "vae_sha": _file_sha(vae_ckpt), "seed": int(seed),
         "channel_names": list(channel_names)}
    p = os.path.join(out_dir, "shared_manifest.json")
    if os.path.exists(p):
        old = json.load(open(p))
        diff = {k: (old.get(k), m[k]) for k in m if old.get(k) != m[k]}
        if diff:
            raise SystemExit(
                f"{p} was written by a different reference and the shared "
                f"real/recon/control images in {out_dir} do not belong to this "
                f"run:\n  " +
                "\n  ".join(f"{k}: on disk {a!r} != now {b!r}"
                            for k, (a, b) in diff.items()) +
                f"\nUse a fresh --dump_images dir, or delete {out_dir} if the "
                f"old dump is dead.")
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(p, "w") as f:
        json.dump(m, f, indent=2)


def _file_sha(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()[:16]
    except OSError:
        return "<unreadable>"
