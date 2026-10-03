"""Which ``epoch_XXXX.pt`` of a VAE run sits at a given training step, and freeze it.

Checkpoints are named by epoch but a VAE is picked by step, and steps/epoch
moves with the fold size, so ``global_step`` is read out of every file. Files
are memory-mapped, so only the pickle header is read, not the weights.

    python autoencoder/ckpt_at_step.py --ckpt_dir $AE/cpg/ckpt --step 500000
    python autoencoder/ckpt_at_step.py --ckpt_dir $AE/cpg/ckpt --step 500000 \\
        --copy_to $CF/cpg/vae/vae_s500k.pt

``--copy_to`` makes the frozen copy the flow reads (``ckpt/`` is a live
training dir), refuses to overwrite, and writes ``<copy>.json`` next to it
recording the source file, epoch and step.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil

import torch


def read_step(path: str):
    """(global_step, epoch, epoch_done, cfg) without loading the tensors."""
    try:
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception:                                             # noqa: BLE001
        # pre-zipfile format, a torch without mmap, or a non-tensor object the
        # weights-only unpickler refuses: full load, still correct
        ck = torch.load(path, map_location="cpu", weights_only=False)
    return (int(ck["global_step"]), int(ck["epoch"]),
            bool(ck.get("epoch_done", True)), ck.get("cfg"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--copy_to", default=None)
    a = p.parse_args()

    files = sorted(glob.glob(os.path.join(a.ckpt_dir, "epoch_*.pt")),
                   key=lambda f: int(re.findall(r"(\d+)", os.path.basename(f))[-1]))
    if not files:
        raise SystemExit(f"no epoch_*.pt under {a.ckpt_dir}")
    rows = []
    for f in files:
        step, epoch, done, cfg = read_step(f)
        rows.append((abs(step - a.step), f, step, epoch, done, cfg))

    best = min(rows, key=lambda r: r[0])
    i = rows.index(best)
    steps = [r[2] for r in rows]
    # epochs may have been pruned from ckpt/, so divide by the epoch gap; with a
    # single file, epoch_*.pt is always end-of-epoch: step / epochs completed
    per_ep = [(b[2] - a_[2]) // (b[3] - a_[3])
              for a_, b in zip(rows, rows[1:]) if b[3] > a_[3]]
    if not per_ep:
        per_ep = [rows[0][2] // (rows[0][3] + 1)]
    print(f"{len(rows)} epoch checkpoint(s), steps {steps[0]}..{steps[-1]}, "
          f"{per_ep[0]} steps/epoch")
    for r in rows[max(0, i - 2): i + 3]:
        mark = "  <-- nearest" if r is best else ""
        print(f"  {os.path.basename(r[1])}  epoch {r[3]:>4}  step {r[2]:>8}  "
              f"({r[2] - a.step:+d}){mark}")
    if best[0] > per_ep[0]:
        why = ("the run may not have reached it yet" if a.step > steps[-1]
               else "the epochs around it are not on disk")
        print(f"! the nearest checkpoint is {best[0]} steps from {a.step}, more "
              f"than one epoch -- {why}")

    cfg = best[5] or {}
    if cfg:
        d = cfg.get("data", {}).get("args", {})
        print(f"\nstored cfg: epochs {cfg.get('epochs')}  batch {cfg.get('batch_size')}  "
              f"lr {cfg.get('lr')}  fold_path {d.get('fold_path')}")
        print("model block (paste into the flow config's vae.model):")
        for k, v in cfg.get("model", {}).items():
            print(f"    {k}: {v}")

    if a.copy_to:
        if os.path.exists(a.copy_to):
            raise SystemExit(f"{a.copy_to} exists; a frozen VAE is never "
                             f"overwritten -- every latent cache built from it "
                             f"would silently change meaning")
        os.makedirs(os.path.dirname(os.path.abspath(a.copy_to)), exist_ok=True)
        shutil.copy2(best[1], a.copy_to)
        side = {"source": os.path.abspath(best[1]), "epoch": best[3],
                "global_step": best[2], "target_step": a.step,
                "epoch_done": best[4]}
        json.dump(side, open(a.copy_to + ".json", "w"), indent=2)
        print(f"\n[frozen] {best[1]}\n      -> {a.copy_to}  (+ .json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
