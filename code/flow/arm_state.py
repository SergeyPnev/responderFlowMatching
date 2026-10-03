"""Is an arm finished, part-trained, or not started? (CPU, instant)

``ckpt/last.pt`` is written at the end of every epoch, so its existence means
"started", not "finished". The epoch is read from ``last.pt`` itself;
``ess.csv`` is only the fallback when the checkpoint cannot be read.

    python flow/arm_state.py --config configs/... --arm $OUT/gamma0

prints one of ``none`` / ``partial 3/1100`` / ``complete 1100/1100`` and exits
0 complete, 1 partial, 2 not started.
"""
from __future__ import annotations

import argparse
import csv
import os

import yaml
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def state(arm: str, total_epochs: int):
    """(code, text). code: 0 complete, 1 partial, 2 not started."""
    last = os.path.join(arm, "ckpt", "last.pt")
    if not os.path.exists(last):
        return 2, "none"
    done = -1
    try:
        from autoencoder.ckpt_at_step import read_step
        done = read_step(last)[1]           # the epoch last.pt was saved after
    except Exception:                                             # noqa: BLE001
        pass                                # mid-write or unreadable: the csv
    ess = os.path.join(arm, "ess.csv")
    if done < 0 and os.path.exists(ess):
        with open(ess) as f:
            for row in csv.DictReader(f):
                try:
                    done = max(done, int(float(row["epoch"])))
                except (KeyError, TypeError, ValueError):
                    pass
    # epochs are 0-based, so the last one is total-1
    if done >= total_epochs - 1:
        return 0, f"complete {done + 1}/{total_epochs}"
    if done < 0:
        # a checkpoint with no ess.csv: started, progress unknown
        return 1, f"partial ?/{total_epochs}"
    return 1, f"partial {done + 1}/{total_epochs}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--arm", required=True)
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    total = int(cfg.get("stage1_epochs", 0)) + int(cfg.get("stage2_epochs", 0))
    if total <= 0:
        raise SystemExit(f"{a.config}: stage1_epochs + stage2_epochs is {total}")
    code, text = state(a.arm, total)
    print(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
