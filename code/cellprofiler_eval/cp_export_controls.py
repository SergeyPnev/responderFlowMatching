"""Every QC-kept DMSO crop on the dump's plates -> the dump, as `control`.

    python cellprofiler_eval/cp_export_controls.py --dump $D \\
        --flow_index $P4/flow_index_bbbc021.parquet \\
        --crop_universe $P4/cache/crop_universe.parquet

eval_flow dumps only the controls it paired as generation sources; for
measurement any DMSO crop on the plate is a valid reference. Pixels go through
the eval loader's transform, so a control already in the dump is the same file
and is listed once. Writing also rewrites load_data.csv with one row per
real / recon / control crop. Rerun cp_measure.sh (FORCE=1) afterwards.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dump", required=True, help="an existing --dump_images dir")
    p.add_argument("--flow_index", required=True,
                   help="the MAIN flow index (all splits' DMSO)")
    p.add_argument("--crop_universe", required=True,
                   help="the scorer's QC: only kept crops are exported")
    p.add_argument("--max_per_plate", type=int, default=0, help="0 = all")
    p.add_argument("--channels", default="Actin,Tubulin,DNA",
                   help="STORED order, as the dump's shared_manifest")
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    from cellprofiler_eval.cp_io import CropWriter
    from datasets.bbbc021_dataset import default_bbbc_transform

    L = pd.read_csv(os.path.join(a.dump, "load_data.csv"))
    plates = set(L.loc[L.Metadata_population == "real", "Metadata_plate"].astype(str))
    fi = pd.read_parquet(a.flow_index)
    cu = pd.read_parquet(a.crop_universe)
    kept = set(cu.loc[cu["kept"].astype(bool), "crop_id"].astype(str))
    c = fi[(fi["y"] == 0) & fi["plate"].astype(str).isin(plates)
           & fi["crop_id"].astype(str).isin(kept)].copy()
    if a.max_per_plate:
        rng = np.random.default_rng(a.seed)
        c = c.groupby("plate", group_keys=False).apply(
            lambda d: d.iloc[np.sort(rng.choice(len(d), min(len(d), a.max_per_plate),
                                                replace=False))])
    had = L[L.Metadata_population == "control"].Metadata_crop_id.astype(str).nunique()
    print(f"[controls] {len(plates)} plates with real crops; dump holds {had} "
          f"distinct control crop(s), exporting {len(c)} (median "
          f"{c.groupby('plate').size().median():.0f} per plate)")

    chans = a.channels.split(",")
    tf = default_bbbc_transform(a.img_size, len(chans))
    w = CropWriter(a.dump, chans, arm="ctlpool")
    meta = pd.DataFrame({
        "unit_id": "control", "compound": "DMSO", "dose": np.nan,
        "moa": c.get("moa", pd.Series(np.nan, index=c.index)).to_numpy(),
        "plate": c["plate"].to_numpy(), "well": c["well"].to_numpy(),
        "well_id": np.nan, "s": np.nan, "y": 0})      # as eval_flow's dump_meta
    for i0 in range(0, len(c), 256):
        sl = c.iloc[i0:i0 + 256]
        imgs = np.stack([tf(image=np.load(pth))["image"].numpy()
                         for pth in sl["path"]])
        w.add(np.clip(imgs, 0, 1), sl["crop_id"].astype(str).tolist(), "control",
              meta.iloc[i0:i0 + 256].reset_index(drop=True))
    w.write()
    L2 = pd.read_csv(os.path.join(a.dump, "load_data.csv"))
    print(f"[controls] load_data.csv: {len(L)} -> {len(L2)} rows  "
          f"{L2.Metadata_population.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
