"""CellFlux's own MoA classifier, applied to our arms and checked on theirs.

    # once: the 5,120 crops CellFlux's Table 2a scores, in their loader's order
    python evaluation/cellflux_moa.py subset \\
        --index <data>/bbbc021_df_all.csv \\
        --config <CellFlux>/configs/bbbc021_all.yaml \\
        --out cellflux_moa_subset.csv
    # the check: their head on their released images
    python evaluation/cellflux_moa.py pngs --head <CellFlux>/moa/checkpoint.pth \\
        --png_root <snapshot>/images/cellflux/bbbc021 \\
        --subset cellflux_moa_subset.csv
    # our arms: eval_flow with their head
    python evaluation/eval_flow.py --config <cfg> --arms <arms> --no_inception \\
        --moa_head <CellFlux>/moa/checkpoint.pth \\
        --moa_subset cellflux_moa_subset.csv --out <csv>

CellFlux's generated-image evaluation walks the IID test loader unshuffled and
stops at 5,120 crops, so the subset is the first 5,120 IID test crops in
bbbc021_df_all.csv order. The head is 13-way (DMSO included).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import moa_eval  # noqa: E402

TABLE_2A = {"acc": 0.711, "f1_macro": 0.490, "f1_weighted": 0.707}
N_EVAL = 5120  # train_moa.py: `if total >= 5120: break`, batch 32


def eval_order(index_csv: str, ood_set) -> pd.DataFrame:
    """IID test treated crops in the order CellFlux's test loader yields them:
    csv order, OOD compounds dropped, SPLIT == test, treated only. Their loader
    reads STATE == 1; the copy in this repo spells it 'trt'."""
    d = pd.read_csv(index_csv, index_col=0)
    d = d[~d["CPD_NAME"].isin(ood_set or [])]   # eval_bbbc_ood.yaml: Null
    t = d[(d["SPLIT"] == "test") & d["STATE"].astype(str).isin(["trt", "1"])]
    return t.reset_index(drop=True)


def cmd_subset(a) -> None:
    ood = yaml.safe_load(open(a.config))["ood_set"]
    t = eval_order(a.index, ood).head(a.n)
    out = pd.DataFrame({"crop_id": t["SAMPLE_KEY"].astype(str),
                        "idx": np.arange(len(t)), "compound": t["CPD_NAME"],
                        "moa": t["ANNOT"]})
    out.to_csv(a.out, index=False)
    print(f"{len(out)} crops, {out.compound.nunique()} compounds, "
          f"{out.moa.nunique()} MoA classes -> {a.out}")


def read_png(path: str) -> torch.Tensor:
    """train_moa.read_img_from_path: 8-bit RGB as [3,H,W] uint8 (then / 255)."""
    from PIL import Image
    img = np.array(Image.open(path).convert("RGB"))
    return torch.from_numpy(img).permute(2, 0, 1)


def gen_files(index_csv: str, ood_set, png_root: str) -> pd.DataFrame:
    """CellFlux's IID test crops, each with the PNG their eval loop writes for
    it: <png_root>/<CPD_NAME>/<SAMPLE_KEY>.png, so mevinolin/lovastatin nests
    a directory."""
    t = eval_order(index_csv, ood_set)
    t["path"] = [os.path.join(png_root, c, f"{k}.png")
                 for c, k in zip(t["CPD_NAME"], t["SAMPLE_KEY"])]
    return t


def _crop_ids(path: str) -> set:
    d = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    return set(d["crop_id"].astype(str))


def cmd_gen(a) -> None:
    """Every head in --heads on cellflux_generate_all.sh's PNGs, on four crop
    sets: all IID test crops, our eval crops (--eval_crops), the Table 2a
    subset, and their overlap."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ood = yaml.safe_load(open(a.config))["ood_set"]
    t = gen_files(a.index, ood, a.png_root)
    miss = ~t["path"].map(os.path.exists)
    if miss.any():
        raise SystemExit(f"{int(miss.sum())} of {len(t)} crops have no PNG, "
                         f"e.g. {t.loc[miss, 'path'].iloc[0]}")
    key = t["SAMPLE_KEY"].astype(str)
    sets = {"all": np.ones(len(t), bool)}
    if a.eval_crops:
        sets["ours"] = key.isin(_crop_ids(a.eval_crops)).to_numpy()
    if a.subset:
        sets["sub"] = key.isin(_crop_ids(a.subset)).to_numpy()
        if a.eval_crops:
            sets["sub_ours"] = sets["sub"] & sets["ours"]
    preds = pd.DataFrame({"crop_id": key, "compound": t["CPD_NAME"],
                          "moa": t["ANNOT"]})
    rows = []
    for h in a.heads.split(","):
        sc = moa_eval.MoAScorer(h, device)
        y = t["ANNOT"].map({c: i for i, c in enumerate(sc.classes)})
        if y.isna().any():
            raise SystemExit(f"{h}: MoA classes it was not trained on: "
                             f"{sorted(t.loc[y.isna(), 'ANNOT'].unique())}")
        y = y.to_numpy(int)
        pred = []
        for i in range(0, len(t), a.batch_size):
            x = torch.stack([read_png(p) for p in
                             t["path"].iloc[i:i + a.batch_size]]).float() / 255.0
            with torch.no_grad():
                pred.append(sc.head(sc.feat(x.to(device))).argmax(1).cpu().numpy())
        pred = np.concatenate(pred)
        preds[f"pred{sc.tag}"] = np.asarray(sc.classes, dtype=object)[pred]
        print(f"\nhead {h} ({len(sc.classes)} classes)")
        for name, m in sets.items():
            r = moa_eval.summarise(y[m], pred[m], "gen")
            rows.append({"head": sc.prefix, "set": name, "n": int(m.sum()),
                         **{k: r[f"moa_{k}_gen"] for k in TABLE_2A}})
            print(f"  {name:9s} n={int(m.sum()):5d}  acc {r['moa_acc_gen']:.4f}"
                  f"  macro-F1 {r['moa_f1_macro_gen']:.4f}"
                  f"  weighted-F1 {r['moa_f1_weighted_gen']:.4f}")
        m = sets.get("ours", sets["all"])
        print("  per class, on", "ours" if "ours" in sets else "all")
        for k, c in enumerate(sc.classes):
            mk = m & (y == k)
            if mk.any():
                print(f"    {c:28s} {int(mk.sum()):5d}  {np.mean(pred[mk] == k):.3f}")
    print(f"\nTable 2a CellFlux (their 5,120 = set 'sub'): "
          + " / ".join(f"{v:.3f}" for v in TABLE_2A.values()))
    pd.DataFrame(rows).to_csv(a.out, index=False)
    preds.to_parquet(os.path.splitext(a.out)[0] + "_preds.parquet", index=False)
    print(f"-> {a.out}")


def cmd_pngs(a) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sub = pd.read_csv(a.subset)
    ck = torch.load(a.head, map_location="cpu", weights_only=False)
    feat, head, classes = moa_eval.cellflux_head(ck["model_state"], device)
    lab = {c: i for i, c in enumerate(classes)}

    def load(r):
        return read_png(os.path.join(a.png_root, r.compound, f"{r.idx}.png"))

    pred = []
    for i in range(0, len(sub), a.batch_size):
        x = torch.stack([load(r) for r in sub.iloc[i:i + a.batch_size]
                         .itertuples()]).float() / 255.0
        with torch.no_grad():
            pred.append(head(feat(x.to(device))).argmax(1).cpu().numpy())
    pred = np.concatenate(pred)
    y = sub["moa"].map(lab).to_numpy()
    r = moa_eval.summarise(y, pred, "gen")
    print(f"{len(y)} released CellFlux images, head {a.head}")
    for k, ref in TABLE_2A.items():
        print(f"  {k:12s} {r[f'moa_{k}_gen']:.4f}   Table 2a {ref:.3f}")
    print("  per class:")
    for k, c in enumerate(classes):
        m = y == k
        if m.any():
            print(f"    {c:28s} {int(m.sum()):5d}  {np.mean(pred[m] == k):.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("subset", help="write the 5,120-crop Table 2a subset")
    s.add_argument("--index", required=True, help="bbbc021_df_all.csv")
    s.add_argument("--config", required=True,
                   help="CellFlux/configs/bbbc021_all.yaml (for ood_set)")
    s.add_argument("--n", type=int, default=N_EVAL)
    s.add_argument("--out", default=os.path.join(HERE, "cellflux_moa_subset.csv"))
    g = sp.add_parser("pngs", help="their head on their released images")
    g.add_argument("--head", required=True, help="CellFlux/moa/checkpoint.pth")
    g.add_argument("--png_root", required=True,
                   help="dir holding <compound>/<index>.png")
    g.add_argument("--subset", default=os.path.join(HERE, "cellflux_moa_subset.csv"))
    g.add_argument("--batch_size", type=int, default=64)
    q = sp.add_parser("gen", help="any head on cellflux_generate_all.sh's "
                                  "PNGs (<compound>/<SAMPLE_KEY>.png)")
    q.add_argument("--heads", required=True,
                   help="comma list: our moa_head.pt and/or their "
                        "CellFlux/moa/checkpoint.pth")
    q.add_argument("--png_root", required=True, help="<GEN>/fid_samples/epoch-N")
    q.add_argument("--index", required=True,
                   help="bbbc021_df_all.csv, either STATE spelling")
    q.add_argument("--config", required=True,
                   help="CellFlux/configs/bbbc021_all.yaml (for ood_set)")
    q.add_argument("--eval_crops", default=None,
                   help="our eval crop_ids, e.g. $OUT/gamma0/moa_preds.parquet")
    q.add_argument("--subset", default=os.path.join(HERE, "cellflux_moa_subset.csv"))
    q.add_argument("--out", required=True, help="csv; per-crop preds go next "
                                                "to it as *_preds.parquet")
    q.add_argument("--batch_size", type=int, default=64)
    a = p.parse_args()
    {"subset": cmd_subset, "pngs": cmd_pngs, "gen": cmd_gen}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
