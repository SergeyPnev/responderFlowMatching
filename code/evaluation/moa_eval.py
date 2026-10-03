"""MoA accuracy of generated crops, following CellFlux's protocol.

    # once: fit the classifier head on REAL training crops (GPU, minutes)
    python evaluation/moa_eval.py --config configs/cellflux_percrop_bbbc_fp.yaml
    # every arm: eval_flow scores its generated crops with that head
    python evaluation/eval_flow.py --config configs/cellflux_percrop_bbbc_fp.yaml \\
           --arms gamma0,gamma1 --moa_head $OUT/moa/moa_head.pt
    # or set eval.moa_head in the config: every arm is scored when it finishes

Features: the frozen Inception-v3 inside torchmetrics' FID. Head:
Linear(2048, 512) - ReLU - Dropout(0.5) - Linear(512, n_moa), Adam 1e-3,
10 epochs, fit on real treated crops of the training split. Real eval crops
and VAE reconstructions are scored with the same head as reference rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POPULATIONS = ("gen", "real", "recon")

# CellFlux/moa/checkpoint.pth's label space (sorted, DMSO included: a treated
# crop the head calls DMSO is scored wrong).
CELLFLUX_CLASSES = ("Actin disruptors", "Aurora kinase inhibitors",
                    "Cholesterol-lowering", "DMSO", "DNA damage",
                    "DNA replication", "Eg5 inhibitors", "Epithelial",
                    "Kinase inhibitors", "Microtubule destabilizers",
                    "Microtubule stabilizers", "Protein degradation",
                    "Protein synthesis")
# CellFlux/moa/checkpoint_ood.pth's label space: the OOD compounds + DMSO.
CELLFLUX_OOD_CLASSES = ("Actin disruptors", "Aurora kinase inhibitors",
                        "Cholesterol-lowering", "DMSO",
                        "Microtubule stabilizers", "Protein degradation",
                        "Protein synthesis")


def inception_features(device, weights=None):
    """[B,3,H,W] in [0,1] -> [B,2048]: the FID network, fed the way FID feeds it.

    ``weights`` is an Inception state dict to use instead of torchmetrics'
    download (CellFlux's classifier checkpoint carries its own copy)."""
    os.environ.setdefault("USE_TF", "0")
    from torchmetrics.image.fid import FrechetInceptionDistance
    inc = FrechetInceptionDistance(normalize=True).inception
    if weights is not None:
        # same module, so every tensor has a counterpart
        inc.load_state_dict(weights, strict=True)
    inc = inc.to(device).eval()

    @torch.no_grad()
    def f(x: torch.Tensor) -> torch.Tensor:
        return inc((x.clamp(0, 1) * 255).byte()).float()
    return f


def make_head(n_classes: int, dim: int = 2048) -> nn.Module:
    return nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Dropout(0.5),
                         nn.Linear(512, n_classes))


def cellflux_head(model_state: Dict, device):
    """CellFlux/moa/checkpoint.pth's ``model_state`` -> (features, head,
    classes). Their MOAClassifier is ``feature_extractor.inception`` (FID's
    Inception) + ``classifier`` (make_head's layout), so both load strictly."""
    pre, hp = "feature_extractor.inception.", "classifier."
    n = int(model_state[hp + "3.weight"].shape[0])
    classes = {len(CELLFLUX_CLASSES): CELLFLUX_CLASSES,
               len(CELLFLUX_OOD_CLASSES): CELLFLUX_OOD_CLASSES}.get(n)
    if classes is None:
        raise SystemExit(f"a {n}-way CellFlux MoA head: neither checkpoint.pth "
                         f"(13) nor checkpoint_ood.pth (7)")
    head = make_head(n)
    head.load_state_dict({k[len(hp):]: v for k, v in model_state.items()
                          if k.startswith(hp)})
    feat = inception_features(device, {k[len(pre):]: v
                                       for k, v in model_state.items()
                                       if k.startswith(pre)})
    return feat, head.to(device).eval(), list(classes)


def fit_head(X: np.ndarray, y: np.ndarray, n_classes: int, epochs: int = 10,
             lr: float = 1e-3, batch: int = 256, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    head = make_head(n_classes, X.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch):
            b = perm[i:i + batch]
            loss = F.cross_entropy(head(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head.eval()


@torch.no_grad()
def predict(head: nn.Module, X) -> np.ndarray:
    return head(torch.as_tensor(X, dtype=torch.float32)).argmax(1).numpy()


def summarise(y: np.ndarray, pred: np.ndarray, pop: str,
              prefix: str = "moa") -> Dict[str, float]:
    """Accuracy and the two F1s CellFlux reports, under one population's name."""
    from sklearn.metrics import f1_score
    return {f"{prefix}_acc_{pop}": float((y == pred).mean()),
            f"{prefix}_f1_macro_{pop}": float(f1_score(y, pred, average="macro")),
            f"{prefix}_f1_weighted_{pop}": float(f1_score(y, pred,
                                                          average="weighted"))}


class _Crops(Dataset):
    """Real crops, loaded as CropFlowDataset._load does and clamped as
    eval_flow clamps them -- the head must see what the metric sees."""

    def __init__(self, paths, tf):
        self.paths, self.tf = list(paths), tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.tf(image=np.load(self.paths[i]))["image"].float().clamp(0, 1)


def extract(paths, feat, tf, device, batch_size: int,
            num_workers: int) -> np.ndarray:
    dl = DataLoader(_Crops(paths, tf), batch_size=batch_size,
                    num_workers=num_workers)
    return np.concatenate([feat(x.to(device)).cpu().numpy() for x in dl])


def fit(cfg: Dict, out: str, epochs: int = 10, batch_size: int = 256,
        num_workers: int = 8, limit: int = 0) -> Dict:
    from datasets.bbbc021_dataset import default_bbbc_transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev_split = cfg.get("eval_split", "val")
    idx = pd.read_parquet(cfg["flow_index"])
    t = idx[(idx["y"] == 1) & idx["moa"].notna()]
    tr = t[t["split"].isin(cfg.get("train_splits", ["train"]))]
    ev = t[t["split"] == ev_split]
    if limit:
        tr = tr.sample(min(limit, len(tr)), random_state=0)
        ev = ev.head(limit)
    classes = sorted(tr["moa"].unique())
    unseen = sorted(set(ev["moa"]) - set(classes))
    if unseen:
        raise SystemExit(f"{ev_split} has MoA classes absent from training: "
                         f"{unseen}")
    print(f"[moa] {len(tr)} train / {len(ev)} {ev_split} real treated crops, "
          f"{len(classes)} classes, device {device}")

    tf = default_bbbc_transform(cfg["img_size"], cfg.get("n_channels", 3))
    feat = inception_features(device)
    Xtr = extract(tr["path"], feat, tf, device, batch_size, num_workers)
    Xev = extract(ev["path"], feat, tf, device, batch_size, num_workers)
    lab = {c: i for i, c in enumerate(classes)}
    ytr = tr["moa"].map(lab).to_numpy()
    yev = ev["moa"].map(lab).to_numpy()
    head = fit_head(Xtr, ytr, len(classes), epochs=epochs)

    pev = predict(head, Xev)
    info = {"classes": classes, "dim": int(Xtr.shape[1]),
            "n_train": len(tr), "n_eval": len(ev), "epochs": epochs,
            "flow_index": cfg["flow_index"],
            "train_splits": list(cfg.get("train_splits", ["train"])),
            "eval_split": ev_split,
            **summarise(ytr, predict(head, Xtr), "real_train"),
            **summarise(yev, pev, "real_eval")}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save({"head": head.state_dict(), **info}, out)
    json.dump(info, open(os.path.splitext(out)[0] + ".json", "w"), indent=2)

    print(f"\n  real {ev_split} accuracy per class -- the ceiling a generated "
          f"arm is read against")
    for k, c in enumerate(classes):
        m = yev == k
        if m.any():
            print(f"    {c:28s} {int(m.sum()):6d}  {np.mean(pev[m] == k):.3f}")
    print(f"  overall: train {info['moa_acc_real_train']:.3f}, {ev_split} "
          f"{info['moa_acc_real_eval']:.3f}  (macro-F1 "
          f"{info['moa_f1_macro_real_eval']:.3f})\n-> {out}")
    return info


class MoAScorer:
    """Scores eval_flow's batches as they are generated: one frozen head for
    every population, so gen / real / recon differ only in the images.

    ``path`` is either this module's head or CellFlux/moa/checkpoint.pth as
    shipped; the latter reports under ``moa_cf_*`` and writes
    ``moa_preds_cf*.parquet``. ``subset`` (crop ids) adds every metric on
    those crops as ``*_sub``.
    """

    def __init__(self, path: str, device, subset=None):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.path = path
        if "model_state" in ck:
            self.feat, self.head, self.classes = cellflux_head(ck["model_state"],
                                                               device)
            # checkpoint.pth -> moa_cf_*, checkpoint_ood.pth -> moa_cfood_*
            self.tag = ("_cf" if len(self.classes) == len(CELLFLUX_CLASSES)
                        else "_cfood")
        else:
            self.classes = list(ck["classes"])
            self.head = make_head(len(self.classes), ck["dim"])
            self.head.load_state_dict(ck["head"])
            self.head.to(device).eval()
            self.feat = inception_features(device)
            self.tag = ""
        self.prefix = "moa" + self.tag
        self.subset = None if subset is None else set(map(str, subset))
        self.pred: Dict[str, List[np.ndarray]] = {p: [] for p in POPULATIONS}

    @torch.no_grad()
    def add(self, pop: str, imgs: torch.Tensor) -> None:
        self.pred[pop].append(self.head(self.feat(imgs)).argmax(1).cpu().numpy())

    def finish(self, trt: pd.DataFrame, out_dir, name: str = None) -> Dict:
        """Metrics for the row, and every prediction to <arm>/<name> so
        accuracy can be re-cut by dose, compound or s without regenerating."""
        lab = trt["moa"].map({c: i for i, c in enumerate(self.classes)})
        if lab.isna().any():
            bad = sorted(trt.loc[lab.isna(), "moa"].astype(str).unique())
            raise SystemExit(f"eval crops with a MoA the head was not trained "
                             f"on: {bad}")
        y = lab.to_numpy(int)
        keep = [c for c in ("crop_id", "unit_id", "compound", "dose", "moa", "s")
                if c in trt.columns]
        preds = trt[keep].copy()
        row = {f"{self.prefix}_head": self.path,
               f"{self.prefix}_n_classes": len(self.classes)}
        sub = (None if self.subset is None else
               trt["crop_id"].astype(str).isin(self.subset).to_numpy())
        if sub is not None:
            row[f"{self.prefix}_n_sub"] = int(sub.sum())
        for p in POPULATIONS:
            pr = np.concatenate(self.pred[p])
            assert len(pr) == len(y), f"{p}: {len(pr)} predictions, {len(y)} crops"
            row.update(summarise(y, pr, p, self.prefix))
            if sub is not None:
                row.update(summarise(y[sub], pr[sub], f"{p}_sub", self.prefix))
            preds[f"pred_{p}"] = np.asarray(self.classes, dtype=object)[pr]
        preds.to_parquet(Path(out_dir) / (name or f"moa_preds{self.tag}.parquet"),
                         index=False)
        return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None,
                   help="default <out_dir>/moa/moa_head.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0,
                   help="crops per split, for a smoke run only")
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = a.out or str(Path(cfg["out_dir"]) / "moa" / "moa_head.pt")
    fit(cfg, out, epochs=a.epochs, batch_size=a.batch_size,
        num_workers=a.num_workers, limit=a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
