"""Perturbation classification and replicate-detection mAP for generated crops
(for datasets without MoA labels: CPG, RxRx1).

    # once per dataset, before any arm is evaluated (GPU)
    python evaluation/pheno_eval.py fit --config configs/cellflux_percrop_rxrx1.yaml
    # every arm: eval_flow scores gen / real / recon inside the generation loop
    python evaluation/eval_flow.py --config configs/cellflux_percrop_rxrx1.yaml \\
        --source_noise --pheno_dir <out_dir>/pheno
    # or set eval.pheno_dir in the config: every arm is scored when it finishes

Features: FID's frozen Inception-v3 on each channel as a grayscale image,
concatenated -> C x 2048. Classification heads are fit on real treated training
crops leave-one-group-out (group = the flow index's ``plate``), reported as
crop-level top-1/5/10. Replicate detection: see pheno_metrics.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import pheno_metrics as pm                                         # noqa: E402
from evaluation.moa_eval import _Crops, inception_features, make_head         # noqa: E402

POPULATIONS = ("gen", "real", "recon")
KS = (1, 5, 10)


def channel_features(device):
    """[B,C,H,W] in [0,1] -> [B, C*2048]: every channel through FID's Inception
    as a grayscale image."""
    inc = inception_features(device)

    @torch.no_grad()
    def f(x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        g = x.reshape(B * C, 1, H, W).expand(-1, 3, -1, -1)
        return inc(g).reshape(B, C * 2048)
    return f


def fit_head(X: torch.Tensor, y: torch.Tensor, n_classes: int, device,
             epochs: int = 10, lr: float = 1e-3, batch: int = 256,
             seed: int = 0) -> torch.nn.Module:
    """moa_eval.fit_head's recipe, on ``device``; ``X`` may be float16."""
    torch.manual_seed(seed)
    head = make_head(n_classes, X.shape[1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch):
            b = perm[i:i + batch].to(X.device)
            loss = F.cross_entropy(head(X[b].to(device).float()),
                                   y.to(X.device)[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head.eval()


@torch.no_grad()
def logits_of(head, X, device, batch: int = 4096) -> np.ndarray:
    return np.concatenate([head(X[i:i + batch].to(device).float()).cpu().numpy()
                           for i in range(0, len(X), batch)])


def _qc_index(cfg: Dict) -> pd.DataFrame:
    idx = pd.read_parquet(cfg["flow_index"])
    if cfg.get("crop_universe"):
        cu = pd.read_parquet(cfg["crop_universe"])
        keep = set(cu.loc[cu["kept"].astype(bool), "crop_id"].astype(str))
        n = len(idx)
        idx = idx[idx["crop_id"].astype(str).isin(keep)]
        print(f"[pheno] QC: {n - len(idx)} crop(s) dropped by the scorer's QC")
    return idx.reset_index(drop=True)


def featurise(rows: pd.DataFrame, cfg: Dict, out: Path, device, batch_size: int,
              num_workers: int, force: bool) -> np.ndarray:
    """Every crop in ``rows``, once, to a float16 memmap keyed by crop_id."""
    from datasets.bbbc021_dataset import default_bbbc_transform
    fpath, ipath = out / "feats.npy", out / "feats_index.parquet"
    C = int(cfg["n_channels"])
    if fpath.exists() and ipath.exists() and not force:
        old = pd.read_parquet(ipath)
        if old["crop_id"].astype(str).tolist() == rows["crop_id"].astype(str).tolist():
            print(f"[pheno] reusing {fpath} ({len(old)} crops)")
            return np.load(fpath, mmap_mode="r")
        raise SystemExit(f"{fpath} holds a different crop set; --force to "
                         f"re-featurise")
    tf = default_bbbc_transform(cfg["img_size"], C)
    dl = DataLoader(_Crops(rows["path"], tf), batch_size=batch_size,
                    num_workers=num_workers)
    feat = channel_features(device)
    X = np.lib.format.open_memmap(fpath, mode="w+", dtype=np.float16,
                                  shape=(len(rows), C * 2048))
    n, t0 = 0, time.time()
    for bi, x in enumerate(dl):
        X[n:n + len(x)] = feat(x.to(device)).cpu().numpy().astype(np.float16)
        n += len(x)
        if bi % 100 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(rows)}  {el:.0f}s  eta "
                  f"{el / max(n, 1) * (len(rows) - n):.0f}s")
    X.flush()
    rows[["crop_id"]].to_parquet(ipath, index=False)
    return np.load(fpath, mmap_mode="r")


def fit(cfg: Dict, out: Path, epochs: int, batch_size: int, num_workers: int,
        force: bool, chunk: int = 8192) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out.mkdir(parents=True, exist_ok=True)
    idx = _qc_index(cfg)
    rows = idx[idx["y"].isin([0, 1])].sort_values("crop_id").reset_index(drop=True)
    rows["plate"] = rows["plate"].astype(str)
    X = featurise(rows, cfg, out, device, batch_size, num_workers, force)
    D = X.shape[1]
    trt, ctl = rows["y"].to_numpy() == 1, rows["y"].to_numpy() == 0

    # ---- replicate-mAP references ---------------------------------------- #
    grp = rows["plate"].to_numpy()
    stats = {}
    for g in np.unique(grp[ctl]):
        ii = np.where(ctl & (grp == g))[0]
        stats.update(pm.robust_stats(np.asarray(X[ii]), grp[ii]))
    miss = sorted(set(grp[trt]) - set(stats))
    if miss:
        raise SystemExit(f"treated crops on group(s) with no control crop: {miss}")

    def profiles(mask):
        acc = pm.WellMeans(D)
        ii = np.where(mask)[0]
        for s in range(0, len(ii), chunk):
            j = ii[s:s + chunk]
            acc.add(X[j], rows["well"].to_numpy()[j])
        wells, M, n = acc.result()
        sub = rows.loc[mask, ["well", "plate", "compound"]]
        if (sub.groupby("well")[["plate", "compound"]].nunique() > 1).any().any():
            raise SystemExit("a well spans more than one group or perturbation")
        meta = sub.drop_duplicates("well").set_index("well").loc[wells]
        Z = pm.normalise(M, meta["plate"], stats)
        return {"well": wells, "group": meta["plate"].to_numpy(), "n": n,
                "Z": Z.astype(np.float32)}

    print("[pheno] control well profiles (negatives) ...")
    np.savez(out / "ctrl_wells.npz", **profiles(ctl))
    gs = sorted(stats)
    np.savez(out / "ctrl_stats.npz", groups=np.asarray(gs),
             med=np.stack([stats[g][0] for g in gs]).astype(np.float32),
             scale=np.stack([stats[g][1] for g in gs]).astype(np.float32))

    # ---- leave-one-group-out heads --------------------------------------- #
    split = rows["split"].to_numpy()
    tr = trt & np.isin(split, cfg.get("train_splits", ["train"]))
    ev = trt & (split == cfg.get("eval_split", "val"))
    classes = sorted(rows.loc[tr, "compound"].astype(str).unique())
    lab = {c: i for i, c in enumerate(classes)}
    y_all = rows["compound"].astype(str).map(lab).fillna(-1).to_numpy(int)
    eval_groups = sorted(set(grp[ev]))
    print(f"[pheno] {len(classes)} classes; {int(tr.sum())} train / "
          f"{int(ev.sum())} eval real treated crops; one head per eval group: "
          f"{eval_groups}")
    # float16, on the GPU when there is one (~3 GB for either dataset)
    Xtr_all = torch.from_numpy(np.asarray(X[np.where(tr)[0]])).to(device)
    ytr_all = torch.from_numpy(y_all[tr]).to(device)
    grp_tr = grp[tr]
    heads, report = {}, {"classes": classes, "dim": D,
                         "n_channels": int(cfg["n_channels"]),
                         "epochs": epochs, "groups": {}}
    for g in eval_groups:
        m = torch.from_numpy(grp_tr != g).to(device)
        t0 = time.time()
        head = fit_head(Xtr_all[m], ytr_all[m], len(classes), device,
                        epochs=epochs)
        heads[g] = {k: v.cpu() for k, v in head.state_dict().items()}
        seen = set(y_all[tr][grp_tr != g])
        e = np.where(ev & (grp == g))[0]
        ye = y_all[e]
        ok = np.isin(ye, list(seen))
        r = pm.true_rank(logits_of(head, torch.from_numpy(np.asarray(X[e])),
                                   device), np.maximum(ye, 0))[ok]
        acc = pm.topk(r, KS)
        report["groups"][g] = {"n_train": int(m.sum()), "n_eval": int(len(e)),
                               "n_unlearnable": int((~ok).sum()),
                               **{f"real_top{k}": v for k, v in acc.items()}}
        print(f"  head {g}: {int(m.sum())} train crops, {time.time() - t0:.0f}s; "
              f"real eval top-1/5/10 "
              + "/".join(f"{acc[k]:.3f}" for k in KS)
              + f"  ({int((~ok).sum())} unlearnable of {len(e)})")
    torch.save({"heads": heads, **report}, out / "heads.pt")
    report.update({"flow_index": cfg["flow_index"],
                   "n_ctrl_wells": int(len(np.load(out / "ctrl_wells.npz")["well"]))})
    json.dump(report, open(out / "fit_report.json", "w"), indent=2)
    print(f"-> {out}")
    return report


# --------------------------------------------------------------------------- #
class PhenoScorer:
    """Scores eval_flow's batches as they are generated, like MoAScorer: the
    same heads and references for every population, so gen / real / recon
    differ only in the images."""

    def __init__(self, pheno_dir: str, trt: pd.DataFrame, device,
                 null_size: int = 100_000, threshold: float = 0.05):
        d = Path(pheno_dir)
        ck = torch.load(d / "heads.pt", map_location="cpu", weights_only=False)
        self.classes = list(ck["classes"])
        self.heads = {}
        for g, sd in ck["heads"].items():
            h = make_head(len(self.classes), ck["dim"])
            h.load_state_dict(sd)
            self.heads[str(g)] = h.to(device).eval()
        self.t = trt.reset_index(drop=True)
        self.group = self.t["plate"].astype(str).to_numpy()
        missing = sorted(set(self.group) - set(self.heads))
        if missing:
            raise SystemExit(f"no classifier head for eval group(s) {missing}: "
                             f"heads.pt was fit for {sorted(self.heads)}. Re-run "
                             f"`pheno_eval.py fit` with this config's eval_split.")
        lab = {c: i for i, c in enumerate(self.classes)}
        self.y = self.t["compound"].astype(str).map(lab).fillna(-1).to_numpy(int)
        self.feat = channel_features(device)
        self.device = device
        self.dir, self.null_size, self.threshold = d, null_size, threshold
        self.dim = int(ck["dim"])
        self.ranks = {p: np.full(len(self.t), -1, dtype=np.int64) for p in POPULATIONS}
        self.acc = {p: pm.WellMeans(self.dim) for p in POPULATIONS}
        self.checked = False
        self.path = str(d)

    def _check_real(self, F_real: np.ndarray, sl: slice) -> None:
        """The in-loop real features must equal the ones `fit` stored for the
        same crops, or the references were built from different images."""
        fi = pd.read_parquet(self.dir / "feats_index.parquet")
        row = dict(zip(fi["crop_id"].astype(str), range(len(fi))))
        ids = self.t["crop_id"].astype(str).to_numpy()[sl][:4]
        stored = np.load(self.dir / "feats.npy", mmap_mode="r")
        got = [np.asarray(stored[row[i]], dtype=np.float32) for i in ids if i in row]
        if len(got) != len(ids):
            raise SystemExit("eval crops missing from the pheno feature cache; "
                             "refit with this flow index")
        if not np.allclose(np.stack(got), F_real[:len(ids)], rtol=2e-2, atol=2e-3):
            raise SystemExit("in-loop real features differ from pheno_eval fit's "
                             "for the same crops: the two paths load images "
                             "differently")
        print(f"[pheno] real features match the fit cache on {len(ids)} crops")
        self.checked = True

    @torch.no_grad()
    def add(self, pop: str, imgs: torch.Tensor, sl: slice) -> None:
        Fx = self.feat(imgs)
        rows = np.arange(len(self.t))[sl]
        g = self.group[sl]
        for gg in np.unique(g):
            m = g == gg
            lg = self.heads[gg](Fx[torch.from_numpy(m).to(Fx.device)]).cpu().numpy()
            self.ranks[pop][rows[m]] = pm.true_rank(lg, np.maximum(self.y[rows[m]], 0))
        Fn = Fx.cpu().numpy()
        if pop == "real" and not self.checked:
            self._check_real(Fn, sl)
        self.acc[pop].add(Fn, self.t["well"].to_numpy()[sl])

    def finish(self, out_dir, name: str = "pheno") -> Dict:
        ok = self.y >= 0
        st = np.load(self.dir / "ctrl_stats.npz")
        stats = {str(g): (m.astype(np.float64), s.astype(np.float64))
                 for g, m, s in zip(st["groups"], st["med"], st["scale"])}
        neg = np.load(self.dir / "ctrl_wells.npz", allow_pickle=True)
        n_meta = pd.DataFrame({"well": neg["well"], "group": neg["group"]})
        wmeta = (self.t[["well", "plate", "compound"]].astype(str)
                 .drop_duplicates("well").set_index("well"))

        row = {"pheno_dir": self.path, "pheno_n_classes": len(self.classes),
               "pheno_n_eval": int(len(self.t)),
               "pheno_n_unlearnable": int((~ok).sum())}
        preds = self.t[["crop_id", "compound", "plate", "well"]].copy()
        maps = []
        for p in POPULATIONS:
            r = self.ranks[p]
            if (r[ok] < 0).any():
                raise SystemExit(f"{p}: {int((r[ok] < 0).sum())} crop(s) never scored")
            for k, v in pm.topk(r[ok], KS).items():
                row[f"clf_top{k}_{p}"] = v
            preds[f"rank_{p}"] = np.where(ok, r, -1)

            wells, M, n = self.acc[p].result()
            meta = wmeta.loc[wells]
            Z = pm.normalise(M, meta["plate"], stats)
            q_meta = pd.DataFrame({"well": wells, "pert": meta["compound"].to_numpy(),
                                   "group": meta["plate"].to_numpy()})
            _, mp = pm.replicate_map(Z, q_meta, Z, q_meta, neg["Z"], n_meta,
                                     null_size=self.null_size,
                                     threshold=self.threshold)
            row[f"map_{p}"] = float(mp["map"].mean()) if len(mp) else float("nan")
            row[f"map_frac_retrieved_{p}"] = (float(mp["retrieved"].mean())
                                              if len(mp) else float("nan"))
            row[f"map_n_perts_{p}"] = int(len(mp))
            maps.append(mp.assign(population=p))
        preds.to_parquet(Path(out_dir) / f"{name}_preds.parquet", index=False)
        pd.concat(maps, ignore_index=True).to_parquet(
            Path(out_dir) / f"{name}_map.parquet", index=False)
        return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="featurise real crops, fit the heads, "
                                   "store the mAP references")
    f.add_argument("--config", required=True)
    f.add_argument("--out", default=None, help="default <out_dir>/pheno")
    f.add_argument("--epochs", type=int, default=10)
    f.add_argument("--batch_size", type=int, default=128)
    f.add_argument("--num_workers", type=int, default=8)
    f.add_argument("--force", action="store_true",
                   help="re-featurise even if feats.npy matches")
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = Path(a.out or Path(cfg["out_dir"]) / "pheno")
    fit(cfg, out, a.epochs, a.batch_size, a.num_workers, a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
