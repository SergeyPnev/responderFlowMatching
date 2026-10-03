"""Replicate-detection mAP and top-k perturbation accuracy. Numpy only, no model.

Profiles are well means of per-crop features, robust-z normalised per group
against that group's real control crops. For a query well of perturbation i in
group g, candidates = the other wells of i (positives) + real control wells of
g (negatives), ranked by cosine similarity; a perturbation's mAP is the mean AP
of its queries.

Significance: per-query AP nulls under a uniformly random ranking, averaged
with independent draws per query; ``p = (1 + #{null >= mAP}) / (1 + null_size)``
(``>=``, not ``>``: the null is discrete with few candidates), then
Benjamini-Hochberg over perturbations within a population.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
def robust_stats(X: np.ndarray, groups: np.ndarray, floor_frac: float = 1e-2
                 ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Per group: (median, scale) of the control crops ``X``.

    ``scale = max(1.4826*MAD, floor_frac * median over features of that)``;
    the floor keeps near-constant features from dominating cosine similarity.
    """
    out = {}
    for g in np.unique(groups):
        x = np.asarray(X[groups == g], dtype=np.float64)
        med = np.median(x, axis=0)
        mad = 1.4826 * np.median(np.abs(x - med), axis=0)
        pos = mad[mad > 0]
        floor = floor_frac * (np.median(pos) if len(pos) else 1.0)
        out[str(g)] = (med, np.maximum(mad, floor))
    return out


def normalise(X: np.ndarray, groups: Iterable, stats) -> np.ndarray:
    groups = np.asarray([str(g) for g in groups])
    Z = np.empty(X.shape, dtype=np.float64)
    for g in np.unique(groups):
        if g not in stats:
            raise KeyError(f"no control statistics for group {g!r}")
        m = groups == g
        med, sc = stats[g]
        Z[m] = (X[m] - med) / sc
    return Z


class WellMeans:
    """Per-well mean profile, accumulated chunk by chunk (memory: n_wells x D)."""

    def __init__(self, dim: int):
        self.dim, self.row, self.sums, self.n = dim, {}, [], []

    def add(self, X: np.ndarray, wells: Iterable) -> None:
        codes, uniq = pd.factorize(pd.Series([str(w) for w in wells]))
        S = np.zeros((len(uniq), self.dim))
        np.add.at(S, codes, np.asarray(X, dtype=np.float64))
        cnt = np.bincount(codes, minlength=len(uniq))
        for k, w in enumerate(uniq):
            j = self.row.get(w)
            if j is None:
                j = self.row[w] = len(self.n)
                self.sums.append(np.zeros(self.dim))
                self.n.append(0)
            self.sums[j] += S[k]
            self.n[j] += int(cnt[k])

    def result(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(well ids, mean profile per well, crop count per well), wells sorted."""
        ids = sorted(self.row)
        n = np.array([self.n[self.row[w]] for w in ids])
        M = (np.stack([self.sums[self.row[w]] for w in ids]) / n[:, None]
             if ids else np.zeros((0, self.dim)))
        return np.asarray(ids), M, n


# --------------------------------------------------------------------------- #
def average_precision(scores: np.ndarray, is_pos: np.ndarray) -> float:
    """AP of ranking ``scores`` descending; ties broken by position (stable)."""
    order = np.argsort(-scores, kind="stable")
    rel = is_pos[order].astype(np.float64)
    n_pos = rel.sum()
    if n_pos == 0:
        return float("nan")
    prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
    return float((prec * rel).sum() / n_pos)


def null_ap(n_pos: int, n_total: int, null_size: int, seed: int) -> np.ndarray:
    """AP under a uniformly random ranking, ``null_size`` draws."""
    rng = np.random.default_rng(seed)
    out = np.empty(null_size, dtype=np.float64)
    k = np.arange(1, n_pos + 1)
    chunk = max(1, 2_000_000 // max(n_total, 1))
    for s in range(0, null_size, chunk):
        m = min(chunk, null_size - s)
        # ranks (1-based) of the positives: the first n_pos of a permutation
        ranks = np.sort(rng.random((m, n_total)).argsort(axis=1)[:, :n_pos],
                        axis=1) + 1
        out[s:s + m] = (k / ranks).mean(axis=1)
    return out


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


def _unit(Z: np.ndarray) -> np.ndarray:
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def replicate_map(Q: np.ndarray, q_meta: pd.DataFrame,
                  P: np.ndarray, p_meta: pd.DataFrame,
                  N: np.ndarray, n_meta: pd.DataFrame,
                  null_size: int = 100_000, threshold: float = 0.05,
                  seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-query AP and per-perturbation mAP / p / q.

    ``*_meta`` carry ``well`` and ``pert`` (queries, positives) or ``well`` and
    ``group`` (negatives); queries also carry ``group``. Profiles must already be
    normalised. A positive with the query's own well id is never counted.
    """
    Qn, Pn, Nn = _unit(Q), _unit(P), _unit(N)
    p_pert = p_meta["pert"].astype(str).to_numpy()
    p_well = p_meta["well"].astype(str).to_numpy()
    n_group = n_meta["group"].astype(str).to_numpy()
    pos_of = pd.Series(np.arange(len(p_pert))).groupby(p_pert).apply(np.asarray).to_dict()
    neg_of = pd.Series(np.arange(len(n_group))).groupby(n_group).apply(np.asarray).to_dict()

    rows = []
    for i, (w, pert, g) in enumerate(zip(q_meta["well"].astype(str),
                                         q_meta["pert"].astype(str),
                                         q_meta["group"].astype(str))):
        pi = pos_of.get(pert, np.empty(0, int))
        pi = pi[p_well[pi] != w]
        ni = neg_of.get(g, np.empty(0, int))
        if len(pi) == 0 or len(ni) == 0:
            rows.append((w, pert, g, np.nan, len(pi), len(pi) + len(ni)))
            continue
        # negatives first: a stable sort then breaks exact ties against the
        # positives, never in their favour
        s = np.concatenate([Nn[ni] @ Qn[i], Pn[pi] @ Qn[i]])
        is_pos = np.r_[np.zeros(len(ni), bool), np.ones(len(pi), bool)]
        rows.append((w, pert, g, average_precision(s, is_pos), len(pi), len(s)))
    ap = pd.DataFrame(rows, columns=["well", "pert", "group", "ap", "n_pos",
                                     "n_total"])

    ok = ap.dropna(subset=["ap"])
    shapes = sorted(set(zip(ok["n_pos"], ok["n_total"])))
    nulls = {sh: null_ap(int(sh[0]), int(sh[1]), null_size,
                         seed=seed + 7919 * int(sh[0]) + int(sh[1]))
             for sh in shapes}
    rng = np.random.default_rng(seed)
    per = []
    for pert, sub in ok.groupby("pert"):
        m = float(sub["ap"].mean())
        nd = np.zeros(null_size)
        for j, (a, b) in enumerate(zip(sub["n_pos"], sub["n_total"])):
            pool = nulls[(a, b)]
            nd += pool if j == 0 else pool[rng.integers(0, null_size, null_size)]
        nd /= len(sub)
        p = (1 + int((nd >= m - 1e-9).sum())) / (1 + null_size)
        per.append((pert, m, p, len(sub)))
    mp = pd.DataFrame(per, columns=["pert", "map", "p", "n_queries"])
    mp["q"] = bh(mp["p"].to_numpy())
    mp["retrieved"] = mp["q"] < threshold
    return ap, mp


# --------------------------------------------------------------------------- #
def true_rank(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """0-based rank of the true class (0 = top-1); ties count against it."""
    t = logits[np.arange(len(y)), y]
    return (logits >= t[:, None]).sum(axis=1) - 1


def topk(ranks: np.ndarray, ks=(1, 5, 10)) -> Dict[int, float]:
    r = np.asarray(ranks)
    return {k: float((r < k).mean()) if len(r) else float("nan") for k in ks}
