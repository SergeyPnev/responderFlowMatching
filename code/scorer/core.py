"""HiDDEN scorer and diagnostics -- pure numpy/sklearn, no dataset knowledge.

Everything here works on a matrix ``X`` (n_crops, D), a binary label ``y``
(1 = treated, 0 = same-plate control) and a group vector ``wells``.

Differences from published HiDDEN:

1. Well-level cross-fitting: every score is out-of-fold with groups = wells
   (crops from one well share illumination, confluency and focus).
2. No binarisation: the continuous posterior is kept.
3. pi from the ROC plateau, not from the mean score:

    pi_hat(a) = (F_t(a) - F_c(a)) / (1 - F_c(a))       = (TPR - FPR) / (1 - FPR)

   with ``F(a) = P(h > a)``; flat in ``a`` when the two-component model holds.
"""
from __future__ import annotations

import hashlib

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

try:                                        # sklearn >= 0.24
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:                         # pragma: no cover
    StratifiedGroupKFold = None
from sklearn.model_selection import GroupKFold

EPS = 1e-6


# ===================================================================== #
# Scoring
# ===================================================================== #
def _robust_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Median / 1.4826*MAD, per feature."""
    m = np.median(X, axis=0)
    s = 1.4826 * np.median(np.abs(X - m), axis=0)
    return m, s


class SingleClassFold(ValueError):
    """A fold's training set holds only one class. Raised, never swallowed."""


def make_folds(y: np.ndarray, groups: np.ndarray, n_splits: int = 5,
               seed: int = 0, mode: str = "lowo",
               small_group_max: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Well-level folds, capped by the thinner class's well count.

    ``mode="lowo"`` (default): when the minority class has ``small_group_max``
    groups or fewer, its groups are dealt leave-one-out and deterministically
    (``k = n_minority_groups``, no RNG) and the majority groups round-robin over
    the same k folds. ``mode="stratified"`` uses StratifiedGroupKFold, which
    balances by crop count and can leave a training set single-class when the
    minority class has few wells.
    """
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    gt = np.unique(groups[y == 1])
    gc = np.unique(groups[y == 0])
    n_t, n_c = len(gt), len(gc)
    shared = set(gt.tolist()) & set(gc.tolist())
    if shared:
        raise ValueError(f"{len(shared)} group(s) carry both classes, so no "
                         f"group-wise split is honest: {sorted(shared)[:3]}")

    minor, major = (gt, gc) if n_t <= n_c else (gc, gt)

    def deal(k: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        # minority groups round-robin over k folds, then majority groups the
        # same way: every fold tests >= 1 group of each class, trains on the rest
        assign = {g: i % k for i, g in enumerate(sorted(minor, key=str))}
        assign.update({g: i % k for i, g in enumerate(sorted(major, key=str))})
        fold_of = np.array([assign[g] for g in groups])
        return [(np.flatnonzero(fold_of != i), np.flatnonzero(fold_of == i))
                for i in range(k)]

    if mode == "lowo" and min(n_t, n_c) <= small_group_max:
        k = len(minor)
        if k < 2:
            raise SingleClassFold(
                f"minority class has {k} group(s); no group-wise split exists "
                f"({n_t} treated / {n_c} control wells)")
        folds = deal(k)
    else:
        k = int(max(2, min(n_splits, n_t, n_c)))
        if StratifiedGroupKFold is not None:
            sp = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            folds = list(sp.split(np.zeros(len(y)), y, groups))
        else:                                                # pragma: no cover
            folds = list(GroupKFold(n_splits=k).split(np.zeros(len(y)), y, groups))
        # StratifiedGroupKFold packs groups by crop count and can leave a
        # training set single-class; lowo then deals the groups instead.
        if mode == "lowo" and any(len(np.unique(y[tr])) < 2 or not len(te)
                                  for tr, te in folds):
            folds = deal(k)

    bad = [i for i, (tr, _) in enumerate(folds) if len(np.unique(y[tr])) < 2]
    if bad:
        raise SingleClassFold(
            f"fold(s) {bad} have a single class in train "
            f"({n_t} treated / {n_c} control wells, mode={mode}, k={len(folds)})")
    empty = [i for i, (_, te) in enumerate(folds) if not len(te)]
    if empty:
        raise SingleClassFold(f"fold(s) {empty} have an empty test set")
    return folds


def fold_hash(groups: np.ndarray,
              folds: List[Tuple[np.ndarray, np.ndarray]]) -> str:
    """Short digest of the group -> fold map, for the unit manifest."""
    m = {}
    for i, (_, te) in enumerate(folds):
        for g in np.unique(np.asarray(groups)[te]):
            m[str(g)] = i
    payload = ";".join(f"{g}:{m[g]}" for g in sorted(m))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def fit_apply(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
              n_pcs: Optional[int] = 50, C: float = 1.0, seed: int = 0,
              class_weight: Optional[str] = "balanced",
              max_iter: int = 2000) -> np.ndarray:
    """Fit one scorer on ``X_tr`` and apply it to ``X_te``.

    The scaler is fit on the training controls and the PCA on the training
    rows only, so nothing about ``X_te`` enters the transform.
    """
    ref = X_tr[y_tr == 0]
    m, s_ = _robust_stats(ref if len(ref) >= 2 else X_tr)
    A = (X_tr - m) / (s_ + 1e-8)
    B = (X_te - m) / (s_ + 1e-8)
    if n_pcs:
        k = int(min(n_pcs, A.shape[0] - 1, A.shape[1]))
        pca = PCA(n_components=k, random_state=seed).fit(A)
        A, B = pca.transform(A), pca.transform(B)
    lr = LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter)
    lr.fit(A, y_tr)
    return lr.predict_proba(B)[:, 1]


def hidden_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  n_pcs: Optional[int] = 50, n_splits: int = 5,
                  C: float = 1.0, seed: int = 0, folds_mode: str = "lowo",
                  folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                  class_weight: Optional[str] = "balanced",
                  max_iter: int = 2000) -> np.ndarray:
    """Out-of-fold P(treated | crop).

    ``n_pcs=None`` skips PCA and fits the logistic regression on the scaled
    features directly. Scaler and PCA are fit inside each fold; folds are by
    group (well), never by crop.
    """
    y = np.asarray(y).astype(int)
    h = np.full(len(y), np.nan, dtype=np.float64)

    for tr, te in (folds if folds is not None else
                   make_folds(y, groups, n_splits, seed, mode=folds_mode)):
        h[te] = fit_apply(X[tr], y[tr], X[te], n_pcs=n_pcs, C=C, seed=seed,
                          class_weight=class_weight, max_iter=max_iter)

    if np.isnan(h).any():                                    # pragma: no cover
        raise RuntimeError("some crops never landed in a test fold")
    return h


# ===================================================================== #
# pi
# ===================================================================== #
def auroc(y: np.ndarray, h: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, h))


def pi_curve(h_t: np.ndarray, h_c: np.ndarray,
             qs: np.ndarray) -> np.ndarray:
    """pi_hat evaluated at thresholds = the ``qs`` quantiles of the controls."""
    a = np.quantile(h_c, qs)
    hc, ht = np.sort(h_c), np.sort(h_t)
    F_c = 1.0 - np.searchsorted(hc, a, side="right") / len(hc)
    F_t = 1.0 - np.searchsorted(ht, a, side="right") / len(ht)
    return np.clip((F_t - F_c) / np.clip(1.0 - F_c, 1e-9, None), 0.0, 1.0)


def pi_hat(h_t: np.ndarray, h_c: np.ndarray,
           band: Tuple[float, float] = (0.75, 0.95),
           tol: float = 0.10, n_grid: int = 50) -> Dict[str, object]:
    """Plateau height, its spread, and the whole curve.

    ``band`` is in control-quantile units. ``spread > tol`` means there is no
    plateau, so the two-component reading of ``pi`` does not hold.
    """
    qs = np.linspace(0.50, 0.99, n_grid)
    curve = pi_curve(h_t, h_c, qs)
    in_band = (qs >= band[0]) & (qs <= band[1])
    seg = curve[in_band]
    spread = float(seg.max() - seg.min()) if seg.size else float("nan")
    return {
        "pi": float(np.median(seg)) if seg.size else float("nan"),
        "spread": spread,
        "has_plateau": bool(spread < tol),
        "qs": qs,
        "curve": curve,
    }


def pi_from_auroc(a: float) -> float:
    """``2*AUROC - 1`` clipped to [0, 1], reported alongside the plateau pi."""
    return float(np.clip(2.0 * a - 1.0, 0.0, 1.0))


def responder_weight(h: np.ndarray, pi: float,
                     prior_ratio: float = 1.0) -> np.ndarray:
    """s = P(R=1 | x) = clip(1 - (1-pi) * rho, 0, 1),  rho = p_c(x)/p_t(x).

    ``prior_ratio`` is ``n_t/n_c`` when the classifier was fit on the natural
    class priors and 1.0 when it was fit with ``class_weight="balanced"``
    (the ``hidden_scores`` default). The clip gives hard zeros.
    """
    hh = np.clip(h, EPS, 1 - EPS)
    rho = prior_ratio * (1.0 - hh) / hh
    return np.clip(1.0 - (1.0 - pi) * rho, 0.0, 1.0)


# ===================================================================== #
# Well-level uncertainty
# ===================================================================== #
def well_bootstrap(h: np.ndarray, y: np.ndarray, wells: np.ndarray,
                   B: int = 1000, seed: int = 0,
                   band: Tuple[float, float] = (0.75, 0.95),
                   ) -> Dict[str, Tuple[float, float]]:
    """Percentile CIs for AUROC and pi, resampling wells (not crops) with
    replacement, treated and control wells separately."""
    rng = np.random.default_rng(seed)
    idx_of = {w: np.flatnonzero(wells == w) for w in np.unique(wells)}
    w_t = np.unique(wells[y == 1])
    w_c = np.unique(wells[y == 0])

    aur, pis = [], []
    for _ in range(B):
        take = np.concatenate(
            [np.concatenate([idx_of[w] for w in rng.choice(w_t, len(w_t))]),
             np.concatenate([idx_of[w] for w in rng.choice(w_c, len(w_c))])])
        yb, hb = y[take], h[take]
        aur.append(auroc(yb, hb))
        pis.append(pi_hat(hb[yb == 1], hb[yb == 0], band=band)["pi"])

    def ci(v):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if not len(v):
            return (float("nan"), float("nan"))
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    return {"auroc": ci(aur), "pi": ci(pis)}


