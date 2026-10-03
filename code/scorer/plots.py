"""Figures for a finished run directory.  ``python scorer/plots.py <out_dir>``"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402


def _read(out, name):
    p = os.path.join(out, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def make_all(out: str, top: int = 24) -> None:
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    tbl = _read(out, "e1_e2_compounds.csv")
    if tbl is None or not len(tbl):
        print("[figures] nothing to plot")
        return
    tbl["name"] = (tbl["compound"].astype(str) + tbl["concentration"]
                   .apply(lambda c: "" if pd.isna(c) else f" @{c:g}"))

    # AUROC with well-level CI
    d = tbl.sort_values("auroc", ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(d) + 1.4))
    ax.barh(d["name"], d["auroc"], color="#4878a8",
            xerr=[d["auroc"] - d["auroc_lo"], d["auroc_hi"] - d["auroc"]],
            error_kw=dict(lw=0.8, ecolor="0.3"))
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("out-of-fold AUROC vs same-plate control")
    ax.tick_params(labelsize=7); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "e1_auroc.png"), dpi=180); plt.close(fig)

    # pi-hat vs threshold: the plateau plot
    cur = _read(out, "e2_pi_curves.csv")
    if cur is not None and len(cur):
        cur["name"] = (cur["compound"].astype(str) + cur["concentration"]
                       .apply(lambda c: "" if pd.isna(c) else f" @{c:g}"))
        pick = tbl.sort_values("auroc", ascending=False).head(12)["name"]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        cmap = plt.get_cmap("viridis")
        for i, n in enumerate(pick):
            g = cur[cur["name"] == n]
            if len(g):
                ax.plot(g["q"], g["pi_q"], lw=1.3, color=cmap(i / max(len(pick) - 1, 1)),
                        label=n[:26])
        ax.axvspan(0.75, 0.95, color="0.9", zorder=0)
        ax.set_xlabel(r"threshold, as a control quantile $q$")
        ax.set_ylabel(r"$\hat\pi(a_0)$"); ax.set_ylim(0, 1)
        ax.set_title("flat over the shaded band = the two-component model fits",
                     fontsize=9)
        ax.legend(fontsize=6, ncol=2); fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "e2_plateau.png"), dpi=180); plt.close(fig)

    # pi is not the mean score
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(tbl["pi"], tbl["mean_score_treated"], s=16, c="#4878a8")
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.plot([0, 1], [0, 1], color="0.8", lw=1)
    p = np.linspace(0, 1, 100)
    ax.plot(p, p + (1 - p) ** 2 / (2 - p), color="#a83232", lw=1.2,
            label=r"$\pi+(1-\pi)^2/(2-\pi)$")
    ax.set_xlabel(r"$\hat\pi$ (ROC plateau)"); ax.set_ylabel("mean classifier score")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "e2_pi_vs_mean.png"), dpi=180); plt.close(fig)

    # pi-hat by mechanism of action
    if "moa" in tbl.columns and tbl["moa"].notna().any():
        g = tbl.dropna(subset=["moa"])
        order = g.groupby("moa")["pi"].mean().sort_values().index
        fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(order) + 1.6))
        ax.boxplot([g.loc[g.moa == m, "pi"].values for m in order], vert=False,
                   labels=[str(m)[:30] for m in order], widths=0.6)
        ax.set_xlabel(r"$\hat\pi$"); ax.set_xlim(0, 1); ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "e3_mechanism.png"), dpi=180); plt.close(fig)

    # pi-hat vs dose
    e4 = _read(out, "e4_dose.csv")
    if e4 is not None and len(e4):
        multi = e4.groupby("compound").filter(lambda g: len(g) > 1)
        if len(multi):
            fig, ax = plt.subplots(figsize=(6, 4.2))
            for c, g in multi.groupby("compound"):
                g = g.sort_values("concentration")
                ax.plot(g["concentration"], g["pi"], "o-", ms=3, lw=1, label=str(c)[:22])
            ax.set_xscale("log"); ax.set_xlabel("concentration")
            ax.set_ylabel(r"$\hat\pi$"); ax.set_ylim(0, 1)
            ax.legend(fontsize=6, ncol=2); fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "e4_dose.png"), dpi=180); plt.close(fig)
    print(f"[figures] -> {fig_dir}")


if __name__ == "__main__":
    make_all(sys.argv[1])
