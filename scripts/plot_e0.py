"""Figures for the E0 simulation (reads results/metrics/e0_reps.csv).

    python scripts/plot_e0.py [--version v2]

Reads results/metrics/e0_<version>/e0_reps.csv and writes
results/figures/e0_<version>/e0_coverage.png, e0_kappa.png, e0_decisions.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPS = ROOT / "results/metrics/e0_v2/e0_reps.csv"
FIG = ROOT / "results/figures/e0_v2"

# reference palette (light mode), validated with the dataviz validator
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b9b8b2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow

SCEN_ORDER = ["kappa_0.5", "kappa_1", "kappa_2", "reversed", "misspec_kappa_1", "bayes_kappa_1"]
SCEN_LABEL = {
    "kappa_0.5": "κ = 0.5",
    "kappa_1": "κ = 1",
    "kappa_2": "κ = 2",
    "reversed": "reversed (κ = −1)",
    "misspec_kappa_1": "misspecified raters, κ = 1",
    "bayes_kappa_1": "exact Bayes update planted",
}
DESIGN_LABEL = {"pcgita": "PC-GITA-sized (40 speakers × 8 items)",
                "gleason": "Gleason2019-sized (215 cores × 2 patterns)"}

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": NEUTRAL,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK,
    "axes.titlecolor": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def coverage_plot(df: pd.DataFrame) -> None:
    designs = list(DESIGN_LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = np.arange(len(SCEN_ORDER))[::-1]
    for ax, d in zip(axes, designs):
        sub = df[df.design == d].groupby("scenario")
        fixed = sub.covered_fixed.mean().reindex(SCEN_ORDER).to_numpy()
        prop = sub.covered.mean().reindex(SCEN_ORDER).to_numpy()
        ax.axvspan(0.9, 1.0, color=SERIES[2], alpha=0.08, linewidth=0)
        ax.axvline(0.9, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
        ax.hlines(y, fixed, prop, color=NEUTRAL, linewidth=2)
        ax.scatter(fixed, y, s=70, color=SERIES[1], edgecolor=SURFACE, linewidth=2, zorder=3,
                   label="references held fixed")
        ax.scatter(prop, y, s=70, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3,
                   label="reference uncertainty propagated")
        for yi, v in zip(y, prop):
            ax.text(1.045, yi, f"{v:.0%}", color=INK, fontsize=9,
                    ha="left", va="center", clip_on=False)
        ax.text(0.9, y.max() + 0.55, "90% target", color=INK_2, fontsize=9, ha="center")
        ax.set_xlim(-0.02, 1.03)
        ax.set_ylim(-0.6, y.max() + 0.9)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(DESIGN_LABEL[d], fontsize=11, loc="left")
        ax.set_xlabel("95% interval covers the planted κ")
        _style(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([SCEN_LABEL[s] for s in SCEN_ORDER])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.01, 1.0))
    fig.suptitle("E0: interval coverage of κ (target ≥ 90%, shaded)", x=0.01, y=1.08,
                 ha="left", fontsize=13, color=INK)
    fig.tight_layout(w_pad=4)
    fig.savefig(FIG / "e0_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def kappa_plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = np.arange(len(SCEN_ORDER))[::-1]
    for ax, d in zip(axes, DESIGN_LABEL):
        sub = df[df.design == d]
        data = [sub[sub.scenario == s].kappa_hat.to_numpy() for s in SCEN_ORDER]
        ax.boxplot(data, positions=y, orientation="horizontal", widths=0.45, showfliers=False,
                   patch_artist=True,
                   boxprops={"facecolor": "#dfeaf8", "edgecolor": SERIES[0], "linewidth": 1.2},
                   medianprops={"color": SERIES[0], "linewidth": 2},
                   whiskerprops={"color": SERIES[0], "linewidth": 1.2},
                   capprops={"color": SERIES[0], "linewidth": 1.2})
        planted = [sub[sub.scenario == s].planted_kappa.iloc[0] for s in SCEN_ORDER]
        ax.scatter(planted, y, marker="|", s=400, color=INK, linewidth=2, zorder=3)
        ax.axvline(0, color=NEUTRAL, linewidth=1)
        ax.set_title(DESIGN_LABEL[d], fontsize=11, loc="left")
        ax.set_xlabel("estimated κ")
        _style(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([SCEN_LABEL[s] for s in SCEN_ORDER])
    fig.suptitle("E0: estimated κ over replicates (box: IQR, whiskers: 1.5 IQR) "
                 "vs planted value (black tick)", x=0.01, y=1.03,
                 ha="left", fontsize=13, color=INK)
    fig.tight_layout(w_pad=3)
    fig.savefig(FIG / "e0_kappa.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def decision_plot(df: pd.DataFrame) -> None:
    cats = ["under", "calibrated", "over", "misdirected", "inconclusive"]
    colors = {"under": SERIES[2], "calibrated": SERIES[0], "over": SERIES[1],
              "misdirected": SERIES[3], "inconclusive": NEUTRAL}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = np.arange(len(SCEN_ORDER))[::-1]
    for ax, d in zip(axes, DESIGN_LABEL):
        sub = df[df.design == d]
        shares = (sub.groupby("scenario").label.value_counts(normalize=True)
                  .unstack(fill_value=0).reindex(index=SCEN_ORDER, columns=cats, fill_value=0))
        left = np.zeros(len(SCEN_ORDER))
        for c in cats:
            w = shares[c].to_numpy()
            ax.barh(y, w, left=left, height=0.6, color=colors[c], edgecolor=SURFACE,
                    linewidth=2, label=c)
            for yi, l0, wi in zip(y, left, w):
                if wi >= 0.12:
                    ax.text(l0 + wi / 2, yi, f"{wi:.0%}", ha="center", va="center",
                            fontsize=9, color=INK)
            left += w
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.set_title(DESIGN_LABEL[d], fontsize=11, loc="left")
        ax.set_xlabel("share of replicates")
        _style(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([SCEN_LABEL[s] for s in SCEN_ORDER])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=5, frameon=False, bbox_to_anchor=(0.01, 1.0))
    fig.suptitle("E0: preregistered decision per replicate (Holm-adjusted, α = 0.05)",
                 x=0.01, y=1.08, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "e0_decisions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global REPS, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    v = ap.parse_args().version
    REPS = ROOT / f"results/metrics/e0_{v}/e0_reps.csv"
    FIG = ROOT / f"results/figures/e0_{v}"
    df = pd.read_csv(REPS)
    FIG.mkdir(parents=True, exist_ok=True)
    coverage_plot(df)
    kappa_plot(df)
    decision_plot(df)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
