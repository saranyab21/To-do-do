"""Preliminary Gleason2019 results with confidence intervals (CPU, seconds).

    python scripts/summarise_gleason_prelim.py

Inputs (git-ignored, per core): zero-shot scores and out-of-fold probe predictions. They
are looked up where the GPU scripts write them, then in data/processed/gleason2019/ (where
copies from the CIP pool land):
    zeroshot/zeroshot_v1/scores.jsonl                  or  zeroshot_scores.jsonl
    features/probe_fp16vision/predictions_probe_v3.csv or  predictions_probe_v3.csv
    features/probe_v1/predictions_probe_v2.csv         or  predictions_probe_v2.csv

The headline probe is probe_v3 (vision tower in fp16, as intended). probe_v2 used features
from a vision tower that was unintentionally quantised to 4-bit; it is kept as a
sensitivity check (methods suffixed "_v2", and the "v3 - v2" paired differences).

Every method is scored on the same 244 cores, so differences are paired. Intervals: 2,000
bootstrap resamples of cores, stratified by slide (each slide keeps its number of cores),
i.e. conditional on these six slides. Feature sets are fixed in advance, not picked as
the best: the vision-tower output (img_L0) and the last layer at the answer position
(last_L36).

Writes results/metrics/gleason2019_prelim/{nll_ci.csv, paired_diffs.csv} and
results/figures/gleason2019_prelim/nll_overview.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.soft_label import per_case

PROC = ROOT / "data/processed/gleason2019"
MET = ROOT / "results/metrics/gleason2019_prelim"
FIG = ROOT / "results/figures/gleason2019_prelim"
CODES = [0, 3, 4, 5]
SIZES = [448, 896, 1344]
B, SEED = 2000, 20261126

# reference palette (light mode), validated with the dataviz validator (as in plot_e0.py)
SURFACE, INK, INK_2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
SIZE_COLOUR = dict(zip(SIZES, ["#2a78d6", "#eb6834", "#1baf7a"]))  # blue, orange, aqua

METHODS = {  # key -> label (top to bottom in the figure)
    "zeroshot_TB": "Zero-shot answer\n(recalibrated: temperature + bias)",
    "colour": "Colour statistics\n(probe)",
    "last_L36": "Answer-position features,\nlast layer (probe)",
    "img_L0": "Image-token features,\nvision encoder output (probe)",
}


def find(*names: str) -> Path:
    for n in names:
        if (PROC / n).exists():
            return PROC / n
    sys.exit(f"missing per-core file; looked for {names} under {PROC}")


def per_core_losses() -> pd.DataFrame:
    """Long table: core, slide, dimension, method, size, nll, top_match."""
    dist = pd.read_csv(PROC / "rater_distributions.csv").query("n_raters > 0")
    folds = pd.read_csv(PROC / "folds.csv")[["core", "slide", "fold"]]
    zs = pd.DataFrame([json.loads(x) for x in
                       find("zeroshot/zeroshot_v1/scores.jsonl", "zeroshot_scores.jsonl")
                       .read_text().splitlines() if x])
    pr = pd.read_csv(find("features/probe_fp16vision/predictions_probe_v3.csv",
                          "predictions_probe_v3.csv"))
    v2 = pd.read_csv(find("features/probe_v1/predictions_probe_v2.csv",
                          "predictions_probe_v2.csv"))
    v2 = v2[v2.feature.isin(["img_L0", "last_L36"])].assign(feature=lambda d: d.feature + "_v2")
    out = []
    for name, df, keys in (("zeroshot", zs, ["size", "dimension"]),
                           ("probe", pr, ["feature", "size", "dimension"]),
                           ("probe", v2, ["feature", "size", "dimension"])):
        df = df.merge(dist, on=["core", "dimension"]).merge(folds[["core", "fold"]], on="core")
        for key, g in df.groupby(keys):
            g = g.sort_values("core")
            c, *_ = per_case(g, CODES)
            meta = dict(zip(keys, key))
            if name == "zeroshot":
                out.append(c.assign(method="zeroshot_TB", nll=c.nll_TB, **meta))
                out.append(c.assign(method="prior", nll=c.nll_prior, **meta))
                out.append(c.assign(method="floor", nll=c.entropy, **meta))
            else:
                out.append(c.assign(method=meta.pop("feature"), **meta))
    long = pd.concat(out, ignore_index=True)
    return long.merge(folds[["core", "slide"]], on="core")


def boot_indices(slides: np.ndarray, rng) -> np.ndarray:
    groups = [np.flatnonzero(slides == s) for s in np.unique(slides)]
    return np.stack([np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(B)])


def summarise(long: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    rows, diffs = [], []
    for dim, d in long.groupby("dimension"):
        wide = d.pivot_table(index=["core", "slide"], columns=["method", "size"],
                             values=["nll", "top_match"])
        idx = boot_indices(wide.index.get_level_values("slide").to_numpy(), rng)
        for (stat, method, size) in wide.columns:
            v = wide[(stat, method, size)].to_numpy()
            bs = v[idx].mean(1)
            rows.append({"dimension": dim, "method": method, "size": size, "stat": stat,
                         "mean": v.mean(), "lo": np.quantile(bs, .025), "hi": np.quantile(bs, .975)})

        def diff(a, b, label, wide=wide, idx=idx, dim=dim):
            va, vb = wide[("nll",) + a].to_numpy(), wide[("nll",) + b].to_numpy()
            bs = (va - vb)[idx].mean(1)
            diffs.append({"dimension": dim, "comparison": label, "diff": (va - vb).mean(),
                          "lo": np.quantile(bs, .025), "hi": np.quantile(bs, .975)})

        for s in SIZES:
            diff(("img_L0", s), ("prior", s), f"img_L0 {s}px - label prior")
            diff(("img_L0", s), ("colour", 448), f"img_L0 {s}px - colour 448px")
            diff(("img_L0", s), ("zeroshot_TB", s), f"img_L0 {s}px - zero-shot recalibrated {s}px")
            diff(("last_L36", s), ("img_L0", s), f"last_L36 {s}px - img_L0 {s}px")
            diff(("zeroshot_TB", s), ("prior", s), f"zero-shot recalibrated {s}px - label prior")
            for f in ("img_L0", "last_L36"):  # vision tower fp16 (v3) vs 4-bit (v2)
                diff((f, s), (f + "_v2", s), f"{f} {s}px: fp16 vision (v3) - 4-bit vision (v2)")
        for s in (896, 1344):
            diff(("img_L0", s), ("img_L0", 448), f"img_L0 {s}px - img_L0 448px")
            diff(("last_L36", s), ("last_L36", 448), f"last_L36 {s}px - last_L36 448px")
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def plot(ci: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": NEUTRAL, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK, "axes.titlecolor": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE})
    nll = ci[ci.stat == "nll"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, dim in zip(axes, ("primary", "secondary")):
        d = nll[nll.dimension == dim]
        ref = {m: d[(d.method == m) & (d["size"] == 448)].iloc[0] for m in ("prior", "floor")}
        ax.axvspan(ref["prior"]["lo"], ref["prior"]["hi"], color=GRID, alpha=0.7, lw=0)
        ax.axvline(ref["prior"]["mean"], color=INK_2, lw=1.2, ls="--")
        ax.axvline(np.log(4), color=NEUTRAL, lw=1.2, ls=":")
        ax.axvline(ref["floor"]["mean"], color=INK_2, lw=1.2)
        top = len(METHODS) - 0.35
        ax.text(ref["prior"]["mean"], top, " label mix of\n other slides", color=INK_2,
                fontsize=9, va="top")
        ax.text(np.log(4), top, " uniform\n guess", color=INK_2, fontsize=9, va="top")
        ax.text(ref["floor"]["mean"], top, " pathologists'\n disagreement", color=INK_2,
                fontsize=9, va="top")
        for y, m in enumerate(reversed(list(METHODS))):
            for j, s in enumerate(SIZES):
                r = d[(d.method == m) & (d["size"] == s)]
                if r.empty or (m == "colour" and s != 448):
                    continue
                r = r.iloc[0]
                yy = y + (0.22 * (1 - j) if m != "colour" else 0)
                ax.plot([r.lo, r.hi], [yy, yy], color=SIZE_COLOUR[s], lw=2, solid_capstyle="round")
                ax.plot(r["mean"], yy, "o", ms=8, color=SIZE_COLOUR[s], mec=SURFACE, mew=2)
        ax.set_yticks(range(len(METHODS)), list(reversed(METHODS.values())))
        ax.set_ylim(-0.6, len(METHODS) - 0.2)
        ax.set_title(f"{dim.capitalize()} Gleason pattern", loc="left", fontsize=12)
        ax.set_xlabel("Soft cross-entropy vs pathologists' answers (lower is better)")
        ax.set_xlim(0.25, 1.45)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=SIZE_COLOUR[s],
                          label=f"{s} px") for s in SIZES]
    fig.legend(handles=handles, title="Image long side:", loc="upper left", ncol=4,
               frameon=False, bbox_to_anchor=(0.005, 0.95), alignment="left",
               title_fontproperties={"size": 10}).get_title().set_color(INK_2)
    fig.suptitle("Qwen2.5-Omni-3B on Gleason2019: the image encoder carries Gleason signal "
                 "that the zero-shot answer does not use", x=0.01, ha="left", fontsize=12.5,
                 color=INK)
    fig.text(0.01, 0.005, "244 cores, leave-one-slide-out. Bars: 95% bootstrap intervals "
             "(cores resampled within slide). Colour statistics shown at 448 px only. "
             "Probes: vision tower in fp16, language model 4-bit (probe_v3).",
             fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.86))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "nll_overview.png", dpi=200)


def main() -> None:
    long = per_core_losses()
    ci, diffs = summarise(long)
    MET.mkdir(parents=True, exist_ok=True)
    ci.round(4).to_csv(MET / "nll_ci.csv", index=False)
    diffs.round(4).to_csv(MET / "paired_diffs.csv", index=False)
    plot(ci)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(diffs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
