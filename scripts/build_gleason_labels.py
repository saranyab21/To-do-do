"""Build Gleason2019 per-pathologist labels, rater distributions, folds and agreement stats.

    python scripts/build_gleason_labels.py [--refresh]

Writes (derived per-core data, git-ignored):
    data/processed/gleason2019/pixel_counts.csv      cached map pixel counts
    data/processed/gleason2019/rater_labels.csv      one row per (core, pathologist)
    data/processed/gleason2019/rater_distributions.csv
    data/processed/gleason2019/folds.csv             leave-one-slide-out folds
Writes (aggregate, committed):
    results/metrics/gleason2019_labels/summary.json, rater_agreement.csv, pairwise_kappa.csv
    results/figures/gleason2019_labels/agreement.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import gleason2019 as g
from src.evaluation.agreement import (
    krippendorff_alpha,
    leave_one_out_agreement,
    pairwise_weighted_kappa,
)

PROC = ROOT / "data/processed/gleason2019"
MET = ROOT / "results/metrics/gleason2019_labels"
FIG = ROOT / "results/figures/gleason2019_labels"
RANK = {0: 0, 3: 1, 4: 2, 5: 3}  # ordinal position of each answer code


def wide(labels: pd.DataFrame, dim: str) -> tuple[list[str], np.ndarray]:
    w = labels.pivot(index="core", columns="rater", values=dim).reindex(columns=range(1, 7))
    return list(w.index), w.to_numpy(dtype=float)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def image_sizes(root: Path, cores) -> pd.DataFrame:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    rows = []
    for c in cores:
        with Image.open(root / "train_imgs" / f"{c}.jpg") as im:
            rows.append({"core": c, "img_w": im.size[0], "img_h": im.size[1]})
    return pd.DataFrame(rows)


def agreement_block(labels: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    summary, per_rater, kappas = {}, [], []
    for dim in ("primary", "secondary"):
        _, r = wide(labels, dim)
        ranks = np.full_like(r, np.nan)
        for code, rank in RANK.items():
            ranks[r == code] = rank
        multi = (~np.isnan(r)).sum(1) >= 2
        all_same = np.array([len(set(row[~np.isnan(row)])) == 1 for row in r[multi]])
        summary[dim] = {
            "krippendorff_alpha_ordinal_ranks": krippendorff_alpha(ranks, "interval"),
            "krippendorff_alpha_nominal": krippendorff_alpha(r, "nominal"),
            "share_cores_all_raters_agree": float(all_same.mean()),
        }
        loo = leave_one_out_agreement(r)
        k = pairwise_weighted_kappa(ranks, levels=np.arange(4))
        for i in range(6):
            per_rater.append({
                "dimension": dim, "rater": i + 1,
                "n_cores_rated": int((~np.isnan(r[:, i])).sum()),
                "agreement_with_others_mode": loo[i],
                "mean_quadratic_kappa_vs_others": float(np.nanmean(np.delete(k[i], i))),
            })
        kd = pd.DataFrame(k, index=range(1, 7), columns=range(1, 7)).stack(future_stack=True)
        kappas.append(kd.rename("quadratic_kappa").reset_index()
                      .rename(columns={"level_0": "rater_a", "level_1": "rater_b"})
                      .assign(dimension=dim))
    return summary, pd.DataFrame(per_rater), pd.concat(kappas)


def plot(dist: pd.DataFrame, kappa: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    surface, ink, ink2 = "#fcfcfb", "#0b0b0b", "#52514e"
    ramp = {1: "#104281", 2: "#2a78d6", 3: "#86b6ef"}  # validated ordinal blue ramp
    plt.rcParams.update({"font.size": 11, "figure.facecolor": surface, "axes.facecolor": surface,
                         "savefig.facecolor": surface, "axes.edgecolor": "#b9b8b2",
                         "xtick.color": ink2, "ytick.color": ink, "axes.labelcolor": ink2})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})

    d = dist[dist.n_raters >= 2].assign(spread=lambda x: x.n_distinct.clip(upper=3))
    shares = d.groupby("dimension").spread.value_counts(normalize=True).unstack(fill_value=0)
    shares = shares.reindex(index=["secondary", "primary"], columns=[1, 2, 3], fill_value=0)
    labels = {1: "all pathologists agree", 2: "2 different ratings", 3: "3 or more"}
    left = np.zeros(2)
    for lvl in (1, 2, 3):
        w = shares[lvl].to_numpy()
        a1.barh([0, 1], w, left=left, height=0.55, color=ramp[lvl], edgecolor=surface,
                linewidth=2, label=labels[lvl])
        for y, l0, wi in zip([0, 1], left, w):
            if wi >= 0.08:
                a1.text(l0 + wi / 2, y, f"{wi:.0%}", ha="center", va="center", fontsize=10,
                        color="#ffffff" if lvl < 3 else ink)
        left += w
    a1.set_yticks([0, 1], ["secondary pattern", "primary pattern"])
    a1.set_xlim(0, 1)
    a1.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    a1.set_xlabel(f"share of cores (n = {d.core.nunique()})")
    for s in ("top", "right", "left"):
        a1.spines[s].set_visible(False)
    a1.tick_params(axis="y", length=0)
    a1.legend(loc="upper left", bbox_to_anchor=(0, 1.22), ncol=3, frameon=False, fontsize=9.5)
    a1.set_title("How many different ratings each core gets", loc="left", fontsize=11,
                 color=ink, pad=34)

    kp = kappa[kappa.dimension == "primary"].pivot(index="rater_a", columns="rater_b",
                                                    values="quadratic_kappa")
    im = a2.imshow(kp.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            v = kp.iat[i, j]
            if np.isfinite(v):
                a2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if v > 0.6 else ink)
    a2.set_xticks(range(6), [f"P{i}" for i in range(1, 7)])
    a2.set_yticks(range(6), [f"P{i}" for i in range(1, 7)])
    a2.tick_params(length=0)
    for s in a2.spines.values():
        s.set_visible(False)
    a2.set_title("Primary pattern: pairwise weighted kappa", loc="left", fontsize=11, color=ink)
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    fig.tight_layout(w_pad=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/gleason2019_labels.yaml"))
    ap.add_argument("--refresh", action="store_true", help="re-decode the label maps")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = g.default_root()
    PROC.mkdir(parents=True, exist_ok=True)
    MET.mkdir(parents=True, exist_ok=True)

    cache = PROC / "pixel_counts.csv"
    if cache.exists() and not args.refresh:
        pix = pd.read_csv(cache)
    else:
        pix = g.pixel_table(g.index(root), workers=cfg["workers"])
        pix.to_csv(cache, index=False)

    sizes = image_sizes(root, sorted(pix.core.unique()))
    pix = pix.merge(sizes, on="core")
    shape_mismatch = pix[(pix.map_h != pix.img_h) | (pix.map_w != pix.img_w)]

    labels = g.rater_labels(pix, cfg["min_pattern_frac"], cfg["whole_core_6"])
    alt_rule = "missing" if cfg["whole_core_6"] == "benign" else "benign"
    alt = g.rater_labels(pix, cfg["min_pattern_frac"], alt_rule)
    changed = int(((labels.primary.fillna(-1) != alt.primary.fillna(-1)) |
                   (labels.secondary.fillna(-1) != alt.secondary.fillna(-1))).sum())

    dist = g.rater_distributions(labels)
    folds = g.slide_folds(labels.core)
    keep = ["core", "slide", "rater", "status", "primary", "secondary",
            "frac_1", "frac_3", "frac_4", "frac_5", "frac_6"]
    labels[keep].to_csv(PROC / "rater_labels.csv", index=False)
    dist.to_csv(PROC / "rater_distributions.csv", index=False)
    folds.to_csv(PROC / "folds.csv", index=False)

    agree, per_rater, kappa = agreement_block(labels)
    per_rater.to_csv(MET / "rater_agreement.csv", index=False, float_format="%.4f")
    kappa.to_csv(MET / "pairwise_kappa.csv", index=False, float_format="%.4f")

    code_counts = {dim: labels[dim].dropna().astype(int).value_counts().sort_index()
                   .rename(index=str).to_dict() for dim in ("primary", "secondary")}
    summary = {
        "config": cfg,
        "git_commit": git_commit(),
        "n_cores": int(labels.core.nunique()),
        "n_ratings": len(labels),
        "raters_per_core": labels.groupby("core").rater.size().value_counts().sort_index()
        .rename(index=str).to_dict(),
        "status_counts": labels.status.value_counts().to_dict(),
        "rating_counts_by_code": code_counts,
        "cores_per_fold": folds.groupby("slide").size().to_dict(),
        "map_image_shape_mismatches": len(shape_mismatch),
        "sensitivity_whole_core_6": {"alternative_rule": alt_rule,
                                     "ratings_changed": changed},
        "agreement": agree,
        "entropy_mean": dist.groupby("dimension").entropy.mean().round(4).to_dict(),
    }
    (MET / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    plot(dist, kappa, FIG / "agreement.png")
    print(json.dumps({k: summary[k] for k in summary if k not in ("config",)}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
