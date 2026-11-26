"""Soft-label linear probe on frozen features, nested leave-one-slide-out (CPU).

    python scripts/run_gleason_probe.py

Reads data/processed/gleason2019/features/<run_name>/features_<size>.npz (from
extract_gleason_features.py). Writes out-of-fold predictions (git-ignored) to
data/processed/gleason2019/features/<run_name>/predictions_<probe_name>.csv and
aggregate metrics (tracked) to results/metrics/gleason2019_probe/<probe_name>/metrics.csv
and summary.json. probe_name defaults to run_name.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.probe import nested_loso
from src.evaluation.soft_label import evaluate
from src.utils.provenance import git_state

PROC = ROOT / "data/processed/gleason2019"
MET = ROOT / "results/metrics/gleason2019_probe"


def run_task(task: tuple) -> pd.DataFrame:
    size, feature, dim, X, P, folds, cores, C_grid, n_comp, codes = task
    logq, chosen = nested_loso(X, P, folds, C_grid, n_comp)
    out = pd.DataFrame(logq, columns=[f"lp_{c}" for c in codes])
    return out.assign(core=cores, size=size, feature=feature, dimension=dim,
                      C_chosen=[chosen[int(f)] for f in folds])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/gleason2019_probe.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    codes = yaml.safe_load((ROOT / cfg["model_config"]).read_text())["codes"]
    fdir = PROC / "features" / cfg["run_name"]
    dist = pd.read_csv(PROC / "rater_distributions.csv").query("n_raters > 0")
    folds = pd.read_csv(PROC / "folds.csv").set_index("core").fold

    tasks = []
    for size in cfg["sizes"]:
        path = fdir / f"features_{size}.npz"
        if not path.exists():
            print(f"skip {size}px: {path.name} not found")
            continue
        npz = np.load(path)
        cores = npz["cores"].astype(str)
        for dim in ("primary", "secondary"):
            d = dist[dist.dimension == dim].set_index("core")
            ok = np.isin(cores, d.index)
            P = d.loc[cores[ok], [f"n_{c}" for c in codes]].to_numpy(float)
            P /= P.sum(1, keepdims=True)
            fo = folds.loc[cores[ok]].to_numpy()
            feats = {k: npz[k][ok].astype(np.float64) for k in npz.files if k != "cores"}
            # negative control: pure noise. Its scores show what "no signal" looks like
            # under LOSO (fold-to-fold prior shifts can push AUROC well below 0.5).
            feats["random_control"] = np.random.default_rng(0).normal(size=(ok.sum(), 64))
            for feature, X in feats.items():
                tasks.append((size, feature, dim, X, P, fo, cores[ok], cfg["C_grid"],
                              cfg["pca_components"], codes))
    if not tasks:
        sys.exit("no features found; run extract_gleason_features.py first")
    print(f"{len(tasks)} probes (size x feature x dimension), nested LOSO")
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        preds = pd.concat(pool.map(run_task, tasks), ignore_index=True)
    name = cfg.get("probe_name", cfg["run_name"])
    preds.to_csv(fdir / f"predictions_{name}.csv", index=False)

    df = preds.merge(dist, on=["core", "dimension"]).merge(
        folds.reset_index()[["core", "fold"]], on="core")
    metrics = pd.concat([evaluate(g, codes).assign(feature=f) for f, g in df.groupby("feature")])
    first = ["feature", "size", "dimension", "n", "nll", "nll_loso_prior", "top_match",
             "auroc_cancer", "spearman_grade"]
    metrics = metrics[first + [c for c in metrics.columns if c not in first]]
    metrics = metrics.sort_values(["dimension", "nll"]).reset_index(drop=True)

    met = MET / name
    met.mkdir(parents=True, exist_ok=True)
    metrics.round(4).to_csv(met / "metrics.csv", index=False)
    C_counts = preds.groupby("C_chosen").core.size() / preds.groupby("C_chosen").core.size().sum()
    summary = {"config": cfg, "git": git_state(), "n_probes": len(tasks),
               "C_chosen_share": {str(k): round(float(v), 3) for k, v in C_counts.items()}}
    (met / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(metrics[first + ["kl", "pred_share_0", "pred_share_3", "pred_share_4",
                               "pred_share_5"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
