"""Run the E0 simulation and write aggregate results.

Usage:
    python scripts/run_e0.py --config experiments/configs/e0.yaml [--reps N]

Outputs (simulated data only, safe to commit):
    results/metrics/e0_reps.csv      one row per replicate
    results/metrics/e0_summary.csv   coverage / decision rates per scenario
    results/metrics/e0_run_info.json config, git commit, runtime
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# one BLAS thread per worker process; parallelism comes from running scenarios side by side
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.deference import center
from src.simulation.e0 import (
    Design,
    Scenario,
    World,
    decision_outcome,
    make_reference,
    population_beta,
    run_replicate,
)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def run_scenario(job) -> list[dict]:
    """Run all replicates of one (design, scenario); executed in a worker process."""
    dname, dcfg, sc, cfg, reps, seed = job
    misspec = sc.get("misspec", False)
    design = Design(
        name=dname,
        K=dcfg["K"],
        n_cases=dcfg["n_cases"],
        n_dims=dcfg["n_dims"],
        rater_reliabilities=tuple(dcfg["rater_reliabilities"]),
        raters_per_case=tuple(dcfg["raters_per_case"]),
        pairs_per_item=dcfg["pairs_per_item"],
        error_model="ordinal" if misspec else "uniform",
        stated_reliability="coarse" if misspec else "true",
    )
    scen = Scenario(
        name=sc["name"],
        design=design,
        kappa=sc["kappa"],
        plant=sc.get("plant", "reference"),
        noise_frac=cfg["noise_frac"],
    )
    rng = np.random.default_rng(seed)
    world = World(design, rng)
    refm = make_reference(cfg.get("reference", "linear"), cfg.get("reference_penalty", "firth"))
    n_pop = max(cfg["population_cases"] // design.n_dims, 500)
    beta_pop = population_beta(world, rng, n_cases=n_pop, ref=refm)
    probe = world.sample(rng, n_cases=200)
    rms = np.sqrt(np.mean(center(refm.update(probe, beta_pop)) ** 2))
    noise_scale = scen.noise_frac * rms

    t_s = time.time()
    rows = []
    for r in range(reps):
        row = run_replicate(
            scen, world, beta_pop, noise_scale, rng,
            n_boot=cfg["n_boot"], tau_cos=cfg["tau_cos"], alpha=cfg["alpha"], ref=refm,
            interval=cfg.get("bootstrap_interval", "percentile"),
        )
        row["rep"] = r
        row["beta_pop"] = json.dumps(np.round(beta_pop, 4).tolist())
        rows.append(row)
    print(f"{dname:8s} {scen.name:16s} {reps} reps  {time.time() - t_s:6.1f}s", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/e0.yaml"))
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results/metrics"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    reps = args.reps or cfg["reps"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root_rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()

    jobs = [
        (dname, dcfg, sc, cfg, reps, int(root_rng.integers(2**63)))
        for dname, dcfg in cfg["designs"].items()
        for sc in cfg["scenarios"]
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for scen_rows in pool.map(run_scenario, jobs):
            rows.extend(scen_rows)

    df = pd.DataFrame(rows)
    df["outcome"] = [decision_outcome(k, lab) for k, lab in zip(df.planted_kappa, df.label)]
    df.to_csv(out / "e0_reps.csv", index=False)

    g = df.groupby(["design", "scenario"], sort=False)
    summary = g.agg(
        planted_kappa=("planted_kappa", "first"),
        reps=("rep", "size"),
        n_items=("n_items", "mean"),
        kappa_mean=("kappa_hat", "mean"),
        kappa_sd=("kappa_hat", "std"),
        coverage=("covered", "mean"),
        coverage_fixed_ref=("covered_fixed", "mean"),
        coverage_alt_interval=("covered_alt", "mean"),
        boot_mean_kappa=("boot_mean_kappa", "mean"),
        ci_width=("kappa_hi", "mean"),
        cos_mean=("cos", "mean"),
        rho_size_mean=("rho_size", "mean"),
        adequacy_cos=("adequacy_cos_r_pop_vs_bayes", "mean"),
    )
    summary["ci_width"] = g.apply(lambda d: (d.kappa_hi - d.kappa_lo).mean(), include_groups=False)
    summary["ci_width_fixed_ref"] = g.apply(
        lambda d: (d.kappa_hi_fixed - d.kappa_lo_fixed).mean(), include_groups=False
    )
    summary["error_rate_fixed_ref"] = g.apply(
        lambda d: np.mean(
            [decision_outcome(k, lab) == "error" for k, lab in zip(d.planted_kappa, d.label_fixed)]
        ),
        include_groups=False,
    )
    summary["error_rate_alt_interval"] = g.apply(
        lambda d: np.mean(
            [decision_outcome(k, lab) == "error" for k, lab in zip(d.planted_kappa, d.label_alt)]
        ),
        include_groups=False,
    )
    summary["bias"] = summary.kappa_mean - summary.planted_kappa
    for lab in ("misdirected", "over", "under", "calibrated", "inconclusive"):
        summary[f"p_{lab}"] = g.apply(lambda d, lab=lab: (d.label == lab).mean(), include_groups=False)
    summary["decision_error_rate"] = g.apply(lambda d: (d.outcome == "error").mean(), include_groups=False)
    summary["power"] = g.apply(lambda d: (d.outcome == "correct").mean(), include_groups=False)
    summary = summary.reset_index()
    summary.to_csv(out / "e0_summary.csv", index=False, float_format="%.4f")

    info = {
        "config": cfg,
        "config_path": (
            str(Path(args.config).resolve().relative_to(ROOT))
            if Path(args.config).resolve().is_relative_to(ROOT)
            else str(Path(args.config).resolve())
        ),
        "reps": reps,
        "git_commit": git_commit(),
        "runtime_s": round(time.time() - t0, 1),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    (out / "e0_run_info.json").write_text(json.dumps(info, indent=2) + "
")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
