"""Zero-shot Qwen2.5-Omni scoring of Gleason2019 cores (GPU machine, e.g. CIP cip4a0).

    python scripts/run_gleason_zeroshot.py --limit 3 --sizes 448     # smoke test
    python scripts/run_gleason_zeroshot.py                           # full run (resumable)
    python scripts/run_gleason_zeroshot.py --eval-only               # metrics from saved scores

Per-core scores (git-ignored): data/processed/gleason2019/zeroshot/<run_name>/scores.jsonl
plus run_info.json. Aggregate metrics (tracked): results/metrics/gleason2019_zeroshot/
<run_name>/metrics.csv and summary.json. Re-running skips rows already in scores.jsonl.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.soft_label import evaluate

PROC = ROOT / "data/processed/gleason2019"
MET = ROOT / "results/metrics/gleason2019_zeroshot"


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                              check=False).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line])


def run_eval(cfg: dict, out: Path) -> None:
    scores = load_scores(out / "scores.jsonl")
    if scores.empty:
        print("no scores yet")
        return
    dist = pd.read_csv(PROC / "rater_distributions.csv")
    folds = pd.read_csv(PROC / "folds.csv")[["core", "fold"]]
    df = scores.merge(dist, on=["core", "dimension"]).merge(folds, on="core")
    metrics = evaluate(df, cfg["codes"])
    met = MET / cfg["run_name"]
    met.mkdir(parents=True, exist_ok=True)
    metrics.round(4).to_csv(met / "metrics.csv", index=False)
    summary = {"run_name": cfg["run_name"], "n_scores": len(scores),
               "n_cores": int(scores.core.nunique()),
               "n_nonfinite": int((~scores.finite.astype(bool)).sum()),
               "top_tokens": scores.top_token.value_counts().head(10).to_dict(),
               "seconds_per_pass_median": round(float(scores.seconds.median()), 2)}
    (met / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(metrics.round(3).to_string(index=False))
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/gleason2019_zeroshot.yaml"))
    ap.add_argument("--sizes", type=int, nargs="+", help="override config sizes")
    ap.add_argument("--limit", type=int, help="score only the first N cores (smoke test)")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = PROC / "zeroshot" / cfg["run_name"]
    if args.eval_only:
        run_eval(cfg, out)
        return

    out.mkdir(parents=True, exist_ok=True)
    info_path = out / "run_info.json"
    if info_path.exists():  # resuming: the scored question must not have changed
        old = json.loads(info_path.read_text())["config"]
        for k in ("model_id", "quant", "codes", "prompts"):
            if old[k] != cfg[k]:
                sys.exit(f"config '{k}' differs from the existing run; use a new run_name")

    import torch
    import transformers
    from PIL import Image

    from src.models.zeroshot import build_messages, code_token_ids, load_thinker, score_codes

    model, processor = load_thinker(cfg["model_id"], cfg["quant"], cfg.get("revision"))
    code_ids = code_token_ids(processor.tokenizer, cfg["codes"])
    info = {"config": cfg, "git": git_state(), "code_token_ids": code_ids,
            "model_commit": getattr(model.config, "_commit_hash", None),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "host": platform.node(), "started": datetime.now(UTC).isoformat(timespec="seconds")}
    info_path.write_text(json.dumps(info, indent=2) + "\n")

    man = pd.read_csv(PROC / "images_manifest.csv")
    rated = set(pd.read_csv(PROC / "rater_distributions.csv").query("n_raters > 0").core)
    cores = sorted(rated & set(man.core))[: args.limit]
    sizes = args.sizes or cfg["sizes"]
    done = load_scores(out / "scores.jsonl")
    done_keys = set(zip(done.core, done["size"], done.dimension)) if len(done) else set()
    todo = [(c, s) for s in sorted(sizes) for c in cores
            if any((c, s, d) not in done_keys for d in cfg["dimensions"])]
    print(f"{len(cores)} cores x {len(sizes)} sizes; {len(todo)} images to score")

    path_of = {(r.core, r.size): ROOT / r.path for r in man.itertuples()}
    with open(out / "scores.jsonl", "a") as f:
        for k, (core, size) in enumerate(todo, 1):
            image = Image.open(path_of[(core, size)]).convert("RGB")
            for dim in cfg["dimensions"]:
                if (core, size, dim) in done_keys:
                    continue
                t0 = time.perf_counter()
                row = score_codes(model, processor, build_messages(cfg["prompts"], dim, image),
                                  image, code_ids)
                row.update(core=core, size=size, dimension=dim,
                           seconds=round(time.perf_counter() - t0, 3))
                f.write(json.dumps(row) + "\n")
                f.flush()
            if k % 10 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}] {core} @ {size}px  last: top={row['top_token']!r} "
                      f"mass={row['mass']:.3f}  {row['seconds']:.1f}s", flush=True)
    run_eval(cfg, out)


if __name__ == "__main__":
    main()
