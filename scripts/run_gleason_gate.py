"""Pillar-1 competence and grounding gate on Gleason2019 (see src/evaluation/gate.py).

    python scripts/run_gleason_gate.py --limit 3          # GPU smoke test
    python scripts/run_gleason_gate.py                    # GPU scoring (resumable) + analysis
    python scripts/run_gleason_gate.py --analyse-only     # CPU: tests, tables, figures

Per-core scores (git-ignored): data/processed/gleason2019/gate/<run_name>/scores.jsonl,
partners.csv, run_info.json. Aggregates (tracked): results/metrics/gleason2019_gate/
<run_name>/ and figures in results/figures/gleason2019_gate/<run_name>/.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import gleason2019 as g
from src.utils.provenance import git_state

PROC = ROOT / "data/processed/gleason2019"


def load_cfg(path: str) -> tuple[dict, dict, dict]:
    cfg = yaml.safe_load(Path(path).read_text())
    mcfg = yaml.safe_load((ROOT / cfg["model_config"]).read_text())
    lcfg = yaml.safe_load((ROOT / cfg["labels_config"]).read_text())
    for d, spec in cfg["dimensions"].items():
        spec.setdefault("question", mcfg["prompts"]["questions"].get(d))
        if not spec["question"]:
            sys.exit(f"dimension {d!r} has no question")
    return cfg, mcfg, lcfg


def distributions(lcfg: dict) -> pd.DataFrame:
    labels = pd.read_csv(PROC / "rater_labels.csv")
    long = g.rating_dimensions(labels, lcfg["min_pattern_frac"])
    return g.dimension_distributions(long)


def shuffle_partners(cores: list[str], seed: int) -> dict[str, str]:
    """Each core gets the image of a random core from another slide."""
    rng = np.random.default_rng(seed)
    slides = {c: g.slide_of(c) for c in cores}
    return {c: str(rng.choice([o for o in cores if slides[o] != slides[c]])) for c in cores}


def modal_codes(dist: pd.DataFrame, dimension: str) -> dict[str, set[int]]:
    """Per core, the set of most common pathologist answers (ties kept)."""
    d = dist[dist.dimension == dimension].set_index("core")
    n = d[[f"n_{c}" for c in range(6)]].to_numpy()
    return {c: set(np.flatnonzero(row == row.max()).tolist()) for c, row in zip(d.index, n)}


def hard_negative_partners(cores: list[str], X: np.ndarray, modes: dict[str, set[int]]
                           ) -> dict[str, str | None]:
    """Most similar core (cosine on standardised features, rows aligned with cores) from
    another slide whose majority answer shares nothing with this core's."""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    sim = Z @ Z.T
    slides = [g.slide_of(c) for c in cores]
    out = {}
    for i, c in enumerate(cores):
        ok = [j for j, o in enumerate(cores)
              if slides[j] != slides[i] and modes[o].isdisjoint(modes[c])]
        out[c] = cores[max(ok, key=lambda j: sim[i, j])] if ok else None
    return out


def frozen_part(cfg: dict) -> dict:
    """What a resumed run must share with the original: everything already scored."""
    return {"size": cfg["size"], "blank_rgb": cfg["blank_rgb"], "seed": cfg["seed"],
            "dims": {d: (s["codes"], s["question"]) for d, s in cfg["dimensions"].items()}}


def variants(cfg: dict, cond: str) -> list[str]:
    """Question wordings scored under an image condition: q0 is the original wording;
    p1, p2, ... are paraphrases, scored on the real image only."""
    n = max((len(s.get("paraphrases", [])) for s in cfg["dimensions"].values()), default=0)
    return ["q0"] + [f"p{k}" for k in range(1, n + 1)] if cond == "real" else ["q0"]


def question(spec: dict, variant: str) -> str | None:
    if variant == "q0":
        return spec["question"]
    k = int(variant[1:]) - 1
    para = spec.get("paraphrases", [])
    return para[k] if k < len(para) else None


def score(cfg, mcfg, lcfg, out: Path, limit: int | None) -> None:
    import torch
    import transformers
    from PIL import Image

    from src.models.zeroshot import build_messages, code_token_ids, load_thinker, score_codes

    info_path = out / "run_info.json"
    if info_path.exists():  # additions (conditions, paraphrases) are fine; changes are not
        old, new = frozen_part(json.loads(info_path.read_text())["config"]), frozen_part(cfg)
        changed = [k for k in ("size", "blank_rgb", "seed") if old[k] != new[k]]
        changed += [d for d, v in old["dims"].items() if new["dims"].get(d) != v]
        if changed:
            sys.exit(f"{changed} differ from the existing run; use a new run_name")

    man = pd.read_csv(PROC / "images_manifest.csv").query("size == @cfg['size']")
    dist = distributions(lcfg)
    cores = sorted(set(dist.core) & set(man.core))
    partners = shuffle_partners(cores, cfg["seed"])  # from all cores, before --limit
    hard = {}
    if "hard_negative" in cfg["conditions"]:
        hn = cfg["hard_negative"]
        npz = np.load(ROOT / hn["features"])
        row_of = {c: i for i, c in enumerate(npz["cores"].astype(str))}
        missing = [c for c in cores if c not in row_of]
        if missing:
            sys.exit(f"{len(missing)} cores lack {hn['feature']} features in {hn['features']}")
        X = npz[hn["feature"]][[row_of[c] for c in cores]].astype(np.float64)
        hard = hard_negative_partners(cores, X, modal_codes(dist, hn["differ_on"]))
    pd.DataFrame({"partner": partners, "hard_partner": pd.Series(hard, dtype=object)}
                 ).rename_axis("core").to_csv(out / "partners.csv")
    cores = cores[:limit]
    path_of = dict(zip(man.core, man.path))

    model, processor = load_thinker(mcfg["model_id"], mcfg["quant"], mcfg.get("revision"))
    code_ids = {d: code_token_ids(processor.tokenizer, s["codes"])
                for d, s in cfg["dimensions"].items()}
    prompts = {v: {"system": mcfg["prompts"]["system"], "context": mcfg["prompts"]["context"],
                   "questions": {d: question(s, v) for d, s in cfg["dimensions"].items()}}
               for v in variants(cfg, "real")}
    info = {"config": cfg, "model_config": mcfg, "git": git_state(), "code_token_ids": code_ids,
            "model_commit": getattr(model.config, "_commit_hash", None),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "host": platform.node(), "started": datetime.now(UTC).isoformat(timespec="seconds")}
    info_path.write_text(json.dumps(info, indent=2) + "\n")

    done = set()
    if (out / "scores.jsonl").exists():
        for line in (out / "scores.jsonl").read_text().splitlines():
            r = json.loads(line)
            done.add((r["core"], r["dimension"], r["condition"], r.get("variant", "q0")))

    def wanted(c, cond):
        return [(d, v) for v in variants(cfg, cond) for d, s in cfg["dimensions"].items()
                if question(s, v) and (c, d, cond, v) not in done
                and not (cond == "hard_negative" and hard.get(c) is None)]

    todo = [(c, cond) for c in cores for cond in cfg["conditions"] if wanted(c, cond)]
    n_pass = sum(len(wanted(c, cond)) for c, cond in todo)
    print(f"{len(cores)} cores x {len(cfg['conditions'])} conditions x "
          f"{len(cfg['dimensions'])} questions (+ paraphrases on real images); "
          f"{len(todo)} images, {n_pass} forward passes to do")

    def image_for(core, cond):
        real = Image.open(ROOT / path_of[core]).convert("RGB")
        if cond == "real":
            return real
        if cond == "shuffled":
            return Image.open(ROOT / path_of[partners[core]]).convert("RGB")
        if cond == "hard_negative":
            return Image.open(ROOT / path_of[hard[core]]).convert("RGB")
        return Image.new("RGB", real.size, tuple(cfg["blank_rgb"]))

    t_start = time.perf_counter()
    with open(out / "scores.jsonl", "a") as f:
        for k, (core, cond) in enumerate(todo, 1):
            image = image_for(core, cond)
            for d, v in wanted(core, cond):
                t0 = time.perf_counter()
                row = score_codes(model, processor, build_messages(prompts[v], d, image), image,
                                  code_ids[d])
                partner = {"shuffled": partners[core], "hard_negative": hard.get(core)}.get(cond)
                row.update(core=core, dimension=d, condition=cond, variant=v, partner=partner,
                           seconds=round(time.perf_counter() - t0, 3))
                f.write(json.dumps(row) + "\n")
                f.flush()
            if k % 25 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}] {time.perf_counter() - t_start:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/gleason2019_gate.yaml"))
    ap.add_argument("--limit", type=int, help="score only the first N cores (smoke test)")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    cfg, mcfg, lcfg = load_cfg(args.config)
    out = PROC / "gate" / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    if not args.analyse_only:
        score(cfg, mcfg, lcfg, out, args.limit)

    from src.evaluation.gate_report import analyse

    analyse(cfg, distributions(lcfg), out)


if __name__ == "__main__":
    main()
