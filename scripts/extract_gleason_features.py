"""Frozen Qwen2.5-Omni features + colour statistics per Gleason2019 core (GPU machine).

    python scripts/extract_gleason_features.py --limit 3 --sizes 448   # smoke test
    python scripts/extract_gleason_features.py                         # all sizes

Writes (git-ignored) data/processed/gleason2019/features/<run_name>/features_<size>.npz
(one array per feature set, rows aligned with the 'cores' array) and run_info.json.
A size whose file already exists is skipped (delete it to redo).
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

from src.utils.provenance import git_state

PROC = ROOT / "data/processed/gleason2019"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/gleason2019_probe.yaml"))
    ap.add_argument("--sizes", type=int, nargs="+")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = yaml.safe_load((ROOT / cfg["model_config"]).read_text())
    out = PROC / "features" / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    from PIL import Image

    from src.models.features import colour_features, image_token_id, qwen_features
    from src.models.zeroshot import build_messages, load_thinker

    sizes = [s for s in (args.sizes or cfg["sizes"]) if not (out / f"features_{s}.npz").exists()]
    if not sizes:
        print("all sizes done")
        return
    model, processor = load_thinker(mcfg["model_id"], mcfg["quant"], mcfg.get("revision"))
    img_tok = image_token_id(model, processor)
    info = {"config": cfg, "model_config": mcfg, "git": git_state(), "image_token_id": img_tok,
            "model_commit": getattr(model.config, "_commit_hash", None),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "host": platform.node(), "started": datetime.now(UTC).isoformat(timespec="seconds")}
    (out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")

    man = pd.read_csv(PROC / "images_manifest.csv")
    rated = set(pd.read_csv(PROC / "rater_distributions.csv").query("n_raters > 0").core)
    cores = sorted(rated & set(man.core))[: args.limit]
    path_of = {(r.core, r.size): ROOT / r.path for r in man.itertuples()}
    for size in sorted(sizes):
        feats: dict[str, list] = {}
        t0 = time.perf_counter()
        for k, core in enumerate(cores, 1):
            image = Image.open(path_of[(core, size)]).convert("RGB")
            msgs = build_messages(mcfg["prompts"], cfg["prompt_dimension"], image)
            row = qwen_features(model, processor, msgs, image, cfg["layers"], img_tok)
            row["colour"] = colour_features(np.asarray(image))
            for name, v in row.items():
                feats.setdefault(name, []).append(v)
            if k % 25 == 0 or k == len(cores):
                print(f"[{size}px {k}/{len(cores)}] {time.perf_counter() - t0:.0f}s", flush=True)
        arrays = {n: np.stack(v).astype(np.float32 if n == "colour" else np.float16)
                  for n, v in feats.items()}
        np.savez_compressed(out / f"features_{size}.npz", cores=np.array(cores), **arrays)
        print(f"saved features_{size}.npz: " +
              ", ".join(f"{n} {a.shape[1]}" for n, a in arrays.items()))


if __name__ == "__main__":
    main()
