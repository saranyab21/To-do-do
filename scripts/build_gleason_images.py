"""Write downscaled Gleason2019 cores for the resolution pilot.

    python scripts/build_gleason_images.py

Writes (git-ignored): data/processed/gleason2019/images_<size>/<core>.jpg and
data/processed/gleason2019/images_manifest.csv. A small aggregate summary goes to
results/metrics/gleason2019_images/summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import gleason2019 as g

PROC = ROOT / "data/processed/gleason2019"
MET = ROOT / "results/metrics/gleason2019_images"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments/configs/gleason2019_images.yaml"))
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())
    root = g.default_root()

    folders = ["train_imgs"] + (["test_imgs"] if cfg["include_test"] else [])
    jobs = [(p.stem, str(p), str(PROC), cfg["sizes"], cfg["crop_to_tissue"], cfg["jpeg_quality"])
            for f in folders for p in sorted((root / f).glob("*.jpg"))]
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        rows = [r for rs in pool.map(g.downscale_core, jobs, chunksize=4) for r in rs]

    man = pd.DataFrame(rows)
    man["path"] = man.path.map(lambda p: str(Path(p).relative_to(ROOT)).replace("\\", "/"))
    man.to_csv(PROC / "images_manifest.csv", index=False)

    MET.mkdir(parents=True, exist_ok=True)
    per_size = man.groupby("size").agg(
        n=("core", "size"), tokens_median=("vision_tokens", "median"),
        tokens_max=("vision_tokens", "max"), orig_px_per_px=("um_per_px_rel", "median"))
    crop_area = ((man.crop_x1 - man.crop_x0) * (man.crop_y1 - man.crop_y0)
                 / (man.orig_w * man.orig_h)).median()
    summary = {"config": cfg, "n_cores": int(man.core.nunique()),
               "median_kept_area_after_crop": round(float(crop_area), 3),
               "per_size": per_size.round(2).reset_index().to_dict(orient="records")}
    (MET / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
