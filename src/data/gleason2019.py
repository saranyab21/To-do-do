"""Gleason2019 per-pathologist core labels (Pillar 1 targets).

Raw layout (see docs/DATA.md):
    data/raw/gleason2019/train_imgs/<core>.jpg
    data/raw/gleason2019/maps<k>/<core>_classimg_nonconvex.png     k = 1..6

Map pixel values: 0 unannotated/background, 1 benign, 3/4/5 Gleason patterns, 6 an
undocumented value used by two pathologists only (pathologist 1 on whole cores,
pathologist 2 on small regions mostly left unannotated by the others).

Label rule (per pathologist, never fused across pathologists):
  * tissue = pixels labelled 1, 3, 4, 5 or 6
  * a cancer pattern counts if it covers >= min_pattern_frac of the tissue (YAAGGS-style
    noise floor for contour slivers)
  * primary = largest counted pattern, secondary = second largest, or = primary when only
    one pattern is present (e.g. 3+3)
  * no counted cancer pattern -> benign, coded 0 for both dimensions
Label 6 is non-cancer under any reading, so it never changes a pattern label; it only
enters the tissue denominator. A rating with no tissue other than 6 is "benign" by default
or missing (option whole_core_6="missing", for the sensitivity check).
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CODES = (0, 3, 4, 5)  # answer codes: benign, Gleason pattern 3, 4, 5
PATTERNS = (3, 4, 5)
TISSUE_VALUES = (1, 3, 4, 5, 6)
N_RATERS = 6
MAP_SUFFIX = "_classimg_nonconvex.png"

# maps without a training image (two orphans) or belonging to a test image (leak)
EXCLUDED_MAPS = {"slide007_core137", "slide007_core145", "slide006_core137"}


def slide_of(core: str) -> str:
    return re.match(r"(slide\d+)_core\d+", core).group(1)


def index(root: str | Path) -> pd.DataFrame:
    """One row per (core, rater) map that belongs to a training image."""
    root = Path(root)
    cores = sorted(p.stem for p in (root / "train_imgs").glob("*.jpg"))
    rows = []
    for c in cores:
        for k in range(1, N_RATERS + 1):
            p = root / f"maps{k}" / f"{c}{MAP_SUFFIX}"
            if p.exists() and c not in EXCLUDED_MAPS:
                rows.append({"core": c, "slide": slide_of(c), "rater": k, "map_path": str(p)})
    return pd.DataFrame(rows)


def pixel_counts(map_path: str) -> dict:
    """Pixel count per map value (0..6) plus map shape."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    m = np.asarray(Image.open(map_path))
    counts = np.bincount(m.ravel(), minlength=8)
    out = {f"px_{v}": int(counts[v]) for v in range(7)}
    out["px_other"] = int(counts[7:].sum())
    out["map_h"], out["map_w"] = m.shape[:2]
    return out


def pixel_table(idx: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        stats = list(pool.map(pixel_counts, idx.map_path, chunksize=8))
    return pd.concat([idx.reset_index(drop=True), pd.DataFrame(stats)], axis=1)


def core_label(row: pd.Series | dict, min_pattern_frac: float = 0.01,
               whole_core_6: str = "benign") -> tuple[float, float, str]:
    """(primary, secondary, status) for one pathologist's map counts."""
    tissue = sum(row[f"px_{v}"] for v in TISSUE_VALUES)
    if tissue == 0:
        return np.nan, np.nan, "no_tissue"
    if row["px_6"] == tissue and whole_core_6 == "missing":
        return np.nan, np.nan, "only_label6"
    frac = {p: row[f"px_{p}"] / tissue for p in PATTERNS}
    present = sorted((p for p in PATTERNS if frac[p] >= min_pattern_frac),
                     key=lambda p: (-frac[p], -p))  # ties -> higher grade first
    if not present:
        status = "only_label6" if row["px_6"] == tissue else "benign"
        return 0, 0, status
    primary = present[0]
    secondary = present[1] if len(present) > 1 else primary
    return primary, secondary, "graded"


def rater_labels(pix: pd.DataFrame, min_pattern_frac: float = 0.01,
                 whole_core_6: str = "benign") -> pd.DataFrame:
    lab = pix.apply(lambda r: core_label(r, min_pattern_frac, whole_core_6),
                    axis=1, result_type="expand")
    lab.columns = ["primary", "secondary", "status"]
    out = pd.concat([pix, lab], axis=1)
    tissue = sum(out[f"px_{v}"] for v in TISSUE_VALUES)
    for v in (1, 3, 4, 5, 6):
        out[f"frac_{v}"] = np.where(tissue > 0, out[f"px_{v}"] / tissue.clip(lower=1), np.nan)
    return out


def rater_distributions(labels: pd.DataFrame) -> pd.DataFrame:
    """Per core and dimension: rating counts over CODES, n raters, entropy, mode."""
    rows = []
    for (core, slide), g in labels.groupby(["core", "slide"]):
        for dim in ("primary", "secondary"):
            v = g[dim].dropna().astype(int)
            counts = {c: int((v == c).sum()) for c in CODES}
            n = sum(counts.values())
            p = np.array([counts[c] for c in CODES], float) / max(n, 1)
            nz = p[p > 0]
            rows.append({
                "core": core, "slide": slide, "dimension": dim, "n_raters": n,
                **{f"n_{c}": counts[c] for c in CODES},
                "n_distinct": int((p > 0).sum()),
                "entropy": float(-(nz * np.log(nz)).sum() / np.log(len(CODES))) if n else np.nan,
                "mode": CODES[int(np.argmax(p))] if n else np.nan,
                "mode_share": float(p.max()) if n else np.nan,
            })
    return pd.DataFrame(rows)


def slide_folds(cores: pd.Series) -> pd.DataFrame:
    """Leave-one-slide-out folds: fold index = position of the slide in sorted order."""
    slides = sorted({slide_of(c) for c in cores})
    return pd.DataFrame({"core": sorted(set(cores))}).assign(
        slide=lambda d: d.core.map(slide_of),
        fold=lambda d: d.slide.map({s: i for i, s in enumerate(slides)}),
    )


def default_root() -> Path:
    return Path(os.environ.get("GLEASON2019_ROOT") or
                Path(__file__).resolve().parents[2] / "data" / "raw" / "gleason2019")
