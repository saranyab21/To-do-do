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


# Query dimensions for the query-conditioned assessor. Each is derived from the same
# per-pathologist map, so every dimension has its own distribution of pathologist answers.
DIMENSION_CODES = {
    "cancer": (0, 1),                  # any counted Gleason pattern
    "primary": CODES,
    "secondary": CODES,
    "highest": CODES,                  # worst counted pattern (>= min_pattern_frac)
    "grade_group": (0, 1, 2, 3, 4, 5),  # ISUP grade group from primary + secondary
}


def grade_group(primary: int, secondary: int) -> int:
    """ISUP grade group (Epstein et al., 2016); 0 = benign."""
    if primary == 0:
        return 0
    score = primary + secondary
    if score <= 6:
        return 1
    if score == 7:
        return 2 if primary == 3 else 3
    return 4 if score == 8 else 5


def rating_dimensions(labels: pd.DataFrame, min_pattern_frac: float = 0.01) -> pd.DataFrame:
    """Long table (core, slide, rater, dimension, value) for every rated map. Uses the
    primary/secondary labels and the pattern fractions written by rater_labels()."""
    lab = labels.dropna(subset=["primary"])
    rows = []
    for r in lab.itertuples():
        p, s = int(r.primary), int(r.secondary)
        present = [q for q in PATTERNS if getattr(r, f"frac_{q}") >= min_pattern_frac]
        vals = {"cancer": int(p > 0), "primary": p, "secondary": s,
                "highest": max(present) if p > 0 else 0, "grade_group": grade_group(p, s)}
        rows += [{"core": r.core, "slide": r.slide, "rater": r.rater, "dimension": d, "value": v}
                 for d, v in vals.items()]
    return pd.DataFrame(rows)


def dimension_distributions(long: pd.DataFrame) -> pd.DataFrame:
    """Per (core, dimension): n_<code> rater counts for codes 0..5 (unused codes stay 0)
    and n_raters."""
    counts = (long.groupby(["core", "slide", "dimension"]).value.value_counts()
              .unstack(fill_value=0).reindex(columns=range(6), fill_value=0))
    counts.columns = [f"n_{c}" for c in counts.columns]
    counts["n_raters"] = counts.sum(axis=1)
    return counts.reset_index()


def slide_folds(cores: pd.Series) -> pd.DataFrame:
    """Leave-one-slide-out folds: fold index = position of the slide in sorted order."""
    slides = sorted({slide_of(c) for c in cores})
    return pd.DataFrame({"core": sorted(set(cores))}).assign(
        slide=lambda d: d.core.map(slide_of),
        fold=lambda d: d.slide.map({s: i for i, s in enumerate(slides)}),
    )


VISION_PATCH = 28  # Qwen2.5-VL/Omni: 14 px patches merged 2x2 -> one token per 28x28 px


def tissue_bbox(rgb: np.ndarray, sat_thresh: float = 0.07, margin: float = 0.02,
                min_line_frac: float = 0.02):
    """Bounding box (x0, y0, x1, y1) of stained tissue. A pixel is tissue if its HSV
    saturation exceeds sat_thresh (glass is near-white); a row/column is kept only if at
    least min_line_frac of its pixels are tissue, so isolated noisy pixels (JPEG
    artefacts, dust) near the border do not stretch the box."""
    arr = rgb.astype(np.float32) / 255.0
    mx, mn = arr.max(2), arr.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    mask = sat > sat_thresh
    xs = np.nonzero(mask.mean(0) >= min_line_frac)[0]
    ys = np.nonzero(mask.mean(1) >= min_line_frac)[0]
    h, w = sat.shape
    if xs.size == 0 or ys.size == 0:
        return 0, 0, w, h
    mx_, my_ = int(margin * w), int(margin * h)
    return (max(xs.min() - mx_, 0), max(ys.min() - my_, 0),
            min(xs.max() + mx_ + 1, w), min(ys.max() + my_ + 1, h))


def fit_to_patches(w: int, h: int, long_side: int) -> tuple[int, int]:
    """Scale (w, h) so the long side is long_side, both sides multiples of VISION_PATCH."""
    s = long_side / max(w, h)
    return (max(VISION_PATCH, round(w * s / VISION_PATCH) * VISION_PATCH),
            max(VISION_PATCH, round(h * s / VISION_PATCH) * VISION_PATCH))


def downscale_core(job: tuple) -> list[dict]:
    """Decode one core once and write it at every requested size. job = (core, src, out_root,
    sizes, crop, quality). Returns one manifest row per size."""
    from PIL import Image

    core, src, out_root, sizes, crop, quality = job
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(src) as im:
        full_w, full_h = im.size
        im.draft("RGB", (2 * max(sizes), 2 * max(sizes)))  # fast DCT-domain downscale
        im = im.convert("RGB")
        scale = full_w / im.size[0]
        box = tissue_bbox(np.asarray(im)) if crop else (0, 0, *im.size)
        region = im.crop(box)
    rows = []
    for size in sizes:
        w, h = fit_to_patches(*region.size, size)
        out = Path(out_root) / f"images_{size}" / f"{core}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        region.resize((w, h), Image.LANCZOS).save(out, quality=quality)
        rows.append({
            "core": core, "size": size, "path": str(out), "width": w, "height": h,
            "vision_tokens": (w // VISION_PATCH) * (h // VISION_PATCH),
            "orig_w": full_w, "orig_h": full_h,
            "crop_x0": round(box[0] * scale), "crop_y0": round(box[1] * scale),
            "crop_x1": round(box[2] * scale), "crop_y1": round(box[3] * scale),
            "um_per_px_rel": round(((box[2] - box[0]) * scale) / w, 3),  # original px per output px
        })
    return rows


def default_root() -> Path:
    return Path(os.environ.get("GLEASON2019_ROOT") or
                Path(__file__).resolve().parents[2] / "data" / "raw" / "gleason2019")
