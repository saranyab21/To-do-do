"""Does 4-bit quantisation change the zero-shot answers? (CPU, seconds)

    python scripts/compare_quantisation.py --a zeroshot_v1 --b zeroshot_8bit_check

Compares two zero-shot runs (same prompts, different quantisation) on the cores and sizes
both scored. Per (size, dimension): how often the top answer agrees, how far apart the
two answer distributions are (Jensen-Shannon, bits), and the soft NLL of each against the
pathologists. For scale, the pathologists' answers for two different cores differ by
0.37-0.43 bits on these questions (gate_v1), and an unchanged model gives 0 bits.
Writes results/metrics/gleason2019_zeroshot/quantisation_<a>_vs_<b>.csv.

Note: runs made before 2026-11-26 (e.g. zeroshot_v1) had the vision tower quantised as
well (see FP16_MODULES in src/models/zeroshot.py); later runs keep it in fp16.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.bootstrap import ci, stratified_indices
from src.evaluation.gate import js_divergence
from src.evaluation.soft_label import soft_nll

PROC = ROOT / "data/processed/gleason2019"
OUT_DIR = ROOT / "results/metrics/gleason2019_zeroshot"


def load(run: str) -> tuple[pd.DataFrame, dict]:
    d = PROC / "zeroshot" / run
    rows = [json.loads(x) for x in (d / "scores.jsonl").read_text().splitlines() if x]
    return pd.DataFrame(rows), json.loads((d / "run_info.json").read_text())["config"]


def compare(a: pd.DataFrame, b: pd.DataFrame, codes, dist: pd.DataFrame, slides: pd.Series,
            n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    lp = [f"lp_{c}" for c in codes]
    rows = []
    m = a.merge(b, on=["core", "size", "dimension"], suffixes=("_a", "_b"))
    for (size, dim), g in m.groupby(["size", "dimension"]):
        g = g.merge(dist, on=["core", "dimension"]).sort_values("core")
        qa = log_softmax(g[[f"{c}_a" for c in lp]].to_numpy(float), axis=1)
        qb = log_softmax(g[[f"{c}_b" for c in lp]].to_numpy(float), axis=1)
        p = g[[f"n_{c}" for c in codes]].to_numpy(float) / g.n_raters.to_numpy(float)[:, None]
        idx = stratified_indices(slides.loc[g.core].to_numpy(), n_boot,
                                 np.random.default_rng(seed))
        js = ci(js_divergence(np.exp(qa), np.exp(qb)), idx)
        dn = ci(soft_nll(qb, p) - soft_nll(qa, p), idx)
        rows.append({"size": size, "dimension": dim, "n_cores": len(g),
                     "top_answer_agreement": float((qa.argmax(1) == qb.argmax(1)).mean()),
                     "top_answer_a": str(codes[np.bincount(qa.argmax(1)).argmax()]),
                     "top_answer_b": str(codes[np.bincount(qb.argmax(1)).argmax()]),
                     "js_bits": js[0], "js_lo": js[1], "js_hi": js[2],
                     "nll_a": float(soft_nll(qa, p).mean()), "nll_b": float(soft_nll(qb, p).mean()),
                     "nll_b_minus_a": dn[0], "nll_b_minus_a_lo": dn[1], "nll_b_minus_a_hi": dn[2]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="zeroshot_v1", help="reference run (4-bit)")
    ap.add_argument("--b", default="zeroshot_8bit_check", help="comparison run")
    args = ap.parse_args()
    a, cfg_a = load(args.a)
    b, cfg_b = load(args.b)
    for k in ("model_id", "codes", "prompts"):
        if cfg_a[k] != cfg_b[k]:
            sys.exit(f"runs differ in '{k}', so the comparison would not isolate quantisation")
    dist = pd.read_csv(PROC / "rater_distributions.csv")
    slides = pd.read_csv(PROC / "folds.csv").set_index("core").slide
    res = compare(a, b, cfg_a["codes"], dist, slides)
    res.insert(0, "b", f"{args.b} ({cfg_b['quant']})")
    res.insert(0, "a", f"{args.a} ({cfg_a['quant']})")
    out = OUT_DIR / f"quantisation_{args.a}_vs_{args.b}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.drop(columns=["a", "b"]).round(3).to_string(index=False))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
