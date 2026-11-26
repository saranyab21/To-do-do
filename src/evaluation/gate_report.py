"""Tables and figures for the Pillar-1 gate (CPU). Called by scripts/run_gleason_gate.py."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.evaluation import gate as G
from src.evaluation.bootstrap import ci, stratified_indices
from src.evaluation.soft_label import soft_nll

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed/gleason2019"

# reference palette (light), validated with the dataviz validator; slots 1-3 all-pairs safe
SURFACE, INK, INK_2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
BLUES = LinearSegmentedColormap.from_list("blue_ramp", ["#f1f6fd", "#9cc3f0", "#2a78d6",
                                                        "#0e3f80"])
DIM_LABEL = {"cancer": "Cancer present", "primary": "Primary pattern",
             "secondary": "Secondary pattern", "highest": "Highest pattern",
             "grade_group": "ISUP grade group"}


def _style():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": NEUTRAL, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "axes.spines.top": False,
                         "axes.spines.right": False})


def _load_scores(out: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(x) for x in (out / "scores.jsonl").read_text().splitlines()
                         if x])


def analyse(cfg: dict, dist: pd.DataFrame, out: Path) -> dict:
    scores = _load_scores(out)
    folds = pd.read_csv(PROC / "folds.csv").set_index("core")
    partners = pd.read_csv(out / "partners.csv").set_index("core").partner
    dims = list(cfg["dimensions"])
    real = scores[scores.condition == "real"]
    cores = sorted(set.intersection(*(set(real[real.dimension == d].core) for d in dims)))
    if len(cores) < 20:
        print(f"only {len(cores)} fully scored cores; skipping analysis")
        return {}
    slides = folds.loc[cores, "slide"].to_numpy()
    fo = folds.loc[cores, "fold"].to_numpy()
    rng = np.random.default_rng(cfg["seed"])
    idx = stratified_indices(slides, cfg["n_boot"], rng)
    one_slide = len(np.unique(fo)) < 2
    eff = cfg.get("min_effect_nats", 0.0)

    def frame(d, cond, core_list=cores):
        s = scores[(scores.dimension == d) & (scores.condition == cond)].set_index("core")
        return s.reindex(core_list)

    dd = {d: dist[dist.dimension == d].set_index("core").loc[cores] for d in dims}
    gate_rows, rel_rows, ent_pts, ground_rows = [], [], {}, []
    for d in dims:
        codes = cfg["dimensions"][d]["codes"]
        r = frame(d, "real").join(dd[d][[f"n_{c}" for c in codes] + ["n_raters"]],
                                  lsuffix="_score")
        logq, p = G.logq_and_p(r, codes)
        logq_T = G.loso_temperature(logq, p, fo)
        prior = G.loso_prior_logq(p, fo)
        nll, nll_T, nll_prior = soft_nll(logq, p), soft_nll(logq_T, p), soft_nll(prior, p)
        floor = -(np.where(p > 0, p * np.log(np.clip(p, 1e-12, None)), 0)).sum(1)
        g1 = ci(nll_T - nll_prior, idx)
        rel_raw, ece_raw = G.soft_reliability(np.exp(logq), p)
        rel_T, ece_T = G.soft_reliability(np.exp(logq_T), p)
        rel_rows += [{"dimension": d, "model": m, "bin": b, "pred": pr, "obs": ob, "n": n}
                     for m, rows in (("raw", rel_raw), ("loso_T", rel_T))
                     for b, pr, ob, n in rows]
        h_q, h_qT, h_p = (G.norm_entropy(np.exp(logq)), G.norm_entropy(np.exp(logq_T)),
                          G.norm_entropy(p))
        g2 = G.spearman_boot(h_qT, h_p, idx)
        g2_raw = G.spearman_boot(h_q, h_p, idx)
        ent_pts[d] = (h_p, h_qT, g2)
        row = {"dimension": d, "n": len(cores), "mass_mean": float(r["mass"].mean()),
               "top_answer_share": float(r["top_token"].value_counts(normalize=True).iloc[0]),
               "top_answer": str(r["top_token"].value_counts().index[0]),
               "nll_raw": nll.mean(), "nll_loso_T": nll_T.mean(), "nll_loso_prior": nll_prior.mean(),
               "nll_floor": floor.mean(), "ece_raw": ece_raw, "ece_loso_T": ece_T,
               "G1_diff": g1[0], "G1_lo": g1[1], "G1_hi": g1[2],
               "G1_pass": (not one_slide) and g1[2] < -eff,
               "G2_rho": g2[0], "G2_lo": g2[1], "G2_hi": g2[2], "G2_pass": g2[1] > 0,
               "G2_rho_raw": g2_raw[0]}
        conds = set(scores[scores.dimension == d].condition)
        if "shuffled" in conds:
            sh = frame(d, "shuffled")
            if sh[f"lp_{codes[0]}"].notna().all():
                lq_sh, _ = G.logq_and_p(sh.join(dd[d][[f"n_{c}" for c in codes] + ["n_raters"]],
                                                lsuffix="_score"), codes)
                g4 = ci(soft_nll(lq_sh, p) - nll, idx)
                p_partner = (dist[dist.dimension == d].set_index("core")
                             .reindex(partners.loc[cores]))
                pp = p_partner[[f"n_{c}" for c in codes]].to_numpy(float) / \
                    p_partner.n_raters.to_numpy(float)[:, None]
                row.update(G4_diff=g4[0], G4_lo=g4[1], G4_hi=g4[2], G4_pass=g4[1] > eff)
                ground_rows.append({"dimension": d, "what": "model: real vs shuffled image",
                                    **dict(zip(("mean", "lo", "hi"),
                                               ci(G.js_divergence(np.exp(logq), np.exp(lq_sh)), idx)))})
                ground_rows.append({"dimension": d, "what": "pathologists: core vs shuffle partner",
                                    **dict(zip(("mean", "lo", "hi"), ci(G.js_divergence(p, pp), idx)))})
        if "blank" in conds:
            bl = frame(d, "blank")
            if bl[f"lp_{codes[0]}"].notna().all():
                lq_bl, _ = G.logq_and_p(
                    bl.join(dd[d][[f"n_{c}" for c in codes] + ["n_raters"]], lsuffix="_score"),
                    codes)
                ground_rows.append({"dimension": d, "what": "model: real vs blank image",
                                    **dict(zip(("mean", "lo", "hi"),
                                               ci(G.js_divergence(np.exp(logq), np.exp(lq_bl)), idx)))})
        gate_rows.append(row)

    # G3: query swap on dimensions sharing one code set
    swap = cfg.get("query_swap", [])
    swap_rows = []
    for t in swap:
        codes = cfg["dimensions"][t]["codes"]
        p_t = dd[t][[f"n_{c}" for c in codes]].to_numpy(float) / dd[t].n_raters.to_numpy(float)[:, None]
        nll_by_ask = {}
        for a in swap:
            lq_a, _ = G.logq_and_p(frame(a, "real").join(dd[t][[f"n_{c}" for c in codes] + ["n_raters"]],
                                                         lsuffix="_score"), codes)
            nll_by_ask[a] = soft_nll(lq_a, p_t)
            oracle = soft_nll(G.smoothed_log_shares(dd[a], codes), p_t)
            m, lo, hi = ci(nll_by_ask[a], idx)
            swap_rows.append({"truth": t, "asked": a, "nll": m, "lo": lo, "hi": hi,
                              "pathologists_as_predictor": float(oracle.mean())})
        diffs = [ci(nll_by_ask[t] - nll_by_ask[a], idx) for a in swap if a != t]
        g = next(r for r in gate_rows if r["dimension"] == t)
        g["G3_worst_hi"] = max(x[2] for x in diffs)
        g["G3_pass"] = all(x[2] < -eff for x in diffs)

    met = ROOT / "results/metrics/gleason2019_gate" / cfg["run_name"]
    fig = ROOT / "results/figures/gleason2019_gate" / cfg["run_name"]
    met.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    gate = pd.DataFrame(gate_rows)
    gate.round(4).to_csv(met / "gate.csv", index=False)
    pd.DataFrame(rel_rows).round(4).to_csv(met / "reliability.csv", index=False)
    swap_df = pd.DataFrame(swap_rows)
    swap_df.round(4).to_csv(met / "query_swap.csv", index=False)
    ground = pd.DataFrame(ground_rows)
    ground.round(4).to_csv(met / "grounding.csv", index=False)
    summary = {"run_name": cfg["run_name"], "n_cores": len(cores), "size": cfg["size"],
               "n_scores": len(scores), "n_nonfinite": int((~scores.finite.astype(bool)).sum()),
               "pass": {r["dimension"]: {k: bool(r[k]) for k in ("G1_pass", "G2_pass", "G3_pass",
                                                                 "G4_pass") if k in r}
                        for r in gate_rows}}
    (met / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    _style()
    plot_reliability(pd.DataFrame(rel_rows), gate, dims, fig / "reliability.png")
    plot_entropy(ent_pts, dims, fig / "entropy_alignment.png")
    if len(swap_df):
        plot_query_swap(swap_df, swap, fig / "query_swap.png")
    if len(ground):
        plot_grounding(ground, dims, fig / "grounding.png")

    cols = ["dimension", "mass_mean", "top_answer", "top_answer_share", "nll_raw", "nll_loso_T",
            "nll_loso_prior", "nll_floor", "ece_raw", "ece_loso_T", "G1_diff", "G1_pass",
            "G2_rho", "G2_pass", "G3_pass", "G4_diff", "G4_pass"]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(gate[[c for c in cols if c in gate]].round(3).to_string(index=False))
    return summary


def plot_reliability(rel, gate, dims, path):
    fig, axes = plt.subplots(1, len(dims), figsize=(3.1 * len(dims), 3.4), sharey=True)
    for ax, d in zip(np.atleast_1d(axes), dims):
        ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls="--")
        for m, c, lab in (("raw", S1, "zero-shot"), ("loso_T", S2, "zero-shot + LOSO temperature")):
            r = rel[(rel.dimension == d) & (rel.model == m)]
            ax.plot(r.pred, r.obs, "-o", color=c, lw=2, ms=6, mec=SURFACE, mew=1.5, label=lab)
        gr = gate.set_index("dimension").loc[d]
        ax.text(0.03, 0.97, f"ECE {gr.ece_raw:.2f} → {gr.ece_loso_T:.2f}", transform=ax.transAxes,
                va="top", fontsize=9, color=INK_2)
        ax.set_title(DIM_LABEL[d], loc="left")
        ax.set_xlabel("predicted probability")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.grid(color=GRID, lw=0.7)
    np.atleast_1d(axes)[0].set_ylabel("share of pathologists giving that answer")
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, frameon=False)
    fig.suptitle("G1 · Calibration to pathologist disagreement", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_entropy(pts, dims, path):
    fig, axes = plt.subplots(1, len(dims), figsize=(3.1 * len(dims), 3.4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, d in zip(np.atleast_1d(axes), dims):
        hp, hq, (rho, lo, hi) = pts[d]
        ax.scatter(hp + rng.uniform(-0.015, 0.015, len(hp)), hq, s=10, color=S1, alpha=0.45, lw=0)
        bins = np.unique(np.round(hp, 3))
        means = [hq[np.isclose(np.round(hp, 3), b)].mean() for b in bins]
        ax.plot(bins, means, "-o", color=INK, lw=1.5, ms=4)
        ax.set_title(DIM_LABEL[d], loc="left")
        ax.text(0.03, 0.97, f"Spearman ρ = {rho:.2f} [{lo:.2f}, {hi:.2f}]", transform=ax.transAxes,
                va="top", fontsize=9, color=INK_2)
        ax.set_xlabel("pathologists' disagreement (entropy)")
        ax.set_xlim(-0.05, 1.05), ax.set_ylim(-0.05, 1.05)
        ax.grid(color=GRID, lw=0.7)
    np.atleast_1d(axes)[0].set_ylabel("model uncertainty (entropy, after LOSO T)")
    fig.suptitle("G2 · Does the model's uncertainty follow pathologist disagreement? "
                 "(dots: cores; line: mean per disagreement level)", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_query_swap(sw, dims, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    vmin = min(sw.nll.min(), sw.pathologists_as_predictor.min())
    vmax = max(sw.nll.max(), sw.pathologists_as_predictor.max())
    for ax, col, title in ((axes[0], "nll", "Model: answer to the asked question"),
                           (axes[1], "pathologists_as_predictor",
                            "Reference: pathologists' answers to the asked question")):
        m = sw.pivot(index="asked", columns="truth", values=col).loc[dims, dims]
        im = ax.imshow(m.to_numpy(), cmap=BLUES, vmin=vmin, vmax=vmax)
        for i in range(len(dims)):
            for j in range(len(dims)):
                v = m.iat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if (v - vmin) / (vmax - vmin + 1e-9) > 0.55 else INK)
        ax.set_xticks(range(len(dims)), [DIM_LABEL[d] for d in dims], rotation=20, ha="right")
        ax.set_yticks(range(len(dims)), [DIM_LABEL[d] for d in dims])
        ax.set_xlabel("scored against pathologists' answers for")
        ax.set_title(title, loc="left", fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("question asked")
    axes[1].set_yticks(range(len(dims)), [""] * len(dims))  # rows as in the left panel
    fig.colorbar(im, ax=axes, shrink=0.8, label="soft cross-entropy (lower is better)")
    fig.suptitle("G3 · Query specificity: the matching question should give the lowest loss in "
                 "each column", x=0.01, ha="left", color=INK)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_grounding(gr, dims, path):
    whats = ["model: real vs shuffled image", "model: real vs blank image",
             "pathologists: core vs shuffle partner"]
    colours = dict(zip(whats, (S1, S2, S3)))
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    w = 0.26
    for k, wh in enumerate(whats):
        r = gr[gr.what == wh].set_index("dimension").reindex(dims)
        xs = np.arange(len(dims)) + (k - 1) * w
        ax.bar(xs, r["mean"], width=w - 0.03, color=colours[wh], label=wh)
        ax.errorbar(xs, r["mean"], yerr=[r["mean"] - r["lo"], r["hi"] - r["mean"]], fmt="none",
                    ecolor=INK_2, lw=1.2, capsize=0)
    ax.set_xticks(range(len(dims)), [DIM_LABEL[d] for d in dims])
    ax.set_ylabel("change in answer distribution\n(Jensen-Shannon, bits)")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.16))
    fig.suptitle("G4 · Grounding: how much does the answer change when the image changes?",
                 x=0.01, ha="left", color=INK, y=1.04)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
