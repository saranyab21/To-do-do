# Master's Thesis

**Query-Conditioned, Uncertainty-Aware Multimodal Clinical Assessment: Modeling
Expert Disagreement and Calibrated Deference Under Human Influence**

M.Sc. Artificial Intelligence, FAU Erlangen-Nürnberg. Supervisor: Dr.-Ing. Tomás
Arias-Vergara (Pattern Recognition Lab, LME).

## Project structure

- `docs/`: papers, proposal, meeting notes, talks, preregistration, [data policy](docs/DATA.md)
- `data/`: datasets (contents never committed; see `docs/DATA.md`)
- `src/`: research code (`src.data`, `src.models`, `src.evaluation`, `src.utils`)
- `experiments/`: run configs (tracked) and checkpoints/logs (ignored)
- `notebooks/`: exploratory analysis (outputs stripped on commit)
- `results/`: aggregate figures, metrics and tables
- `tests/`: unit tests
- `thesis/`: LaTeX source (`main.tex`, chapters, bibliography, figures)
- `scripts/`: entry points and cluster job scripts

## Data

**PC-GITA is under NDA and lives only on the LME cluster.** Read
[docs/DATA.md](docs/DATA.md) before writing any loader or notebook.
Dataset paths come from environment variables; copy `.env.example` to `.env`.

## Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   Linux: source .venv/bin/activate
pip install -e ".[dev]"            # core: simulation, calibration, statistics
pip install -e ".[model,data]"     # model + data stack (GPU machine / cluster)
pre-commit install
```

Use the Python version the cluster provides (3.11/3.12 recommended); pin exact
versions with `pip freeze > requirements-lock.txt` once the cluster environment
is fixed.

## Reproducibility

Every experiment runs from a config in `experiments/configs/` with a fixed seed,
and records the git commit it ran on. The analysis plan is preregistered
(OSF) before real-data evaluation; see `docs/preregistration/`.
