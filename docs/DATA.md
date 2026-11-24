# Data access and handling

This file says where each dataset lives, who may access it, and what may enter
this repository. Read it before adding any loader, notebook or result file.

## Rules

1. **No raw or derived PC-GITA data is ever committed, copied off the lab
   cluster, or stored on a personal machine.** This includes audio, rater
   sheets, transcripts, features, embeddings, per-speaker predictions, and
   notebook outputs that display any of these.
2. Code reads dataset locations from environment variables (see
   `.env.example`), never from hard-coded paths. The same code then runs on the
   cluster without moving data.
3. Only aggregate results (e.g. κ with confidence intervals, calibration curves
   pooled over speakers) leave the cluster, and only into `results/`.
4. Notebook outputs are stripped on commit (`nbstripout` via pre-commit).
   Still, do not display per-speaker PC-GITA data in notebooks.
5. `.gitignore` blocks common data file types (`*.wav`, `*.xlsx`, `*.csv`,
   `*.npy`, ...) everywhere. Whitelist a specific non-sensitive file
   explicitly (`!path/to/file.csv`) rather than weakening the rule. The one
   standing exception is `results/metrics/*.csv` and `results/tables/*.csv`,
   which may hold **aggregate** results only, never per-speaker or
   per-patient rows.

## Datasets

| Dataset | Status | Location | Access / licence | May leave the cluster? |
|---|---|---|---|---|
| **Gleason2019** (Nir et al., 2018): 244 TMA cores, up to 6 pathologists | Public | `$GLEASON2019_ROOT` (local `data/raw/gleason2019/` or CIP pool) | Public challenge release, check licence terms before redistribution | Yes (do not redistribute) |
| **PC-GITA** (Orozco-Arroyave et al., 2014): 50 PD + 50 HC speakers, per-rater mFDA sheets | **Sensitive, NDA** | `$PCGITA_ROOT` on the LME internal cluster only | Via supervisor (Dr.-Ing. T. Arias-Vergara); NDA signed: _date_ | **No** |

## Open items

- [ ] NDA signed and returned
- [ ] Cluster account granted; record `PCGITA_ROOT` path (not here, in `.env` on the cluster)
- [ ] Gleason2019 download, patient-ID map, licence terms noted
- [ ] mFDA task-to-category table (Vásquez-Correa et al., 2018) transcribed into `src/data/`
- [ ] Score-binning rule for summed mFDA categories confirmed with supervisor
