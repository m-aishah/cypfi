# PanelPath — TEKNOFEST 2026 AI in Healthcare

## What this is

We are competing in **TEKNOFEST 2026 AI in Healthcare, University Level**. The task is binary classification of genetic missense variants as **Pathogenic (1)** or **Benign (0)** across four disease-specific panels. The primary evaluation metric is **macro F1 score**.

We passed Stage 1 (project proposal) with a score of **94.5/100**. The Project Details Report is due **June 29, 2026**. The final competition (live code re-run in front of jury) is **August–September 2026**.

## Files in this repo

- `specs/` — competition specification PDF (read this carefully)
- `report/` — our Stage 1 proposal report (read this to understand what we committed to)
- `data/` — four training CSVs:
  - `YARISMA_TRAIN_MASTER.csv` — General dataset
  - `YARISMA_TRAIN_KANSER.csv` — Hereditary Cancer panel
  - `YARISMA_TRAIN_PAH.csv` — PAH (Phenylketonuria) panel
  - `YARISMA_TRAIN_CFTR.csv` — CFTR (Cystic Fibrosis) panel

## Stage 1 proposal score breakdown

| Section | Score |
|---|---|
| Literature review | 9.00/10 |
| Dataset & labels | 5.00/5 |
| Data restrictions | 4.50/5 |
| Preprocessing strategy | 5.00/5 |
| Label reliability | 5.00/5 |
| Class balance | 5.00/5 |
| Algorithm justification | 4.50/5 |
| Experimental protocol | 4.50/5 |
| Performance metrics | 4.50/5 |
| Error analysis | 4.50/5 |
| Explainability | 5.00/5 |
| Learning process | 5.00/5 |
| Why this algorithm | 5.00/5 |
| Why alternatives rejected | 4.50/5 |
| Parameter settings | 5.00/5 |
| Computational resources | 4.50/5 |
| Originality | 4.50/5 |
| References & layout | 9.50/10 |
| **Total** | **94.5/100** |

## Hard constraints (non-negotiable)

- **No genomic coordinates** — column names are anonymised (AL_1..AL_334, CAT_1..6, EK_1..9, AA_1, AA_2). Never attempt to recover position info or look anything up externally.
- **No external data** — only the provided CSVs.
- **Fully reproducible** — must run deterministically, seed=42 everywhere.
- **CPU only** — must run on Google Colab free tier (~12GB RAM). No GPU dependency.
- **No label leakage** — hyperparameter tuning must never see the fold it is evaluated on.

## What I want from you

1. **Read the competition spec and Stage 1 report thoroughly** before doing anything else. Understand what we proposed and what we're being held to.

2. **Do your own rigorous analysis of all four datasets** — don't assume anything from the spec or the proposal is accurate. Verify everything empirically. Look for surprises.

3. **Build the best possible model** — use the proposal as a reference point, not a constraint. If the data analysis reveals a smarter approach, take it. The goal is the highest macro F1 on unseen test data, a report worth reading, and code that can be re-run live in front of a jury without breaking.

4. **Produce clean, well-commented, reproducible code** structured so it can be explained and re-run under pressure at the finals.

The jury can ask us to re-run our code live. Build accordingly.
