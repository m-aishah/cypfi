# PanelPath — TEKNOFEST 2026 AI in Healthcare
## Claude Code Project Brief

This is the implementation project for **PanelPath**, a competition entry for TEKNOFEST 2026 University-level AI in Healthcare. The model is due for final submission as part of the **Project Details Report on June 29, 2026**.

Read all docs in this folder before writing any code. Start with this file, then read `DATA_ANALYSIS.md`, then `PIPELINE_SPEC.md`.

---

## Competition in One Sentence

Binary classification of genetic missense variants as **Pathogenic (1)** or **Benign (0)** across four disease panels, evaluated by **macro F1 score**.

## What We Proposed (Approved, Scored 94.5/100)

A 4-stage panel-aware ensemble pipeline:
- **Stage 0:** Preprocessing — feature group identification, MICE imputation, Atchley encoding, log-transform of MAF features, Winsorisation, standardisation
- **Stage 1:** XGBoost trained on the General (MASTER) dataset → outputs calibrated probability P₁ per variant
- **Stage 2:** Panel-specialist models (Extra Trees for KANSER, TabPFN v2 for PAH and CFTR) → outputs P₂ per variant
- **Stage 3:** Logistic Regression stacking on [P₁, P₂] → final probability, with per-panel threshold τ tuned to maximise macro F1

Full pipeline spec is in `PIPELINE_SPEC.md`.

## Critical: What the Real Data Revealed

The actual training data **differs significantly from the competition spec**. See `DATA_ANALYSIS.md` for full details. The most important changes needed vs. our proposal:

1. **Datasets are NOT balanced** — spec said 50/50, actual data is heavily pathogenic-skewed. Must use `scale_pos_weight` in XGBoost and `class_weight='balanced'` in Extra Trees.
2. **MASTER is a superset** — it contains 578 variant IDs that also appear in the panel files. The panel-unique portion is 2,353 rows. Handle overlap carefully to prevent leakage.
3. **CAT_6 is inconsistent** — String type in MASTER/KANSER/CFTR, Float in PAH, all-null in PAH. Either drop it per panel or force to string before encoding.
4. **PAH is now the hardest panel** (83% pathogenic, not CFTR) — revise difficulty expectations accordingly.

## Key Constraints (Never Violate These)

- **No genomic coordinates** — column names are anonymous (AL_1..AL_334, CAT_1..CAT_6, EK_1..EK_9, AA_1, AA_2). Never try to recover position info.
- **No external data** — only the provided CSVs. No ClinVar lookups, no gnomAD queries.
- **Fully reproducible** — seed=42 everywhere. Must run on CPU only (Google Colab free tier, ~12GB RAM).
- **No label leakage** — hyperparameter tuning only sees training folds, never validation or test folds.

## Files in This Repo

```
data/
  YARISMA_TRAIN_MASTER.csv     # General dataset (2931 rows, superset)
  YARISMA_TRAIN_KANSER.csv     # Hereditary Cancer panel (388 rows)
  YARISMA_TRAIN_PAH.csv        # PAH panel (372 rows)
  YARISMA_TRAIN_CFTR.csv       # CFTR panel (111 rows)

docs/
  CLAUDE.md                    # This file
  DATA_ANALYSIS.md             # Actual data findings vs spec
  PIPELINE_SPEC.md             # Full pipeline architecture and implementation notes
  REPORT_CONTEXT.md            # Context for writing the Project Details Report

src/
  (to be built)
```

## Python Environment

```
Python 3.10
xgboost==2.0.3
scikit-learn==1.4.0
optuna==3.5.0
tabpfn  # latest from PyPI (2025)
shap==0.44.0
pandas==2.1.0
numpy==1.26.0
scipy
```

## Global Seed

`SEED = 42` — pass to every model, every CV splitter, every random operation.

## Primary Metric

**Macro F1** — this is the competition ranking metric. Optimise everything around this. Secondary metrics to also track and report: ROC-AUC, PR-AUC, Accuracy, Brier score.
