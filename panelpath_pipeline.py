"""
PanelPath — TEKNOFEST 2026 AI in Healthcare
Missense variant pathogenicity classification (binary: Pathogenic=1, Benign=0)
Primary metric: macro F1

Architecture (data-driven):
  MASTER  → LightGBM, panel-only CV
  KANSER  → LightGBM, panel-only CV  (MASTER augmentation hurts: −0.016)
  PAH     → LightGBM, MASTER-augmented CV  (+0.070)
  CFTR    → LightGBM, MASTER-augmented CV  (+0.143, variance halved)

All experiments seed=42, CPU-only, no external data.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS & SEED
# ─────────────────────────────────────────────────────────────────────────────
import os, warnings, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import (f1_score, matthews_corrcoef, roc_auc_score,
                              average_precision_score, confusion_matrix,
                              classification_report, precision_recall_curve,
                              roc_curve, balanced_accuracy_score)
from sklearn.preprocessing import label_binarize
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import shap

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

# Output directories
OUT_PLOTS   = 'outputs/plots'
OUT_MODELS  = 'outputs/models'
OUT_RESULTS = 'outputs/results'
for d in [OUT_PLOTS, OUT_MODELS, OUT_RESULTS]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = 'data'
raw = {
    'MASTER': pd.read_csv(f'{DATA_DIR}/YARISMA_TRAIN_MASTER.csv'),
    'KANSER': pd.read_csv(f'{DATA_DIR}/YARISMA_TRAIN_KANSER.csv'),
    'PAH':    pd.read_csv(f'{DATA_DIR}/YARISMA_TRAIN_PAH.csv'),
    'CFTR':   pd.read_csv(f'{DATA_DIR}/YARISMA_TRAIN_CFTR.csv'),
}

# Column groups (same schema across all datasets)
_sample = raw['MASTER']
AL_COLS  = [c for c in _sample.columns if c.startswith('AL_')]    # 334 continuous
EK_COLS  = [c for c in _sample.columns if c.startswith('EK_')]    # 9  continuous
CAT_COLS = [c for c in _sample.columns if c.startswith('CAT_')]   # 6  categorical
AA_COLS  = [c for c in _sample.columns if c.startswith('AA_')]    # 2  amino acid
FEAT_COLS = AL_COLS + EK_COLS + CAT_COLS + AA_COLS

print("=" * 65)
print("DATASET SUMMARY")
print("=" * 65)
for name, df in raw.items():
    n0 = (df['Label'] == 0).sum()
    n1 = (df['Label'] == 1).sum()
    miss = df[AL_COLS].isnull().mean().mean()
    print(f"  {name:6s}  n={len(df):4d}  B={n0}({n0/len(df)*100:.0f}%)  "
          f"P={n1}({n1/len(df)*100:.0f}%)  AL_miss={miss*100:.0f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
# Atchley factors for standard 20 amino acids (Atchley et al. 2005 PNAS)
# Dimensions: polarity, secondary-structure, molecular volume,
#             codon diversity, electrostatic charge
ATCHLEY = {
    'A': [-0.591, -1.302, -0.733,  1.570, -0.146],
    'C': [-1.343,  0.465, -0.862, -1.020, -0.255],
    'D': [ 1.050,  0.302, -3.656, -0.259, -3.242],
    'E': [ 1.357, -1.453,  1.477,  0.113, -0.837],
    'F': [-1.006, -0.590,  1.891, -0.397,  0.412],
    'G': [-0.384,  1.652,  1.330,  1.045,  2.064],
    'H': [ 0.336, -0.417, -1.673, -1.474, -0.078],
    'I': [-1.239, -0.547,  2.131,  0.393,  0.816],
    'K': [ 1.831, -0.561,  0.533, -0.277,  1.648],
    'L': [-1.019, -0.987, -1.505,  1.266, -0.912],
    'M': [-0.663, -1.524,  2.219, -1.005,  1.212],
    'N': [ 0.945,  0.828,  1.299, -0.169,  0.933],
    'P': [ 0.189,  2.081, -1.628,  0.421, -1.392],
    'Q': [ 0.931, -0.179, -3.005, -0.503, -1.853],
    'R': [ 1.538, -0.055,  1.502,  0.440,  2.897],
    'S': [-0.228,  1.399, -4.760,  0.670, -2.647],
    'T': [-0.032,  0.326,  2.213,  0.908,  1.313],
    'V': [-1.337, -0.279, -0.544,  1.242, -1.262],
    'W': [-0.595,  0.009,  0.672, -2.128, -0.184],
    'Y': [ 0.260,  0.830,  3.097, -0.838,  1.512],
}
ATCHLEY_ZERO = [0.0] * 5  # fallback for non-standard / missing


def encode_features(train_df, val_df=None):
    """
    Build feature matrices for train (and optionally val) dataframes.
    - AL + EK: kept as-is (LightGBM handles NaN natively)
    - miss_count: total number of missing AL values per sample
    - CAT cols: string → integer code (unseen in val → NaN → LightGBM handles)
    - AA cols: Atchley 5-factor encoding (10 new cols) + original label-encoded col
    Returns (X_train, X_val) or (X_train,) if val_df is None.
    """
    dfs = [train_df] + ([val_df] if val_df is not None else [])
    results = []

    # Build category maps from train only
    cat_maps = {}
    for c in CAT_COLS:
        tr_vals = train_df[c].fillna('__NA__').astype(str)
        cats = sorted(tr_vals.unique())
        cat_maps[c] = {v: i for i, v in enumerate(cats)}

    for df in dfs:
        X = df[AL_COLS + EK_COLS].copy()

        # Missingness summary feature
        X['miss_count'] = df[AL_COLS].isnull().sum(axis=1).astype(float)

        # CAT encoding (unseen label → NaN, fine for LightGBM)
        for c in CAT_COLS:
            if c == 'CAT_6':
                # Mostly null (97-100%) and single-valued in panels → binary flag
                X['CAT_6_present'] = (~df[c].isnull()).astype(float)
            else:
                vals = df[c].fillna('__NA__').astype(str)
                X[c] = vals.map(cat_maps[c]).astype(float)

        # AA encoding: Atchley factors (5 dims each) + raw label
        aa_cat_maps = {}
        for c in AA_COLS:
            tr_aa = train_df[c].fillna('__NA__').astype(str)
            aa_cats = sorted(tr_aa.unique())
            aa_cat_maps[c] = {v: i for i, v in enumerate(aa_cats)}

        for c in AA_COLS:
            vals = df[c].fillna('__NA__').astype(str)
            # Raw label code
            X[f'{c}_code'] = vals.map(aa_cat_maps[c]).astype(float)
            # Atchley factors: use first char if multi-char (e.g. 'AG' → 'A')
            for dim in range(5):
                X[f'{c}_atch{dim}'] = vals.apply(
                    lambda v: ATCHLEY.get(v[0] if v != '__NA__' else '', ATCHLEY_ZERO)[dim]
                    if v != '__NA__' else np.nan
                ).astype(float)

        results.append(X)

    return tuple(results) if val_df is not None else results[0]


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset: which training data to use
MASTER_DF = raw['MASTER']

def get_augmented_train(panel_name, panel_df):
    """Return MASTER rows not in this panel, for augmentation."""
    panel_ids = set(panel_df['Variant_ID'])
    return MASTER_DF[~MASTER_DF['Variant_ID'].isin(panel_ids)].reset_index(drop=True)

# Empirically determined augmentation strategy:
#   KANSER: panel-only (augmentation hurts: −0.016)
#   PAH:    MASTER-augmented (+0.070)
#   CFTR:   MASTER-augmented (+0.143, variance halved)
USE_AUGMENTATION = {'KANSER': False, 'PAH': True, 'CFTR': True}

# CV config
CV_FOLDS = 5
CV_REPEATS = 3  # repeated stratified CV for stable estimates


# ─────────────────────────────────────────────────────────────────────────────
# 4. OPTUNA HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
def tune_lgbm(X_train, y_train, n_trials=80, cv_folds=5, aug_df=None, aug_y=None):
    """
    Tune LightGBM via Optuna TPE. Returns best params dict.
    aug_df/aug_y: extra training rows added to every fold's train split
    (used for panel augmentation — aug data never appears in val).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int('n_estimators', 200, 2000),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int('num_leaves', 15, 255),
            min_child_samples=trial.suggest_int('min_child_samples', 5, 100),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.4, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            class_weight='balanced',
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
        fold_f1s = []
        for tr_idx, va_idx in cv.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]
            if aug_df is not None:
                X_tr = pd.concat([aug_df, X_tr], ignore_index=True)
                y_tr = pd.concat([aug_y.reset_index(drop=True),
                                   y_tr.reset_index(drop=True)], ignore_index=True)
            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(period=-1)])
            preds = m.predict(X_va)
            fold_f1s.append(f1_score(y_va, preds, average='macro'))
        return np.mean(fold_f1s)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# ─────────────────────────────────────────────────────────────────────────────
# 5. THRESHOLD OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────
def optimise_threshold(y_true, y_prob, grid=np.arange(0.05, 0.95, 0.01)):
    """Find threshold maximising macro F1."""
    best_tau, best_f1 = 0.5, 0.0
    for tau in grid:
        preds = (y_prob >= tau).astype(int)
        f = f1_score(y_true, preds, average='macro', zero_division=0)
        if f > best_f1:
            best_f1, best_tau = f, tau
    return best_tau, best_f1


# ─────────────────────────────────────────────────────────────────────────────
# 6. FULL EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    return {
        'macro_f1':   f1_score(y_true, y_pred, average='macro'),
        'f1_benign':  f1_score(y_true, y_pred, pos_label=0, average='binary'),
        'f1_pathog':  f1_score(y_true, y_pred, pos_label=1, average='binary'),
        'mcc':        matthews_corrcoef(y_true, y_pred),
        'roc_auc':    roc_auc_score(y_true, y_prob),
        'pr_auc':     average_precision_score(y_true, y_prob),
        'bal_acc':    balanced_accuracy_score(y_true, y_pred),
        'sensitivity': confusion_matrix(y_true, y_pred).ravel()[3] /
                       max(1, (confusion_matrix(y_true, y_pred).ravel()[2] +
                               confusion_matrix(y_true, y_pred).ravel()[3])),
        'specificity': confusion_matrix(y_true, y_pred).ravel()[0] /
                       max(1, (confusion_matrix(y_true, y_pred).ravel()[0] +
                               confusion_matrix(y_true, y_pred).ravel()[1])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
all_results   = {}  # dataset → metrics dict
all_models    = {}  # dataset → fitted LGBMClassifier
all_oof_probs = {}  # dataset → (y_true_oof, y_prob_oof)
all_thresholds = {} # dataset → tau

DATASETS_ORDER = ['MASTER', 'KANSER', 'PAH', 'CFTR']

for ds_name in DATASETS_ORDER:
    print(f"\n{'='*65}")
    print(f"  DATASET: {ds_name}")
    print(f"{'='*65}")

    panel_df = raw[ds_name]
    y_all    = panel_df['Label']

    # ── Build augmentation data (pre-encoded) ──────────────────────────────
    use_aug = USE_AUGMENTATION.get(ds_name, False)
    if use_aug:
        master_aug = get_augmented_train(ds_name, panel_df)
        print(f"  MASTER augmentation: +{len(master_aug)} rows")
    else:
        master_aug = None

    # ── Encode panel features ─────────────────────────────────────────────
    # For non-augmented datasets: encode on full panel
    # For augmented: encode on combined so categories include MASTER's vocab
    if use_aug:
        combined_all = pd.concat([master_aug, panel_df], ignore_index=True)
        X_all_enc = encode_features(combined_all)
        # split back
        n_aug = len(master_aug)
        X_aug_enc = X_all_enc.iloc[:n_aug].reset_index(drop=True)
        y_aug     = master_aug['Label'].reset_index(drop=True)
        X_panel_enc = X_all_enc.iloc[n_aug:].reset_index(drop=True)
    else:
        X_panel_enc = encode_features(panel_df)
        X_aug_enc, y_aug = None, None

    # ── Hyperparameter tuning ─────────────────────────────────────────────
    print(f"  Tuning LightGBM (Optuna, 80 trials)...")
    best_params, best_cv_f1 = tune_lgbm(
        X_panel_enc, y_all,
        n_trials=80,
        cv_folds=CV_FOLDS,
        aug_df=X_aug_enc,
        aug_y=y_aug,
    )
    print(f"  Best CV macro F1 (during tuning): {best_cv_f1:.4f}")
    print(f"  Best params: {best_params}")

    # ── Repeated stratified CV with best params (for stable estimates) ────
    print(f"  Running {CV_REPEATS}×{CV_FOLDS}-fold CV for final evaluation...")
    rcv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS,
                                   random_state=SEED)
    fold_metrics  = []
    oof_probs_all = np.zeros(len(panel_df))
    oof_true_all  = np.zeros(len(panel_df), dtype=int)
    fold_count    = np.zeros(len(panel_df), dtype=int)

    for fold_i, (tr_idx, va_idx) in enumerate(rcv.split(X_panel_enc, y_all)):
        X_tr = X_panel_enc.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_all.iloc[tr_idx].reset_index(drop=True)
        X_va = X_panel_enc.iloc[va_idx].reset_index(drop=True)
        y_va = y_all.iloc[va_idx].reset_index(drop=True)

        if use_aug:
            X_tr = pd.concat([X_aug_enc, X_tr], ignore_index=True)
            y_tr = pd.concat([y_aug, y_tr], ignore_index=True)

        params = {**best_params, 'class_weight': 'balanced',
                  'random_state': SEED, 'n_jobs': -1, 'verbose': -1}
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                          lgb.log_evaluation(period=-1)])

        y_prob = m.predict_proba(X_va)[:, 1]
        y_pred = m.predict(X_va)
        fold_metrics.append(compute_metrics(y_va.values, y_pred, y_prob))

        # Accumulate OOF (averaged across repeats)
        oof_probs_all[va_idx] += y_prob
        oof_true_all[va_idx]   = y_va.values
        fold_count[va_idx]    += 1

    # Average OOF probs across repeats
    oof_probs_avg = oof_probs_all / np.maximum(fold_count, 1)
    all_oof_probs[ds_name] = (oof_true_all, oof_probs_avg)

    # Threshold optimisation on OOF
    tau, tau_f1 = optimise_threshold(oof_true_all, oof_probs_avg)
    oof_preds_opt = (oof_probs_avg >= tau).astype(int)
    all_thresholds[ds_name] = tau

    # Aggregate fold metrics
    mean_metrics = {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
    std_metrics  = {k: np.std( [f[k] for f in fold_metrics]) for k in fold_metrics[0]}
    opt_metrics  = compute_metrics(oof_true_all, oof_preds_opt, oof_probs_avg)

    print(f"\n  ── CV Results (mean ± std across {CV_REPEATS}×{CV_FOLDS} folds) ──")
    print(f"  macro F1   : {mean_metrics['macro_f1']:.4f} ± {std_metrics['macro_f1']:.4f}")
    print(f"  MCC        : {mean_metrics['mcc']:.4f} ± {std_metrics['mcc']:.4f}")
    print(f"  ROC-AUC    : {mean_metrics['roc_auc']:.4f} ± {std_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC     : {mean_metrics['pr_auc']:.4f} ± {std_metrics['pr_auc']:.4f}")
    print(f"  Sensitivity: {mean_metrics['sensitivity']:.4f} ± {std_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {mean_metrics['specificity']:.4f} ± {std_metrics['specificity']:.4f}")
    print(f"\n  ── Threshold-optimised OOF (τ={tau:.2f}) ──")
    print(f"  macro F1   : {opt_metrics['macro_f1']:.4f}")
    print(f"  MCC        : {opt_metrics['mcc']:.4f}")
    print(f"  Sensitivity: {opt_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {opt_metrics['specificity']:.4f}")

    all_results[ds_name] = {
        'mean': mean_metrics, 'std': std_metrics, 'opt': opt_metrics,
        'tau': tau, 'best_params': best_params,
        'oof_true': oof_true_all.tolist(),
        'oof_prob': oof_probs_avg.tolist(),
    }

    # ── Train final model on full data ────────────────────────────────────
    print(f"\n  Training final model on full data...")
    if use_aug:
        X_final = pd.concat([X_aug_enc, X_panel_enc], ignore_index=True)
        y_final = pd.concat([y_aug, y_all.reset_index(drop=True)], ignore_index=True)
    else:
        X_final, y_final = X_panel_enc, y_all

    final_params = {**best_params, 'class_weight': 'balanced',
                    'random_state': SEED, 'n_jobs': -1, 'verbose': -1}
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_final, y_final)
    all_models[ds_name] = final_model

    # Save model
    with open(f'{OUT_MODELS}/{ds_name}_lgbm.pkl', 'wb') as f:
        pickle.dump({'model': final_model, 'params': final_params,
                     'tau': tau, 'feature_cols': list(X_panel_enc.columns)}, f)
    print(f"  Model saved to {OUT_MODELS}/{ds_name}_lgbm.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {'MASTER': '#2196F3', 'KANSER': '#E91E63', 'PAH': '#4CAF50', 'CFTR': '#FF9800'}
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 11})

# ── 8a. ROC curves (2×2 grid) ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, ds_name in zip(axes.flat, DATASETS_ORDER):
    y_true = np.array(all_results[ds_name]['oof_true'])
    y_prob = np.array(all_results[ds_name]['oof_prob'])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = all_results[ds_name]['mean']['roc_auc']
    ax.plot(fpr, tpr, color=COLORS[ds_name], lw=2,
            label=f'AUC = {auc_val:.3f}')
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {ds_name}')
    ax.legend(loc='lower right')
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
plt.tight_layout()
plt.savefig(f'{OUT_PLOTS}/roc_curves.png', dpi=150)
plt.close()
print("\nSaved: roc_curves.png")

# ── 8b. Precision-Recall curves ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, ds_name in zip(axes.flat, DATASETS_ORDER):
    y_true = np.array(all_results[ds_name]['oof_true'])
    y_prob = np.array(all_results[ds_name]['oof_prob'])
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap_val = all_results[ds_name]['mean']['pr_auc']
    baseline = y_true.mean()
    ax.plot(rec, prec, color=COLORS[ds_name], lw=2,
            label=f'AP = {ap_val:.3f}')
    ax.axhline(baseline, color='gray', lw=1, linestyle='--',
               label=f'Baseline = {baseline:.2f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision–Recall — {ds_name}')
    ax.legend(loc='upper right')
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
plt.tight_layout()
plt.savefig(f'{OUT_PLOTS}/pr_curves.png', dpi=150)
plt.close()
print("Saved: pr_curves.png")

# ── 8c. Confusion matrices ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, ds_name in zip(axes.flat, DATASETS_ORDER):
    y_true = np.array(all_results[ds_name]['oof_true'])
    y_prob = np.array(all_results[ds_name]['oof_prob'])
    tau    = all_results[ds_name]['tau']
    y_pred = (y_prob >= tau).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    # Normalise rows for display, keep raw counts as annotations
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)"
                        for j in range(2)] for i in range(2)])
    sns.heatmap(cm_norm, annot=annot, fmt='', ax=ax,
                cmap='Blues', vmin=0, vmax=1,
                xticklabels=['Pred Benign','Pred Pathog'],
                yticklabels=['True Benign','True Pathog'],
                cbar=False)
    f1_val = all_results[ds_name]['opt']['macro_f1']
    ax.set_title(f'{ds_name}  (τ={tau:.2f}, F1={f1_val:.3f})')
plt.suptitle('Confusion Matrices — Threshold-Optimised OOF Predictions', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_PLOTS}/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrices.png")

# ── 8d. Threshold sweep — macro F1 vs τ ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, ds_name in zip(axes.flat, DATASETS_ORDER):
    y_true = np.array(all_results[ds_name]['oof_true'])
    y_prob = np.array(all_results[ds_name]['oof_prob'])
    taus = np.arange(0.05, 0.95, 0.01)
    f1s_sweep = [f1_score(y_true, (y_prob >= t).astype(int),
                          average='macro', zero_division=0) for t in taus]
    best_tau = all_results[ds_name]['tau']
    ax.plot(taus, f1s_sweep, color=COLORS[ds_name], lw=2)
    ax.axvline(best_tau, color='red', lw=1.5, linestyle='--',
               label=f'τ*={best_tau:.2f}')
    ax.axvline(0.5, color='gray', lw=1, linestyle=':', label='τ=0.5')
    ax.set_xlabel('Decision threshold τ')
    ax.set_ylabel('Macro F1')
    ax.set_title(f'{ds_name} — Threshold Sweep')
    ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_PLOTS}/threshold_sweep.png', dpi=150)
plt.close()
print("Saved: threshold_sweep.png")

# ── 8e. Feature importance (top 30) ─────────────────────────────────────────
for ds_name in DATASETS_ORDER:
    model = all_models[ds_name]
    df_panel = raw[ds_name]
    if USE_AUGMENTATION.get(ds_name, False):
        master_aug = get_augmented_train(ds_name, df_panel)
        combined   = pd.concat([master_aug, df_panel], ignore_index=True)
        X_enc = encode_features(combined)
    else:
        X_enc = encode_features(df_panel)

    importance = pd.Series(model.feature_importances_, index=X_enc.columns)
    top30 = importance.nlargest(30)

    # Color by feature group
    def group_color(col):
        if col.startswith('AL_'): return '#2196F3'
        if col.startswith('EK_'): return '#E91E63'
        if col.startswith('CAT'): return '#4CAF50'
        if col.startswith('AA_') or 'atch' in col or '_code' in col: return '#FF9800'
        return '#9E9E9E'  # miss_count etc.

    colors = [group_color(c) for c in top30.index]
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top30)), top30.values, color=colors)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (gain)')
    ax.set_title(f'{ds_name} — Top 30 Feature Importance')
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label='AL (in silico scores)'),
        Patch(facecolor='#E91E63', label='EK (conservation/CADD-like)'),
        Patch(facecolor='#4CAF50', label='CAT (population/genotype)'),
        Patch(facecolor='#FF9800', label='AA (amino acid)'),
        Patch(facecolor='#9E9E9E', label='Derived (miss_count)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUT_PLOTS}/{ds_name}_feature_importance.png', dpi=150)
    plt.close()
print("Saved: feature importance plots")

# ── 8f. Summary comparison bar chart ────────────────────────────────────────
metrics_to_plot = ['macro_f1', 'mcc', 'roc_auc', 'pr_auc']
metric_labels   = ['Macro F1', 'MCC', 'ROC-AUC', 'PR-AUC']
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
    means = [all_results[ds]['mean'][metric] for ds in DATASETS_ORDER]
    stds  = [all_results[ds]['std'][metric]  for ds in DATASETS_ORDER]
    colors_bar = [COLORS[ds] for ds in DATASETS_ORDER]
    bars = ax.bar(DATASETS_ORDER, means, yerr=stds, capsize=5,
                  color=colors_bar, alpha=0.85, edgecolor='black')
    ax.set_ylim([max(0, min(means)-0.1), 1.0])
    ax.set_ylabel(label)
    ax.set_title(label)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
plt.suptitle('Performance Summary — CV Mean ± Std', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT_PLOTS}/performance_summary.png', dpi=150)
plt.close()
print("Saved: performance_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing SHAP values...")

FEATURE_GROUPS = {
    'AL': lambda c: c.startswith('AL_'),
    'EK': lambda c: c.startswith('EK_'),
    'CAT': lambda c: c.startswith('CAT'),
    'AA': lambda c: c.startswith('AA_') or 'atch' in c or '_code' in c,
    'Derived': lambda c: c == 'miss_count',
}

for ds_name in DATASETS_ORDER:
    model = all_models[ds_name]
    df_panel = raw[ds_name]
    if USE_AUGMENTATION.get(ds_name, False):
        master_aug = get_augmented_train(ds_name, df_panel)
        combined   = pd.concat([master_aug, df_panel], ignore_index=True)
        X_enc_full = encode_features(combined)
        # Use only panel portion for SHAP (representative, not too large)
        X_shap = X_enc_full.iloc[len(master_aug):].reset_index(drop=True)
    else:
        X_shap = encode_features(df_panel)

    # Sample max 300 rows for speed
    n_shap = min(300, len(X_shap))
    rng = np.random.default_rng(SEED)
    shap_idx = rng.choice(len(X_shap), size=n_shap, replace=False)
    X_shap_sample = X_shap.iloc[shap_idx].reset_index(drop=True)

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_shap_sample)
    # LightGBM binary: shap_values returns array for class 1
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # ── Group-level SHAP bar chart ────────────────────────────────────────
    group_shap = {}
    for grp, cond in FEATURE_GROUPS.items():
        grp_cols = [i for i, c in enumerate(X_shap.columns) if cond(c)]
        if grp_cols:
            group_shap[grp] = np.abs(shap_vals[:, grp_cols]).mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    grp_names = list(group_shap.keys())
    grp_vals  = [group_shap[g] for g in grp_names]
    grp_colors = ['#2196F3','#E91E63','#4CAF50','#FF9800','#9E9E9E']
    bars = ax.bar(grp_names, grp_vals, color=grp_colors[:len(grp_names)],
                  edgecolor='black', alpha=0.85)
    for bar, val in zip(bars, grp_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Mean |SHAP value|')
    ax.set_title(f'{ds_name} — SHAP Feature Group Importance')
    plt.tight_layout()
    plt.savefig(f'{OUT_PLOTS}/{ds_name}_shap_groups.png', dpi=150)
    plt.close()

    # ── Top-20 individual SHAP beeswarm ──────────────────────────────────
    shap_mean = np.abs(shap_vals).mean(axis=0)
    top20_idx = np.argsort(shap_mean)[-20:][::-1]
    top20_cols = X_shap.columns[top20_idx].tolist()

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals[:, top20_idx],
                      X_shap_sample.iloc[:, top20_idx],
                      feature_names=top20_cols,
                      show=False, plot_type='dot', max_display=20)
    plt.title(f'{ds_name} — SHAP Beeswarm (Top 20 Features)')
    plt.tight_layout()
    plt.savefig(f'{OUT_PLOTS}/{ds_name}_shap_beeswarm.png', dpi=150,
                bbox_inches='tight')
    plt.close()

    print(f"  {ds_name}: SHAP done. Group importances: "
          + ", ".join(f"{g}={v:.4f}" for g, v in group_shap.items()))


# ─────────────────────────────────────────────────────────────────────────────
# 10. ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\nError analysis...")
error_summary = {}
for ds_name in DATASETS_ORDER:
    df_panel = raw[ds_name]
    y_true = np.array(all_results[ds_name]['oof_true'])
    y_prob = np.array(all_results[ds_name]['oof_prob'])
    tau    = all_results[ds_name]['tau']
    y_pred = (y_prob >= tau).astype(int)

    fp_mask = (y_pred == 1) & (y_true == 0)  # false positives (benign → pathog)
    fn_mask = (y_pred == 0) & (y_true == 1)  # false negatives (pathog → benign)
    tp_mask = (y_pred == 1) & (y_true == 1)
    tn_mask = (y_pred == 0) & (y_true == 0)

    n_fp = fp_mask.sum()
    n_fn = fn_mask.sum()
    n_tp = tp_mask.sum()
    n_tn = tn_mask.sum()

    # Confidence of errors (mean predicted probability for FP and FN)
    fp_conf = y_prob[fp_mask].mean() if n_fp > 0 else 0
    fn_conf = (1 - y_prob[fn_mask]).mean() if n_fn > 0 else 0  # confidence of wrong class

    # EK feature analysis for errors vs correct
    if USE_AUGMENTATION.get(ds_name, False):
        master_aug = get_augmented_train(ds_name, df_panel)
        combined   = pd.concat([master_aug, df_panel], ignore_index=True)
        X_enc_full = encode_features(combined)
        X_enc_panel = X_enc_full.iloc[len(master_aug):].reset_index(drop=True)
    else:
        X_enc_panel = encode_features(df_panel)

    ek_fn_means = X_enc_panel.loc[fn_mask, EK_COLS].mean()
    ek_fp_means = X_enc_panel.loc[fp_mask, EK_COLS].mean()
    ek_all_means = X_enc_panel[EK_COLS].mean()

    error_summary[ds_name] = {
        'TP': int(n_tp), 'TN': int(n_tn), 'FP': int(n_fp), 'FN': int(n_fn),
        'FP_avg_confidence': float(fp_conf),
        'FN_avg_confidence': float(fn_conf),
        'FN_EK7_mean': float(ek_fn_means.get('EK_7', np.nan)),
        'FP_EK7_mean': float(ek_fp_means.get('EK_7', np.nan)),
        'ALL_EK7_mean': float(ek_all_means.get('EK_7', np.nan)),
    }

    print(f"  {ds_name}: TP={n_tp} TN={n_tn} FP={n_fp} FN={n_fn} "
          f"| FP_conf={fp_conf:.3f} FN_conf={fn_conf:.3f}")
    print(f"    EK_7 mean: FN={ek_fn_means.get('EK_7',np.nan):.3f}  "
          f"FP={ek_fp_means.get('EK_7',np.nan):.3f}  All={ek_all_means.get('EK_7',np.nan):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. SAVE ALL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
# Combine results for JSON serialisation
save_pkg = {}
for ds_name in DATASETS_ORDER:
    save_pkg[ds_name] = {
        'cv_mean': all_results[ds_name]['mean'],
        'cv_std':  all_results[ds_name]['std'],
        'opt':     all_results[ds_name]['opt'],
        'tau':     all_results[ds_name]['tau'],
        'best_params': {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in all_results[ds_name]['best_params'].items()},
        'error_analysis': error_summary[ds_name],
    }

with open(f'{OUT_RESULTS}/all_results.json', 'w') as f:
    json.dump(save_pkg, f, indent=2)

# ── Final summary table ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL RESULTS SUMMARY")
print("=" * 65)
header = f"{'Dataset':8s} {'F1(CV)':>10s} {'F1(opt)':>10s} {'MCC':>8s} {'ROC-AUC':>9s} {'PR-AUC':>8s} {'τ':>6s}"
print(header)
print("-" * 65)
for ds_name in DATASETS_ORDER:
    r = all_results[ds_name]
    print(f"{ds_name:8s} "
          f"{r['mean']['macro_f1']:.4f}±{r['std']['macro_f1']:.3f} "
          f"{r['opt']['macro_f1']:>10.4f} "
          f"{r['mean']['mcc']:>8.4f} "
          f"{r['mean']['roc_auc']:>9.4f} "
          f"{r['mean']['pr_auc']:>8.4f} "
          f"{r['tau']:>6.2f}")

print(f"\nAll outputs saved to outputs/")
print("  plots/   — ROC, PR, confusion matrices, threshold sweeps, SHAP, feature importance")
print("  models/  — fitted LightGBM models (.pkl)")
print("  results/ — all_results.json")
