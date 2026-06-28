"""
PanelPath — TEKNOFEST 2026 AI in Healthcare
Missense variant pathogenicity classification (binary: Pathogenic=1, Benign=0)
Primary metric: macro F1

Architecture (data-driven, empirically validated):
  MASTER  → LightGBM, panel-only CV
  KANSER  → LightGBM, panel-only CV  (MASTER augmentation hurts: −0.016)
  PAH     → LightGBM, MASTER-augmented CV  (+0.070)
  CFTR    → LightGBM, MASTER-augmented CV  (+0.143, variance halved)

All experiments: seed=42, CPU-only, no external data.
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
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import (f1_score, matthews_corrcoef, roc_auc_score,
                              average_precision_score, confusion_matrix,
                              roc_curve, precision_recall_curve,
                              balanced_accuracy_score, brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import shap

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

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

_s = raw['MASTER']
AL_COLS  = [c for c in _s.columns if c.startswith('AL_')]   # 334 continuous
EK_COLS  = [c for c in _s.columns if c.startswith('EK_')]   # 9  continuous
CAT_COLS = [c for c in _s.columns if c.startswith('CAT_')]  # 6  categorical
AA_COLS  = [c for c in _s.columns if c.startswith('AA_')]   # 2  amino acid

# AL cols most correlated with label (missingness as signal)
# Pre-computed: 297 have |corr|>0.1, top ones reach 0.35
# Use top-50 to keep dimensionality manageable without blowing memory
_master_miss_corr = (
    raw['MASTER'][AL_COLS].isnull()
    .astype(int)
    .corrwith(raw['MASTER']['Label'])
    .abs()
    .nlargest(50)
)
TOP_MISS_COLS = list(_master_miss_corr.index)

print("=" * 65)
print("DATASET SUMMARY")
print("=" * 65)
for name, df in raw.items():
    n0 = (df['Label'] == 0).sum()
    n1 = (df['Label'] == 1).sum()
    miss = df[AL_COLS].isnull().mean().mean()
    print(f"  {name:6s}  n={len(df):4d}  B={n0}({n0/len(df)*100:.0f}%)  "
          f"P={n1}({n1/len(df)*100:.0f}%)  AL_miss={miss*100:.0f}%")
print(f"\nTop missingness-signal cols (used as binary indicators): {len(TOP_MISS_COLS)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def encode_features(train_df, val_df=None):
    """
    Build feature matrices. Fit category maps on train_df, apply to val_df.
    Returns X_train (and X_val if val_df provided).

    Features:
      - AL_*: kept as-is (LightGBM handles NaN natively)            [334]
      - EK_*: kept as-is                                             [9]
      - miss_count: total missing AL cols per sample                 [1]
      - miss_{col}: binary indicator for top-50 most correlated AL   [50]
      - CAT_1..5: label-encoded (NaN → own category)                 [5]
      - CAT_6_present: binary flag (nearly always null)              [1]
      - AA_1, AA_2: label-encoded (handles multi-char + * + NA)      [2]
    Total: 334 + 9 + 1 + 50 + 5 + 1 + 2 = 402 features
    """
    dfs = [train_df] + ([val_df] if val_df is not None else [])

    # ── Build category maps from train only ──────────────────────────────
    cat_maps = {}
    for c in [c for c in CAT_COLS if c != 'CAT_6']:
        tr_vals = train_df[c].fillna('__NA__').astype(str)
        cats = sorted(tr_vals.unique())
        cat_maps[c] = {v: i for i, v in enumerate(cats)}

    aa_maps = {}
    for c in AA_COLS:
        tr_vals = train_df[c].fillna('__NA__').astype(str)
        cats = sorted(tr_vals.unique())
        aa_maps[c] = {v: i for i, v in enumerate(cats)}

    results = []
    for df in dfs:
        X = df[AL_COLS + EK_COLS].copy()

        # Missingness count
        X['miss_count'] = df[AL_COLS].isnull().sum(axis=1).astype(float)

        # Top-50 missingness indicators (binary, 0/1)
        for c in TOP_MISS_COLS:
            X[f'miss_{c}'] = df[c].isnull().astype(float)

        # CAT encoding (unseen label in val → NaN, fine for LightGBM)
        for c in [c for c in CAT_COLS if c != 'CAT_6']:
            vals = df[c].fillna('__NA__').astype(str)
            X[c] = vals.map(cat_maps[c]).astype(float)  # unseen → NaN
        # CAT_6: binary present/absent (97-100% null, single-valued in panels)
        X['CAT_6_present'] = (~df['CAT_6'].isnull()).astype(float)

        # AA encoding: pure label encoding
        # (Atchley factors discarded — LightGBM doesn't need physicochemical
        #  structure; label encoding lets tree splits work naturally)
        for c in AA_COLS:
            vals = df[c].fillna('__NA__').astype(str)
            X[c] = vals.map(aa_maps[c]).astype(float)  # unseen → NaN

        results.append(X)

    return tuple(results) if val_df is not None else results[0]


# ─────────────────────────────────────────────────────────────────────────────
# 3. AUGMENTATION CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MASTER_DF = raw['MASTER']

def get_master_aug(panel_name, panel_df):
    panel_ids = set(panel_df['Variant_ID'])
    return MASTER_DF[~MASTER_DF['Variant_ID'].isin(panel_ids)].reset_index(drop=True)

# Empirically: augmentation helps PAH (+0.07) and CFTR (+0.14), hurts KANSER
USE_AUG = {'MASTER': False, 'KANSER': False, 'PAH': True, 'CFTR': True}


# ─────────────────────────────────────────────────────────────────────────────
# 4. OPTUNA TUNING
# ─────────────────────────────────────────────────────────────────────────────
def tune_lgbm(X_panel, y_panel, n_trials=80, aug_X=None, aug_y=None):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int('n_estimators', 300, 2000),
            learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            num_leaves        = trial.suggest_int('num_leaves', 15, 255),
            min_child_samples = trial.suggest_int('min_child_samples', 5, 100),
            subsample         = trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree  = trial.suggest_float('colsample_bytree', 0.3, 1.0),
            reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            class_weight='balanced', random_state=SEED, n_jobs=-1, verbose=-1,
        )
        fold_f1s = []
        for tr_idx, va_idx in cv.split(X_panel, y_panel):
            X_tr = X_panel.iloc[tr_idx].reset_index(drop=True)
            y_tr = y_panel.iloc[tr_idx].reset_index(drop=True)
            X_va = X_panel.iloc[va_idx].reset_index(drop=True)
            y_va = y_panel.iloc[va_idx]
            if aug_X is not None:
                X_tr = pd.concat([aug_X, X_tr], ignore_index=True)
                y_tr = pd.concat([aug_y.reset_index(drop=True), y_tr], ignore_index=True)
            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(period=-1)])
            fold_f1s.append(f1_score(y_va, m.predict(X_va), average='macro'))
        return np.mean(fold_f1s)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value, study


# ─────────────────────────────────────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        'macro_f1':    f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_benign':   f1_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0),
        'f1_pathog':   f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0),
        'mcc':         matthews_corrcoef(y_true, y_pred),
        'roc_auc':     roc_auc_score(y_true, y_prob),
        'pr_auc':      average_precision_score(y_true, y_prob),
        'bal_acc':     balanced_accuracy_score(y_true, y_pred),
        'brier':       brier_score_loss(y_true, y_prob),
        'sensitivity': tp / max(1, tp + fn),
        'specificity': tn / max(1, tn + fp),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    }

def optimise_threshold(y_true, y_prob):
    grid = np.arange(0.05, 0.95, 0.01)
    best_tau, best_f1 = 0.5, 0.0
    for tau in grid:
        f = f1_score(y_true, (y_prob >= tau).astype(int),
                     average='macro', zero_division=0)
        if f > best_f1:
            best_f1, best_tau = f, tau
    return best_tau, best_f1


# ─────────────────────────────────────────────────────────────────────────────
# 6. BASELINE (MASTER-trained, applied to all panels — no specialisation)
# ─────────────────────────────────────────────────────────────────────────────
print("\nTraining BASELINE (MASTER model applied to all panels)...")
_X_master_base = encode_features(MASTER_DF)
_y_master_base = MASTER_DF['Label']
_baseline_model = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    class_weight='balanced', random_state=SEED, n_jobs=-1, verbose=-1
)
# Train on full MASTER for baseline (cross-dataset evaluation)
_baseline_model.fit(_X_master_base, _y_master_base)

baseline_results = {}
for ds_name in ['MASTER', 'KANSER', 'PAH', 'CFTR']:
    df = raw[ds_name]
    # Encode using MASTER vocab (baseline assumes MASTER column distributions)
    X_b = encode_features(MASTER_DF, df)[1]
    y_b = df['Label'].values
    prob_b = _baseline_model.predict_proba(X_b)[:, 1]
    tau_b, _ = optimise_threshold(y_b, prob_b)
    pred_b = (prob_b >= tau_b).astype(int)
    baseline_results[ds_name] = compute_metrics(y_b, pred_b, prob_b)
    print(f"  Baseline {ds_name}: F1={baseline_results[ds_name]['macro_f1']:.4f}  "
          f"ROC-AUC={baseline_results[ds_name]['roc_auc']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
all_results    = {}
all_models     = {}
all_oof        = {}  # ds → (y_true, y_prob)
all_fold_metrics = {}  # ds → list of per-fold metric dicts

DATASETS_ORDER = ['MASTER', 'KANSER', 'PAH', 'CFTR']

for ds_name in DATASETS_ORDER:
    print(f"\n{'='*65}")
    print(f"  DATASET: {ds_name}")
    print(f"{'='*65}")

    panel_df = raw[ds_name]
    y_all    = panel_df['Label']

    # ── Pre-encode (categories fitted on combined train+aug vocab) ─────────
    use_aug = USE_AUG[ds_name]
    if use_aug:
        master_aug = get_master_aug(ds_name, panel_df)
        combined   = pd.concat([master_aug, panel_df], ignore_index=True)
        X_combined = encode_features(combined)
        n_aug      = len(master_aug)
        X_aug      = X_combined.iloc[:n_aug].reset_index(drop=True)
        y_aug      = master_aug['Label'].reset_index(drop=True)
        X_panel    = X_combined.iloc[n_aug:].reset_index(drop=True)
        print(f"  MASTER augmentation: +{n_aug} rows")
    else:
        X_panel = encode_features(panel_df)
        X_aug, y_aug = None, None

    # ── Optuna tuning ──────────────────────────────────────────────────────
    print(f"  Optuna tuning (80 trials)...")
    best_params, best_cv_f1, study = tune_lgbm(
        X_panel, y_all, n_trials=80,
        aug_X=X_aug, aug_y=y_aug
    )
    print(f"  Best tuning CV F1: {best_cv_f1:.4f}")

    # ── Repeated stratified CV for stable estimates ────────────────────────
    print(f"  3×5-fold CV (final evaluation)...")
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    fold_metrics   = []
    oof_probs      = np.zeros(len(panel_df))
    oof_true       = np.zeros(len(panel_df), dtype=int)
    fold_counts    = np.zeros(len(panel_df), dtype=int)

    model_params = {**best_params, 'class_weight': 'balanced',
                    'random_state': SEED, 'n_jobs': -1, 'verbose': -1}

    for tr_idx, va_idx in rcv.split(X_panel, y_all):
        X_tr = X_panel.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_all.iloc[tr_idx].reset_index(drop=True)
        X_va = X_panel.iloc[va_idx].reset_index(drop=True)
        y_va = y_all.iloc[va_idx].reset_index(drop=True)

        if use_aug:
            X_tr = pd.concat([X_aug, X_tr], ignore_index=True)
            y_tr = pd.concat([y_aug, y_tr], ignore_index=True)

        m = lgb.LGBMClassifier(**model_params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                          lgb.log_evaluation(period=-1)])

        y_prob_fold = m.predict_proba(X_va)[:, 1]
        y_pred_fold = m.predict(X_va)
        fold_metrics.append(compute_metrics(y_va.values, y_pred_fold, y_prob_fold))

        oof_probs[va_idx]   += y_prob_fold
        oof_true[va_idx]     = y_va.values
        fold_counts[va_idx] += 1

    oof_probs_avg = oof_probs / np.maximum(fold_counts, 1)
    all_oof[ds_name] = (oof_true, oof_probs_avg)
    all_fold_metrics[ds_name] = fold_metrics

    # Threshold optimisation on OOF
    tau, _ = optimise_threshold(oof_true, oof_probs_avg)
    oof_preds_opt = (oof_probs_avg >= tau).astype(int)
    opt_metrics = compute_metrics(oof_true, oof_preds_opt, oof_probs_avg)

    mean_m = {k: float(np.mean([f[k] for f in fold_metrics])) for k in fold_metrics[0]}
    std_m  = {k: float(np.std( [f[k] for f in fold_metrics])) for k in fold_metrics[0]}

    print(f"\n  ── CV (3×5-fold mean ± std) ──")
    print(f"  macro_F1   : {mean_m['macro_f1']:.4f} ± {std_m['macro_f1']:.4f}")
    print(f"  f1_benign  : {mean_m['f1_benign']:.4f} ± {std_m['f1_benign']:.4f}")
    print(f"  f1_pathog  : {mean_m['f1_pathog']:.4f} ± {std_m['f1_pathog']:.4f}")
    print(f"  MCC        : {mean_m['mcc']:.4f} ± {std_m['mcc']:.4f}")
    print(f"  ROC-AUC    : {mean_m['roc_auc']:.4f} ± {std_m['roc_auc']:.4f}")
    print(f"  PR-AUC     : {mean_m['pr_auc']:.4f} ± {std_m['pr_auc']:.4f}")
    print(f"  Brier      : {mean_m['brier']:.4f} ± {std_m['brier']:.4f}")
    print(f"  Sensitivity: {mean_m['sensitivity']:.4f} ± {std_m['sensitivity']:.4f}")
    print(f"  Specificity: {mean_m['specificity']:.4f} ± {std_m['specificity']:.4f}")
    print(f"\n  ── Threshold-opt OOF (τ={tau:.2f}) ──")
    print(f"  macro_F1   : {opt_metrics['macro_f1']:.4f}")
    print(f"  MCC        : {opt_metrics['mcc']:.4f}  Brier: {opt_metrics['brier']:.4f}")
    print(f"  Sensitivity: {opt_metrics['sensitivity']:.4f}  Specificity: {opt_metrics['specificity']:.4f}")
    print(f"  CM: TP={opt_metrics['TP']} TN={opt_metrics['TN']} FP={opt_metrics['FP']} FN={opt_metrics['FN']}")

    all_results[ds_name] = {
        'mean': mean_m, 'std': std_m, 'opt': opt_metrics, 'tau': tau,
        'best_params': {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                        for k, v in best_params.items()},
        'baseline': baseline_results[ds_name],
    }

    # ── Train final model on full data ─────────────────────────────────────
    print(f"\n  Final model (full data)...")
    if use_aug:
        X_final = pd.concat([X_aug, X_panel], ignore_index=True)
        y_final = pd.concat([y_aug, y_all.reset_index(drop=True)], ignore_index=True)
    else:
        X_final, y_final = X_panel, y_all.reset_index(drop=True)

    final_model = lgb.LGBMClassifier(**model_params)
    final_model.fit(X_final, y_final)
    all_models[ds_name] = (final_model, list(X_panel.columns), tau)
    with open(f'{OUT_MODELS}/{ds_name}_lgbm.pkl', 'wb') as f:
        pickle.dump({'model': final_model, 'feature_cols': list(X_panel.columns),
                     'tau': tau, 'params': model_params}, f)
    print(f"  Saved to {OUT_MODELS}/{ds_name}_lgbm.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {'MASTER': '#2196F3', 'KANSER': '#E91E63', 'PAH': '#4CAF50', 'CFTR': '#FF9800'}
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12})

# ── ROC curves ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, ds in zip(axes.flat, DATASETS_ORDER):
    yt, yp = all_oof[ds]
    fpr, tpr, _ = roc_curve(yt, yp)
    ax.plot(fpr, tpr, color=COLORS[ds], lw=2,
            label=f'AUC={all_results[ds]["mean"]["roc_auc"]:.3f}')
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC — {ds}'); ax.legend(loc='lower right')
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
plt.tight_layout()
plt.savefig(f'{OUT_PLOTS}/roc_curves.png', dpi=150); plt.close()

# ── PR curves ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, ds in zip(axes.flat, DATASETS_ORDER):
    yt, yp = all_oof[ds]
    prec, rec, _ = precision_recall_curve(yt, yp)
    ax.plot(rec, prec, color=COLORS[ds], lw=2,
            label=f'AP={all_results[ds]["mean"]["pr_auc"]:.3f}')
    ax.axhline(yt.mean(), color='gray', lw=1, linestyle='--',
               label=f'Baseline={yt.mean():.2f}')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'Precision–Recall — {ds}'); ax.legend()
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
plt.tight_layout()
plt.savefig(f'{OUT_PLOTS}/pr_curves.png', dpi=150); plt.close()

# ── Confusion matrices ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, ds in zip(axes.flat, DATASETS_ORDER):
    yt, yp = all_oof[ds]
    tau = all_results[ds]['tau']
    cm = confusion_matrix(yt, (yp >= tau).astype(int))
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{cm[i,j]}\n({cm_n[i,j]*100:.1f}%)" for j in range(2)]
                       for i in range(2)])
    sns.heatmap(cm_n, annot=annot, fmt='', ax=ax, cmap='Blues',
                vmin=0, vmax=1, cbar=False,
                xticklabels=['Pred Benign','Pred Pathog'],
                yticklabels=['True Benign','True Pathog'])
    ax.set_title(f'{ds}  τ={tau:.2f}  F1={all_results[ds]["opt"]["macro_f1"]:.3f}')
plt.suptitle('Confusion Matrices (OOF, threshold-optimised)', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_PLOTS}/confusion_matrices.png', dpi=150, bbox_inches='tight'); plt.close()

# ── Threshold sweep ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, ds in zip(axes.flat, DATASETS_ORDER):
    yt, yp = all_oof[ds]
    taus = np.arange(0.05, 0.95, 0.01)
    f1s  = [f1_score(yt, (yp>=t).astype(int), average='macro', zero_division=0)
            for t in taus]
    tau_opt = all_results[ds]['tau']
    ax.plot(taus, f1s, color=COLORS[ds], lw=2)
    ax.axvline(tau_opt, color='red', lw=1.5, linestyle='--', label=f'τ*={tau_opt:.2f}')
    ax.axvline(0.5, color='gray', lw=1, linestyle=':', label='τ=0.5')
    ax.set_xlabel('τ'); ax.set_ylabel('Macro F1')
    ax.set_title(f'{ds} — Threshold Sweep'); ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_PLOTS}/threshold_sweep.png', dpi=150); plt.close()

# ── Baseline vs Pipeline comparison ─────────────────────────────────────────
metrics_compare = ['macro_f1', 'roc_auc', 'pr_auc', 'mcc']
labels_compare  = ['Macro F1', 'ROC-AUC', 'PR-AUC', 'MCC']
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
x = np.arange(len(DATASETS_ORDER))
width = 0.35
for ax, metric, label in zip(axes, metrics_compare, labels_compare):
    base_vals = [baseline_results[ds][metric] for ds in DATASETS_ORDER]
    pipe_vals = [all_results[ds]['mean'][metric] for ds in DATASETS_ORDER]
    pipe_stds = [all_results[ds]['std'][metric]  for ds in DATASETS_ORDER]
    ax.bar(x - width/2, base_vals, width, label='Baseline', color='#9E9E9E',
           alpha=0.8, edgecolor='black')
    ax.bar(x + width/2, pipe_vals, width, label='Pipeline', yerr=pipe_stds,
           capsize=4, color=[COLORS[d] for d in DATASETS_ORDER], alpha=0.85,
           edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(DATASETS_ORDER)
    ax.set_ylabel(label); ax.set_title(label)
    ax.legend(fontsize=9)
    ymin = max(0, min(min(base_vals), min(pipe_vals)) - 0.08)
    ax.set_ylim([ymin, 1.0])
    for xi, (bv, pv) in enumerate(zip(base_vals, pipe_vals)):
        delta = pv - bv
        ax.text(xi + width/2, pv + (pipe_stds[xi] or 0) + 0.01,
                f'+{delta:.3f}' if delta >= 0 else f'{delta:.3f}',
                ha='center', va='bottom', fontsize=8,
                color='green' if delta >= 0 else 'red')
plt.suptitle('Baseline vs Pipeline — All Panels', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT_PLOTS}/baseline_vs_pipeline.png', dpi=150); plt.close()
print("Saved comparison plots")

# ── Per-dataset feature importance ──────────────────────────────────────────
for ds_name in DATASETS_ORDER:
    final_model, feat_cols, _ = all_models[ds_name]
    importance = pd.Series(final_model.feature_importances_, index=feat_cols)
    top30 = importance.nlargest(30)

    def group_color(col):
        if col.startswith('miss_A'): return '#78909C'   # missingness indicators
        if col.startswith('AL_'):    return '#2196F3'
        if col.startswith('EK_'):    return '#E91E63'
        if 'CAT' in col:             return '#4CAF50'
        if col.startswith('AA_'):    return '#FF9800'
        return '#9E9E9E'

    from matplotlib.patches import Patch
    palette = [group_color(c) for c in top30.index]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top30)), top30.values, color=palette)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (gain)')
    ax.set_title(f'{ds_name} — Top 30 Features')
    legend_elements = [
        Patch(facecolor='#2196F3', label='AL — in silico scores'),
        Patch(facecolor='#E91E63', label='EK — conservation/CADD'),
        Patch(facecolor='#4CAF50', label='CAT — population/genotype'),
        Patch(facecolor='#FF9800', label='AA — amino acid'),
        Patch(facecolor='#78909C', label='miss_* — missingness indicators'),
        Patch(facecolor='#9E9E9E', label='Derived (miss_count)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUT_PLOTS}/{ds_name}_feature_importance.png', dpi=150)
    plt.close()
print("Saved feature importance plots")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\nSHAP analysis...")

FEATURE_GROUPS = {
    'AL (in silico)':          lambda c: c.startswith('AL_'),
    'EK (conservation/CADD)':  lambda c: c.startswith('EK_'),
    'CAT (population)':        lambda c: 'CAT' in c,
    'AA (amino acid)':         lambda c: c.startswith('AA_'),
    'Missingness indicators':  lambda c: c.startswith('miss_'),
}

for ds_name in DATASETS_ORDER:
    final_model, feat_cols, _ = all_models[ds_name]
    panel_df = raw[ds_name]

    if USE_AUG[ds_name]:
        master_aug = get_master_aug(ds_name, panel_df)
        combined   = pd.concat([master_aug, panel_df], ignore_index=True)
        X_full = encode_features(combined)
        X_shap = X_full.iloc[len(master_aug):].reset_index(drop=True)
    else:
        X_shap = encode_features(panel_df)

    # Sample for speed
    rng = np.random.default_rng(SEED)
    n_shap = min(300, len(X_shap))
    idx = rng.choice(len(X_shap), size=n_shap, replace=False)
    X_s = X_shap.iloc[idx].reset_index(drop=True)

    explainer = shap.TreeExplainer(final_model)
    shap_vals = explainer.shap_values(X_s)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # Group-level bar chart
    group_means = {}
    for grp, cond in FEATURE_GROUPS.items():
        cols_idx = [i for i, c in enumerate(feat_cols) if cond(c)]
        if cols_idx:
            group_means[grp] = float(np.abs(shap_vals[:, cols_idx]).mean())

    fig, ax = plt.subplots(figsize=(9, 4))
    gnames = list(group_means.keys())
    gvals  = [group_means[g] for g in gnames]
    gcols  = ['#2196F3','#E91E63','#4CAF50','#FF9800','#78909C'][:len(gnames)]
    bars = ax.bar(gnames, gvals, color=gcols, edgecolor='black', alpha=0.85)
    for bar, val in zip(bars, gvals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Mean |SHAP value|')
    ax.set_title(f'{ds_name} — SHAP Group Importance')
    ax.set_xticklabels(gnames, rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUT_PLOTS}/{ds_name}_shap_groups.png', dpi=150); plt.close()

    # Individual beeswarm (top 20)
    top20_idx = np.argsort(np.abs(shap_vals).mean(axis=0))[-20:][::-1]
    top20_cols = [feat_cols[i] for i in top20_idx]
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals[:, top20_idx],
                      X_s.iloc[:, top20_idx],
                      feature_names=top20_cols,
                      show=False, plot_type='dot', max_display=20)
    plt.title(f'{ds_name} — SHAP Beeswarm (Top 20)')
    plt.tight_layout()
    plt.savefig(f'{OUT_PLOTS}/{ds_name}_shap_beeswarm.png', dpi=150,
                bbox_inches='tight'); plt.close()

    print(f"  {ds_name}: " + ", ".join(f"{g.split()[0]}={v:.4f}"
                                        for g, v in group_means.items()))


# ─────────────────────────────────────────────────────────────────────────────
# 10. ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\nError analysis...")
error_analysis = {}
for ds_name in DATASETS_ORDER:
    panel_df = raw[ds_name]
    yt, yp   = all_oof[ds_name]
    tau      = all_results[ds_name]['tau']
    ypred    = (yp >= tau).astype(int)

    fp = (ypred==1) & (yt==0)
    fn = (ypred==0) & (yt==1)

    if USE_AUG[ds_name]:
        master_aug = get_master_aug(ds_name, panel_df)
        X_enc = encode_features(pd.concat([master_aug, panel_df], ignore_index=True))
        X_panel_enc = X_enc.iloc[len(master_aug):].reset_index(drop=True)
    else:
        X_panel_enc = encode_features(panel_df)

    def ek_means(mask):
        return {c: float(X_panel_enc.loc[mask, c].mean())
                for c in EK_COLS if c in X_panel_enc.columns}

    error_analysis[ds_name] = {
        'FP_count': int(fp.sum()),
        'FN_count': int(fn.sum()),
        'FP_avg_prob': float(yp[fp].mean()) if fp.sum() > 0 else None,
        'FN_avg_prob': float(yp[fn].mean()) if fn.sum() > 0 else None,
        'FP_EK_means': ek_means(fp) if fp.sum() > 0 else {},
        'FN_EK_means': ek_means(fn) if fn.sum() > 0 else {},
        'ALL_EK_means': ek_means(np.ones(len(panel_df), dtype=bool)),
    }
    print(f"  {ds_name}: FP={fp.sum()} FN={fn.sum()} | "
          f"FP_EK7={error_analysis[ds_name]['FP_EK_means'].get('EK_7',0):.3f} "
          f"FN_EK7={error_analysis[ds_name]['FN_EK_means'].get('EK_7',0):.3f} "
          f"All_EK7={error_analysis[ds_name]['ALL_EK_means'].get('EK_7',0):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. SAVE ALL RESULTS TO JSON
# ─────────────────────────────────────────────────────────────────────────────
def make_serialisable(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serialisable(v) for v in obj]
    return obj

save_pkg = {}
for ds_name in DATASETS_ORDER:
    save_pkg[ds_name] = {
        'n_samples':     len(raw[ds_name]),
        'class_balance': {
            'benign':    int((raw[ds_name]['Label']==0).sum()),
            'pathogenic':int((raw[ds_name]['Label']==1).sum()),
        },
        'cv_mean':       all_results[ds_name]['mean'],
        'cv_std':        all_results[ds_name]['std'],
        'opt_metrics':   all_results[ds_name]['opt'],
        'tau':           all_results[ds_name]['tau'],
        'best_params':   all_results[ds_name]['best_params'],
        'baseline':      baseline_results[ds_name],
        'error_analysis':error_analysis[ds_name],
        'augmentation':  USE_AUG[ds_name],
    }

with open(f'{OUT_RESULTS}/all_results.json', 'w') as f:
    json.dump(make_serialisable(save_pkg), f, indent=2)
print(f"\nSaved: {OUT_RESULTS}/all_results.json")


# ─────────────────────────────────────────────────────────────────────────────
# 12. FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FINAL RESULTS — PIPELINE vs BASELINE")
print("=" * 80)
print(f"{'Dataset':8s} | {'Base_F1':>8s} | {'CV_F1':>15s} | {'Opt_F1':>8s} | "
      f"{'MCC':>7s} | {'ROC-AUC':>8s} | {'PR-AUC':>8s} | {'Brier':>7s} | τ")
print("-" * 80)
for ds_name in DATASETS_ORDER:
    r = all_results[ds_name]
    b = baseline_results[ds_name]
    print(f"{ds_name:8s} | {b['macro_f1']:>8.4f} | "
          f"{r['mean']['macro_f1']:.4f}±{r['std']['macro_f1']:.3f} | "
          f"{r['opt']['macro_f1']:>8.4f} | "
          f"{r['mean']['mcc']:>7.4f} | "
          f"{r['mean']['roc_auc']:>8.4f} | "
          f"{r['mean']['pr_auc']:>8.4f} | "
          f"{r['mean']['brier']:>7.4f} | "
          f"{r['tau']:.2f}")

print("\nOutputs:")
print(f"  plots/  : ROC, PR, confusion matrices, threshold sweeps,")
print(f"            baseline comparison, SHAP groups & beeswarms,")
print(f"            feature importance (all datasets)")
print(f"  models/ : fitted LightGBM models (.pkl)")
print(f"  results/: all_results.json (all numbers for report)")
