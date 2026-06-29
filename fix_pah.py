"""
fix_pah.py — PAH macro F1 improvement experiments
Current baseline: 0.748 macro F1, specificity 0.51 (62 benign vs 310 pathogenic)
Tests: SMOTE, stacking, TabPFN, feature selection, threshold tuning, combinations
"""

import os, warnings, json
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

SEED = 42
np.random.seed(SEED)

# ── Load data (same as main pipeline) ────────────────────────────────────────
DATA_DIR = 'data'
raw = {
    'MASTER': pd.read_csv(f'{DATA_DIR}/YARISMA_TRAIN_MASTER.csv'),
    'PAH':    pd.read_csv(f'{DATA_DIR}/YARISMA_TRAIN_PAH.csv'),
}

_s = raw['MASTER']
AL_COLS  = [c for c in _s.columns if c.startswith('AL_')]
EK_COLS  = [c for c in _s.columns if c.startswith('EK_')]
CAT_COLS = [c for c in _s.columns if c.startswith('CAT_')]
AA_COLS  = [c for c in _s.columns if c.startswith('AA_')]

_master_miss_corr = (
    raw['MASTER'][AL_COLS].isnull()
    .astype(int).corrwith(raw['MASTER']['Label']).abs().nlargest(50)
)
TOP_MISS_COLS = list(_master_miss_corr.index)

MASTER_DF = raw['MASTER']
PAH_DF    = raw['PAH']

print(f"PAH: {len(PAH_DF)} samples | Benign={( PAH_DF['Label']==0).sum()} Pathogenic={(PAH_DF['Label']==1).sum()}")
print(f"MASTER (non-PAH aug rows): {len(MASTER_DF) - PAH_DF['Variant_ID'].isin(MASTER_DF['Variant_ID']).sum()}")

# ── Feature engineering (identical to main pipeline) ─────────────────────────
def encode_features(train_df, val_df=None):
    dfs = [train_df] + ([val_df] if val_df is not None else [])
    cat_maps = {}
    for c in [c for c in CAT_COLS if c != 'CAT_6']:
        tr_vals = train_df[c].fillna('__NA__').astype(str)
        cat_maps[c] = {v: i for i, v in enumerate(sorted(tr_vals.unique()))}
    aa_maps = {}
    for c in AA_COLS:
        tr_vals = train_df[c].fillna('__NA__').astype(str)
        aa_maps[c] = {v: i for i, v in enumerate(sorted(tr_vals.unique()))}

    results = []
    for df in dfs:
        X = df[AL_COLS + EK_COLS].copy()
        X['miss_count'] = df[AL_COLS].isnull().sum(axis=1).astype(float)
        for c in TOP_MISS_COLS:
            X[f'miss_{c}'] = df[c].isnull().astype(float)
        for c in [c for c in CAT_COLS if c != 'CAT_6']:
            X[c] = df[c].fillna('__NA__').astype(str).map(cat_maps[c]).astype(float)
        X['CAT_6_present'] = (~df['CAT_6'].isnull()).astype(float)
        for c in AA_COLS:
            X[c] = df[c].fillna('__NA__').astype(str).map(aa_maps[c]).astype(float)
        results.append(X)
    return tuple(results) if val_df is not None else results[0]

def get_master_aug(panel_df):
    panel_ids = set(panel_df['Variant_ID'])
    return MASTER_DF[~MASTER_DF['Variant_ID'].isin(panel_ids)].reset_index(drop=True)

def optimise_threshold(y_true, y_prob):
    best_tau, best_f1 = 0.5, 0.0
    for tau in np.arange(0.05, 0.95, 0.01):
        f = f1_score(y_true, (y_prob >= tau).astype(int), average='macro', zero_division=0)
        if f > best_f1:
            best_f1, best_tau = f, tau
    return best_tau, best_f1

# Pre-encode PAH with MASTER aug vocab
master_aug = get_master_aug(PAH_DF)
combined   = pd.concat([master_aug, PAH_DF], ignore_index=True)
X_combined = encode_features(combined)
n_aug      = len(master_aug)
X_aug      = X_combined.iloc[:n_aug].reset_index(drop=True)
y_aug      = master_aug['Label'].reset_index(drop=True)
X_pah      = X_combined.iloc[n_aug:].reset_index(drop=True)
y_pah      = PAH_DF['Label'].reset_index(drop=True)

FEAT_COLS  = list(X_pah.columns)
CV5        = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Best LightGBM params from main pipeline run
BEST_PARAMS = dict(
    n_estimators=874, learning_rate=0.254, num_leaves=191,
    min_child_samples=62, subsample=0.578, colsample_bytree=0.494,
    reg_alpha=0.000195, reg_lambda=2.14,
    class_weight='balanced', random_state=SEED, n_jobs=-1, verbose=-1
)

results = {}

def cv_eval(name, get_preds_fn):
    """Run 5-fold CV, get OOF probs, optimise threshold, report."""
    oof_prob = np.zeros(len(y_pah))
    for tr_idx, va_idx in CV5.split(X_pah, y_pah):
        oof_prob[va_idx] = get_preds_fn(tr_idx, va_idx)
    tau, f1 = optimise_threshold(y_pah.values, oof_prob)
    pred = (oof_prob >= tau).astype(int)
    sens = ((pred == 1) & (y_pah.values == 1)).sum() / (y_pah.values == 1).sum()
    spec = ((pred == 0) & (y_pah.values == 0)).sum() / (y_pah.values == 0).sum()
    auc  = roc_auc_score(y_pah.values, oof_prob)
    results[name] = {'f1': f1, 'tau': tau, 'sens': sens, 'spec': spec, 'auc': auc}
    print(f"  [{name}]  F1={f1:.4f}  τ={tau:.2f}  Sens={sens:.3f}  Spec={spec:.3f}  AUC={auc:.3f}")
    return f1


print("\n" + "="*70)
print("0. BASELINE (main pipeline config — augmented, tuned params)")
print("="*70)

def baseline_preds(tr_idx, va_idx):
    X_tr = pd.concat([X_aug, X_pah.iloc[tr_idx]], ignore_index=True)
    y_tr = pd.concat([y_aug, y_pah.iloc[tr_idx]], ignore_index=True)
    m = lgb.LGBMClassifier(**BEST_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_pah.iloc[va_idx], y_pah.iloc[va_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return m.predict_proba(X_pah.iloc[va_idx])[:, 1]

cv_eval('0_baseline', baseline_preds)


print("\n" + "="*70)
print("1. SMOTE oversampling on benign class (inside fold)")
print("="*70)
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

for smote_name, smote_cls in [
    ('SMOTE', SMOTE(random_state=SEED, k_neighbors=3)),
    ('BorderlineSMOTE', BorderlineSMOTE(random_state=SEED, k_neighbors=3)),
    ('ADASYN', ADASYN(random_state=SEED, n_neighbors=3)),
]:
    def smote_preds(tr_idx, va_idx, _s=smote_cls, _n=smote_name):
        X_tr = pd.concat([X_aug, X_pah.iloc[tr_idx]], ignore_index=True)
        y_tr = pd.concat([y_aug, y_pah.iloc[tr_idx]], ignore_index=True)
        # Fill NaN for SMOTE (needs complete data)
        X_tr_filled = X_tr.fillna(X_tr.median())
        try:
            X_res, y_res = _s.fit_resample(X_tr_filled, y_tr)
            X_res = pd.DataFrame(X_res, columns=X_tr.columns)
        except Exception as e:
            X_res, y_res = X_tr, y_tr
        m = lgb.LGBMClassifier(**BEST_PARAMS)
        X_va = X_pah.iloc[va_idx]
        m.fit(X_res, y_res, eval_set=[(X_va, y_pah.iloc[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        return m.predict_proba(X_va)[:, 1]
    cv_eval(f'1_{smote_name}', smote_preds)


print("\n" + "="*70)
print("2. Two-stage stacking (MASTER LightGBM prob as extra feature)")
print("="*70)

# Train MASTER model on full MASTER data
X_master_full = encode_features(MASTER_DF)
y_master_full = MASTER_DF['Label']
master_model  = lgb.LGBMClassifier(**BEST_PARAMS)
master_model.fit(X_master_full, y_master_full)

# Add master prob as feature to PAH
pah_master_prob = master_model.predict_proba(X_pah)[:, 1]
X_pah_stack = X_pah.copy()
X_pah_stack['master_prob'] = pah_master_prob

aug_master_prob = master_model.predict_proba(X_aug)[:, 1]
X_aug_stack = X_aug.copy()
X_aug_stack['master_prob'] = aug_master_prob

def stack_preds(tr_idx, va_idx):
    X_tr = pd.concat([X_aug_stack, X_pah_stack.iloc[tr_idx]], ignore_index=True)
    y_tr = pd.concat([y_aug, y_pah.iloc[tr_idx]], ignore_index=True)
    m = lgb.LGBMClassifier(**BEST_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_pah_stack.iloc[va_idx], y_pah.iloc[va_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return m.predict_proba(X_pah_stack.iloc[va_idx])[:, 1]

cv_eval('2_stacking', stack_preds)


print("\n" + "="*70)
print("3. TabPFN")
print("="*70)
try:
    from tabpfn import TabPFNClassifier
    # TabPFN works on raw data, max 1000 samples, no NaN
    X_pah_tfpn = X_pah.fillna(X_pah.median()).values
    tfpn_oof = np.zeros(len(y_pah))
    for tr_idx, va_idx in CV5.split(X_pah_tfpn, y_pah):
        # TabPFN max 1000 train samples — sample if needed
        if len(tr_idx) > 1000:
            rng = np.random.default_rng(SEED)
            tr_idx = rng.choice(tr_idx, 1000, replace=False)
        clf = TabPFNClassifier(device='cpu', N_ensemble_configurations=8)
        clf.fit(X_pah_tfpn[tr_idx], y_pah.values[tr_idx])
        tfpn_oof[va_idx] = clf.predict_proba(X_pah_tfpn[va_idx])[:, 1]
    tau, f1 = optimise_threshold(y_pah.values, tfpn_oof)
    pred = (tfpn_oof >= tau).astype(int)
    sens = ((pred==1)&(y_pah.values==1)).sum()/(y_pah.values==1).sum()
    spec = ((pred==0)&(y_pah.values==0)).sum()/(y_pah.values==0).sum()
    auc  = roc_auc_score(y_pah.values, tfpn_oof)
    results['3_TabPFN'] = {'f1': f1, 'tau': tau, 'sens': sens, 'spec': spec, 'auc': auc}
    print(f"  [3_TabPFN]  F1={f1:.4f}  τ={tau:.2f}  Sens={sens:.3f}  Spec={spec:.3f}  AUC={auc:.3f}")
except Exception as e:
    print(f"  TabPFN failed: {e}")
    results['3_TabPFN'] = {'f1': 0, 'note': str(e)}


print("\n" + "="*70)
print("4. Feature selection (top-50 and top-100 by importance)")
print("="*70)

# Get feature importances from a full-data model
_m_imp = lgb.LGBMClassifier(**BEST_PARAMS)
_X_imp = pd.concat([X_aug, X_pah], ignore_index=True)
_y_imp = pd.concat([y_aug, y_pah], ignore_index=True)
_m_imp.fit(_X_imp, _y_imp)
importances = pd.Series(_m_imp.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)

for k in [30, 50, 100]:
    top_cols = list(importances.head(k).index)
    X_pah_k  = X_pah[top_cols]
    X_aug_k  = X_aug[top_cols]

    def feat_sel_preds(tr_idx, va_idx, _xp=X_pah_k, _xa=X_aug_k):
        X_tr = pd.concat([_xa, _xp.iloc[tr_idx]], ignore_index=True)
        y_tr = pd.concat([y_aug, y_pah.iloc[tr_idx]], ignore_index=True)
        m = lgb.LGBMClassifier(**BEST_PARAMS)
        m.fit(X_tr, y_tr, eval_set=[(_xp.iloc[va_idx], y_pah.iloc[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        return m.predict_proba(_xp.iloc[va_idx])[:, 1]

    cv_eval(f'4_top{k}features', feat_sel_preds)


print("\n" + "="*70)
print("5. Aggressive threshold search (fine grid 0.01 to 0.99)")
print("="*70)
# Already done inside cv_eval with 0.01 step — but let's check
# whether an even finer grid or different objective helps
# Try optimising for balanced accuracy instead of macro F1
from sklearn.metrics import balanced_accuracy_score

oof_prob_base = np.zeros(len(y_pah))
for tr_idx, va_idx in CV5.split(X_pah, y_pah):
    oof_prob_base[va_idx] = baseline_preds(tr_idx, va_idx)

# Fine grid
best_tau_fine, best_f1_fine = 0.5, 0.0
for tau in np.arange(0.01, 0.99, 0.005):
    f = f1_score(y_pah.values, (oof_prob_base >= tau).astype(int),
                 average='macro', zero_division=0)
    if f > best_f1_fine:
        best_f1_fine, best_tau_fine = f, tau

pred_fine = (oof_prob_base >= best_tau_fine).astype(int)
sens_fine = ((pred_fine==1)&(y_pah.values==1)).sum()/(y_pah.values==1).sum()
spec_fine = ((pred_fine==0)&(y_pah.values==0)).sum()/(y_pah.values==0).sum()
results['5_fine_threshold'] = {'f1': best_f1_fine, 'tau': best_tau_fine,
                                'sens': sens_fine, 'spec': spec_fine,
                                'auc': roc_auc_score(y_pah.values, oof_prob_base)}
print(f"  [5_fine_threshold]  F1={best_f1_fine:.4f}  τ={best_tau_fine:.3f}  "
      f"Sens={sens_fine:.3f}  Spec={spec_fine:.3f}")


print("\n" + "="*70)
print("6. Cost-sensitive learning (asymmetric class weights)")
print("="*70)

# The default 'balanced' weights penalise both equally.
# For PAH the benign class needs MORE penalty — try heavier weights.
for scale in [2.0, 3.0, 5.0]:
    n_b = (y_pah == 0).sum()
    n_p = (y_pah == 1).sum()
    w = {0: (n_b + n_p) / (2 * n_b) * scale, 1: (n_b + n_p) / (2 * n_p)}

    def cost_preds(tr_idx, va_idx, _w=w):
        X_tr = pd.concat([X_aug, X_pah.iloc[tr_idx]], ignore_index=True)
        y_tr = pd.concat([y_aug, y_pah.iloc[tr_idx]], ignore_index=True)
        params = {**BEST_PARAMS, 'class_weight': _w}
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_pah.iloc[va_idx], y_pah.iloc[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        return m.predict_proba(X_pah.iloc[va_idx])[:, 1]

    cv_eval(f'6_cost_weight_{scale}x', cost_preds)


print("\n" + "="*70)
print("7. EK-only model (9 features — cleanest signal, no noise)")
print("="*70)

X_pah_ek = X_pah[EK_COLS]
X_aug_ek  = X_aug[EK_COLS]

def ek_only_preds(tr_idx, va_idx):
    X_tr = pd.concat([X_aug_ek, X_pah_ek.iloc[tr_idx]], ignore_index=True)
    y_tr = pd.concat([y_aug, y_pah.iloc[tr_idx]], ignore_index=True)
    m = lgb.LGBMClassifier(**BEST_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_pah_ek.iloc[va_idx], y_pah.iloc[va_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return m.predict_proba(X_pah_ek.iloc[va_idx])[:, 1]

cv_eval('7_EK_only', ek_only_preds)


print("\n" + "="*70)
print("8. Best combination: top features + SMOTE + stacking feature")
print("="*70)

# Use top-50 features + master_prob + SMOTE
top50_cols = list(importances.head(50).index)
X_pah_combo = X_pah[top50_cols].copy()
X_pah_combo['master_prob'] = pah_master_prob
X_aug_combo  = X_aug[top50_cols].copy()
X_aug_combo['master_prob'] = aug_master_prob

from imblearn.over_sampling import SMOTE as SMOTE_
_smote_combo = SMOTE_(random_state=SEED, k_neighbors=3)

def combo_preds(tr_idx, va_idx):
    X_tr = pd.concat([X_aug_combo, X_pah_combo.iloc[tr_idx]], ignore_index=True)
    y_tr = pd.concat([y_aug, y_pah.iloc[tr_idx]], ignore_index=True)
    X_tr_filled = X_tr.fillna(X_tr.median())
    try:
        X_res, y_res = _smote_combo.fit_resample(X_tr_filled, y_tr)
        X_res = pd.DataFrame(X_res, columns=X_tr.columns)
    except Exception:
        X_res, y_res = X_tr, y_tr
    m = lgb.LGBMClassifier(**BEST_PARAMS)
    m.fit(X_res, y_res,
          eval_set=[(X_pah_combo.iloc[va_idx], y_pah.iloc[va_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return m.predict_proba(X_pah_combo.iloc[va_idx])[:, 1]

cv_eval('8_combo_top50_smote_stack', combo_preds)


print("\n" + "="*70)
print("9. Logistic regression meta-learner (stacking ensemble)")
print("="*70)

# Get OOF probs from multiple base models, feed into LR meta-learner
def get_oof_probs(X_p, X_a, params):
    oof = np.zeros(len(y_pah))
    for tr_idx, va_idx in CV5.split(X_p, y_pah):
        X_tr = pd.concat([X_a, X_p.iloc[tr_idx]], ignore_index=True)
        y_tr = pd.concat([y_aug, y_pah.iloc[tr_idx]], ignore_index=True)
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr)
        oof[va_idx] = m.predict_proba(X_p.iloc[va_idx])[:, 1]
    return oof

# Base models: full features, top-50 features, EK-only
oof_full  = get_oof_probs(X_pah, X_aug, BEST_PARAMS)
oof_top50 = get_oof_probs(X_pah[top50_cols], X_aug[top50_cols], BEST_PARAMS)
oof_ek    = get_oof_probs(X_pah_ek, X_aug_ek, BEST_PARAMS)

# Meta-features
meta_X = np.column_stack([oof_full, oof_top50, oof_ek, pah_master_prob])
meta_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
meta_oof = np.zeros(len(y_pah))
for tr_idx, va_idx in meta_cv.split(meta_X, y_pah):
    lr = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
    lr.fit(meta_X[tr_idx], y_pah.values[tr_idx])
    meta_oof[va_idx] = lr.predict_proba(meta_X[va_idx])[:, 1]

tau, f1 = optimise_threshold(y_pah.values, meta_oof)
pred = (meta_oof >= tau).astype(int)
sens = ((pred==1)&(y_pah.values==1)).sum()/(y_pah.values==1).sum()
spec = ((pred==0)&(y_pah.values==0)).sum()/(y_pah.values==0).sum()
auc  = roc_auc_score(y_pah.values, meta_oof)
results['9_LR_meta_ensemble'] = {'f1': f1, 'tau': tau, 'sens': sens, 'spec': spec, 'auc': auc}
print(f"  [9_LR_meta_ensemble]  F1={f1:.4f}  τ={tau:.2f}  Sens={sens:.3f}  Spec={spec:.3f}  AUC={auc:.3f}")


# ── RANKED SUMMARY ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("RANKED SUMMARY")
print("="*70)
print(f"{'Rank':<5} {'Approach':<35} {'Macro F1':>9} {'τ':>6} {'Sens':>7} {'Spec':>7} {'AUC':>7}")
print("-"*70)
ranked = sorted([(v['f1'], k, v) for k, v in results.items() if 'f1' in v and v['f1'] > 0],
                reverse=True)
for rank, (f1, name, v) in enumerate(ranked, 1):
    marker = " ← BEST" if rank == 1 else (" ← baseline" if '0_baseline' in name else "")
    print(f"{rank:<5} {name:<35} {f1:>9.4f} {v.get('tau',0):>6.2f} "
          f"{v.get('sens',0):>7.3f} {v.get('spec',0):>7.3f} {v.get('auc',0):>7.3f}{marker}")

print(f"\nBaseline macro F1 (from main pipeline): 0.7480")
best_name, best_val = ranked[0][1], ranked[0][0]
delta = best_val - 0.7480
print(f"Best approach: {best_name}  →  {best_val:.4f}  ({'+' if delta>=0 else ''}{delta:.4f} vs baseline)")
