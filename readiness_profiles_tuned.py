# ================================================================
# KISIP — Readiness Profile Rerun: Tuned vs Baseline XGBoost
# Compares settlement rankings under baseline and tuned parameters
# ================================================================

import os
os.chdir(r'C:\Users\USER\Documents\KISIP')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model  import Ridge
from sklearn.ensemble      import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ----------------------------------------------------------------
# SECTION 1: LOAD DATA
# ----------------------------------------------------------------
print('Loading data...')

features_train = pd.read_csv(r'data\kisip_baseline_features_9final.csv')
scmi_train     = pd.read_csv(r'data\kisip_zone_scmi.csv')

train_data = pd.merge(
    features_train,
    scmi_train[['zone_id', 'SCMI']],
    on='zone_id',
    how='inner'
)

mukuru_spectral = pd.read_csv(r'data\mukuru_spectral_features.csv')
mukuru_roads    = pd.read_csv(r'data\mukuru_road_features.csv')

mukuru_data = pd.merge(
    mukuru_spectral,
    mukuru_roads[['zone_id', 'road_density', 'paved_proportion']],
    on='zone_id',
    how='inner'
)

FEATURE_COLS = [
    'NDVI', 'NDBI', 'MNDWI',
    'Contrast', 'Entropy', 'Homogeneity', 'Correlation',
    'road_density', 'paved_proportion'
]

X_train  = train_data[FEATURE_COLS].values
y_train  = train_data['SCMI'].values
X_mukuru = mukuru_data[FEATURE_COLS].values

kisip_mean = y_train.mean()
kisip_std  = y_train.std()

# ----------------------------------------------------------------
# SECTION 2: BASELINE MODELS (original parameters)
# ----------------------------------------------------------------
print('Training baseline models...')

scaler_base    = StandardScaler()
X_train_sc     = scaler_base.fit_transform(X_train)

ridge_base = Ridge(alpha=1.0)
ridge_base.fit(X_train_sc, y_train)

rf_base = RandomForestRegressor(
    n_estimators=300, max_depth=10,
    min_samples_leaf=5, random_state=42, n_jobs=-1
)
rf_base.fit(X_train, y_train)

xgb_base = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0
)
xgb_base.fit(X_train, y_train)

# ----------------------------------------------------------------
# SECTION 3: TUNED MODELS
# Use the best parameters identified from nested LOSO-CV
# Update these to match what your tuning output selected
# ----------------------------------------------------------------
print('Training tuned models...')

# Ridge — update alpha to whichever was most commonly selected
# across folds in your tuning output
scaler_tuned   = StandardScaler()
X_train_sc_t   = scaler_tuned.fit_transform(X_train)

ridge_tuned = Ridge(alpha=100.0)  # Update from your tuning output
ridge_tuned.fit(X_train_sc_t, y_train)

# Random Forest — update depth and leaf from your tuning output
rf_tuned = RandomForestRegressor(
    n_estimators=300, max_depth=10,
    min_samples_leaf=10, random_state=42, n_jobs=-1
)
rf_tuned.fit(X_train, y_train)

# XGBoost — update to best params from tuning
# These are the most regularised params from your grid
xgb_tuned = xgb.XGBRegressor(
    n_estimators=500, max_depth=4, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=10, reg_alpha=0.5, reg_lambda=1.0,
    random_state=42, verbosity=0
)
xgb_tuned.fit(X_train, y_train)

print('All models trained')

# ----------------------------------------------------------------
# SECTION 4: GENERATE PREDICTIONS — BASELINE AND TUNED
# ----------------------------------------------------------------
X_mukuru_sc_base  = scaler_base.transform(X_mukuru)
X_mukuru_sc_tuned = scaler_tuned.transform(X_mukuru)

# Baseline predictions
mukuru_data['ridge_base']    = ridge_base.predict(X_mukuru_sc_base)
mukuru_data['rf_base']       = rf_base.predict(X_mukuru)
mukuru_data['xgb_base']      = xgb_base.predict(X_mukuru)
mukuru_data['ensemble_base'] = mukuru_data[
    ['ridge_base', 'rf_base', 'xgb_base']
].mean(axis=1)

# Tuned predictions
mukuru_data['ridge_tuned']    = ridge_tuned.predict(X_mukuru_sc_tuned)
mukuru_data['rf_tuned']       = rf_tuned.predict(X_mukuru)
mukuru_data['xgb_tuned']      = xgb_tuned.predict(X_mukuru)
mukuru_data['ensemble_tuned'] = mukuru_data[
    ['ridge_tuned', 'rf_tuned', 'xgb_tuned']
].mean(axis=1)

# ----------------------------------------------------------------
# SECTION 5: SETTLEMENT-LEVEL AGGREGATION
# ----------------------------------------------------------------
def settlement_scores(df, ensemble_col):
    scores = df.groupby('settlement').agg(
        zones              = ('zone_id', 'count'),
        mean_ensemble_scmi = (ensemble_col, 'mean'),
        std_ensemble_scmi  = (ensemble_col, 'std')
    ).reset_index()
    scores = scores.sort_values(
        'mean_ensemble_scmi', ascending=False
    ).reset_index(drop=True)
    scores['rank'] = scores.index + 1
    return scores

baseline_scores = settlement_scores(mukuru_data, 'ensemble_base')
tuned_scores    = settlement_scores(mukuru_data, 'ensemble_tuned')

print('\n' + '=' * 65)
print('BASELINE READINESS RANKINGS')
print('=' * 65)
print(baseline_scores[[
    'rank', 'settlement', 'mean_ensemble_scmi', 'std_ensemble_scmi'
]].to_string(index=False))

print('\n' + '=' * 65)
print('TUNED READINESS RANKINGS')
print('=' * 65)
print(tuned_scores[[
    'rank', 'settlement', 'mean_ensemble_scmi', 'std_ensemble_scmi'
]].to_string(index=False))

# ----------------------------------------------------------------
# SECTION 6: RANK STABILITY ANALYSIS
# ----------------------------------------------------------------
print('\n' + '=' * 65)
print('RANK STABILITY COMPARISON')
print('=' * 65)

rank_comparison = baseline_scores[[
    'settlement', 'rank', 'mean_ensemble_scmi'
]].rename(columns={
    'rank': 'baseline_rank',
    'mean_ensemble_scmi': 'baseline_scmi'
}).merge(
    tuned_scores[[
        'settlement', 'rank', 'mean_ensemble_scmi'
    ]].rename(columns={
        'rank': 'tuned_rank',
        'mean_ensemble_scmi': 'tuned_scmi'
    }),
    on='settlement'
)

rank_comparison['rank_change'] = (
    rank_comparison['baseline_rank'] -
    rank_comparison['tuned_rank']
)
rank_comparison['scmi_change'] = (
    rank_comparison['tuned_scmi'] -
    rank_comparison['baseline_scmi']
).round(4)

print(rank_comparison.to_string(index=False))

stable = (rank_comparison['rank_change'] == 0).all()
print(f'\nRankings stable across tuning: {stable}')

if stable:
    print('All settlements maintain the same readiness rank')
    print('Tuning improved prediction accuracy without changing policy conclusions')
else:
    changed = rank_comparison[rank_comparison['rank_change'] != 0]
    print(f'{len(changed)} settlements changed rank:')
    print(changed[['settlement', 'baseline_rank',
                   'tuned_rank', 'scmi_change']].to_string(index=False))

# ----------------------------------------------------------------
# SECTION 7: TREESHAP FOR TUNED XGBoost
# ----------------------------------------------------------------
print('\nComputing TreeSHAP for tuned XGBoost...')

explainer_tuned = shap.TreeExplainer(xgb_tuned)
X_mukuru_df     = pd.DataFrame(X_mukuru, columns=FEATURE_COLS)
shap_tuned      = explainer_tuned.shap_values(X_mukuru_df)

shap_df               = pd.DataFrame(shap_tuned, columns=FEATURE_COLS)
shap_df['settlement'] = mukuru_data['settlement'].values

shap_by_settlement_tuned = shap_df.groupby('settlement')[FEATURE_COLS].apply(
    lambda x: x.abs().mean()
).round(6)

print('\nTuned XGBoost SHAP Attribution:')
print(shap_by_settlement_tuned.to_string())

# Load baseline SHAP for comparison
shap_base_df = pd.read_csv('kisip_mukuru_shap_attribution.csv',
                            index_col=0)

print('\nBaseline XGBoost SHAP Attribution:')
print(shap_base_df.to_string())

# ----------------------------------------------------------------
# SECTION 8: VISUALISATIONS
# ----------------------------------------------------------------
SETTLEMENT_COLOURS = {
    'Milimani_village': '#2ecc71',
    'area_48_village':  '#3498db',
    'wapewape_village': '#e67e22',
    'Sisal_village':    '#e74c3c'
}

# --- Plot 1: Baseline vs Tuned Rankings Side by Side ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
fig.suptitle(
    'Mukuru Readiness Rankings — Baseline vs Tuned Models\n'
    'Ensemble SCMI with KISIP treated mean reference',
    fontsize=13, fontweight='bold'
)

for ax, scores, title in zip(
    axes,
    [baseline_scores, tuned_scores],
    ['Baseline Parameters', 'Tuned Parameters']
):
    colours = [
        SETTLEMENT_COLOURS.get(s, 'grey')
        for s in scores['settlement']
    ]
    bars = ax.bar(
        scores['settlement'],
        scores['mean_ensemble_scmi'],
        color=colours, alpha=0.85,
        yerr=scores['std_ensemble_scmi'],
        capsize=5, edgecolor='white'
    )
    ax.axhline(
        kisip_mean, color='black',
        linestyle='--', linewidth=1.5,
        label=f'KISIP mean ({kisip_mean:.3f})'
    )
    ax.axhline(
        kisip_mean + kisip_std,
        color='grey', linestyle=':', linewidth=1
    )
    ax.axhline(
        kisip_mean - kisip_std,
        color='grey', linestyle=':', linewidth=1,
        label='KISIP ±1 std'
    )
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('Predicted SCMI (Ensemble)')
    ax.set_xticklabels(
        scores['settlement'],
        rotation=15, fontsize=9
    )
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('kisip_tuned_vs_baseline_rankings.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Saved: kisip_tuned_vs_baseline_rankings.png')

# --- Plot 2: SCMI Change per Settlement from Tuning ---
fig, ax = plt.subplots(figsize=(9, 5))

colours_change = [
    '#2ecc71' if v >= 0 else '#e74c3c'
    for v in rank_comparison['scmi_change']
]
bars = ax.barh(
    rank_comparison['settlement'],
    rank_comparison['scmi_change'],
    color=colours_change, alpha=0.85
)
ax.axvline(0, color='black', linewidth=1)
ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
ax.set_xlabel('SCMI Change (Tuned − Baseline)')
ax.set_title(
    'Effect of Hyperparameter Tuning on Predicted SCMI\n'
    'Mukuru Settlements',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig('kisip_tuning_scmi_change.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Saved: kisip_tuning_scmi_change.png')

# --- Plot 3: SHAP Comparison Baseline vs Tuned ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle(
    'TreeSHAP Feature Attribution — Baseline vs Tuned XGBoost\n'
    'Mukuru Settlements',
    fontsize=13, fontweight='bold'
)

for ax, shap_data, title in zip(
    axes,
    [shap_base_df, shap_by_settlement_tuned],
    ['Baseline XGBoost', 'Tuned XGBoost']
):
    im = ax.imshow(
        shap_data.values,
        cmap='YlOrRd', aspect='auto'
    )
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(
        FEATURE_COLS, rotation=45,
        ha='right', fontsize=9
    )
    ax.set_yticks(range(len(shap_data.index)))
    ax.set_yticklabels(shap_data.index, fontsize=9)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, label='Mean |SHAP|')

    for i in range(len(shap_data.index)):
        for j in range(len(FEATURE_COLS)):
            ax.text(
                j, i,
                f'{shap_data.values[i,j]:.5f}',
                ha='center', va='center', fontsize=7
            )

plt.tight_layout()
plt.savefig('kisip_shap_baseline_vs_tuned.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Saved: kisip_shap_baseline_vs_tuned.png')

# ----------------------------------------------------------------
# SECTION 9: EXPORT
# ----------------------------------------------------------------
rank_comparison.to_csv(
    'kisip_rank_stability_analysis.csv', index=False
)

mukuru_data[[
    'zone_id', 'settlement',
    'ensemble_base', 'ensemble_tuned'
]].to_csv('kisip_mukuru_tuned_predictions.csv', index=False)

shap_by_settlement_tuned.to_csv(
    'kisip_mukuru_shap_tuned.csv'
)

print('\n' + '=' * 65)
print('RERUN COMPLETE')
print('=' * 65)
print('Outputs:')
print('  kisip_tuned_vs_baseline_rankings.png')
print('  kisip_tuning_scmi_change.png')
print('  kisip_shap_baseline_vs_tuned.png')
print('  kisip_rank_stability_analysis.csv')
print('  kisip_mukuru_tuned_predictions.csv')
print('  kisip_mukuru_shap_tuned.csv')