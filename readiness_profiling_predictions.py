# ================================================================
# KISIP — Stage 6C: Readiness Profiling
# Applies trained models to Mukuru untreated settlements
# Generates SCMI predictions and SHAP readiness profiles
# ================================================================

import os
os.chdir(r'C:\Users\USER\Documents\KISIP')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model  import Ridge
from sklearn.ensemble      import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# ================================================================
# SECTION 1: LOAD KISIP TRAINING DATA
# ================================================================
print('Loading KISIP training data...')

features_train = pd.read_csv(r'data\kisip_baseline_features_9final.csv')
scmi_train     = pd.read_csv(r'data\kisip_zone_scmi.csv')

train_data = pd.merge(
    features_train,
    scmi_train[['zone_id', 'SCMI']],
    on='zone_id',
    how='inner'
)

FEATURE_COLS = [
    'NDVI', 'NDBI', 'MNDWI',
    'Contrast', 'Entropy', 'Homogeneity', 'Correlation',
    'road_density', 'paved_proportion'
]

X_train = train_data[FEATURE_COLS].values
y_train = train_data['SCMI'].values

print(f'Training zones:       {len(train_data)}')
print(f'Training settlements: {train_data["settlement"].unique()}')
print(f'SCMI range:           {y_train.min():.4f} — {y_train.max():.4f}')
print(f'SCMI mean:            {y_train.mean():.4f}')

# ================================================================
# SECTION 2: RETRAIN ALL THREE MODELS ON FULL KISIP DATASET
# ================================================================
print('\nTraining models on full KISIP dataset...')

# Ridge — requires scaling
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
ridge      = Ridge(alpha=1.0)
ridge.fit(X_train_sc, y_train)
print('Ridge trained')

# Random Forest
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print('Random Forest trained')

# XGBoost — primary model for SHAP
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
print('XGBoost trained')

# ================================================================
# SECTION 3: LOAD AND MERGE MUKURU FEATURES
# ================================================================
print('\nLoading Mukuru features...')

mukuru_spectral = pd.read_csv(r'data\mukuru_spectral_features.csv')
mukuru_roads    = pd.read_csv(r'data\mukuru_road_features.csv')

print(f'Spectral features shape: {mukuru_spectral.shape}')
print(f'Road features shape:     {mukuru_roads.shape}')

# Verify zone_id alignment
print(f'\nSpectral zone_id sample: {mukuru_spectral["zone_id"].head(3).tolist()}')
print(f'Road zone_id sample:     {mukuru_roads["zone_id"].head(3).tolist()}')

mukuru_data = pd.merge(
    mukuru_spectral,
    mukuru_roads[['zone_id', 'road_density', 'paved_proportion']],
    on='zone_id',
    how='inner'
)

print(f'\nMerged Mukuru zones: {len(mukuru_data)}')
print(f'Settlements: {mukuru_data["settlement"].unique()}')

# Handle any nulls
for col in FEATURE_COLS:
    null_count = mukuru_data[col].isnull().sum()
    if null_count > 0:
        mukuru_data[col] = mukuru_data[col].fillna(
            mukuru_data.groupby('settlement')[col].transform('mean')
        )
        print(f'Filled {null_count} nulls in {col}')

# Verify all features present
missing = [f for f in FEATURE_COLS if f not in mukuru_data.columns]
if missing:
    print(f'ERROR: Missing features: {missing}')
else:
    print('All 9 features confirmed present')

X_mukuru = mukuru_data[FEATURE_COLS].values

# ================================================================
# SECTION 4: GENERATE READINESS PREDICTIONS
# ================================================================
print('\nGenerating readiness predictions...')

X_mukuru_sc = scaler.transform(X_mukuru)

mukuru_data['ridge_scmi']    = ridge.predict(X_mukuru_sc)
mukuru_data['rf_scmi']       = rf.predict(X_mukuru)
mukuru_data['xgb_scmi']      = xgb_model.predict(X_mukuru)
mukuru_data['ensemble_scmi'] = mukuru_data[
    ['ridge_scmi', 'rf_scmi', 'xgb_scmi']
].mean(axis=1)

print('Predictions generated for all 315 zones')

# ================================================================
# SECTION 5: SETTLEMENT-LEVEL READINESS SCORES
# ================================================================
settlement_scores = mukuru_data.groupby('settlement').agg(
    zones              = ('zone_id',        'count'),
    mean_ridge_scmi    = ('ridge_scmi',     'mean'),
    mean_rf_scmi       = ('rf_scmi',        'mean'),
    mean_xgb_scmi      = ('xgb_scmi',       'mean'),
    mean_ensemble_scmi = ('ensemble_scmi',  'mean'),
    std_ensemble_scmi  = ('ensemble_scmi',  'std'),
    min_ensemble_scmi  = ('ensemble_scmi',  'min'),
    max_ensemble_scmi  = ('ensemble_scmi',  'max'),
    mean_NDBI          = ('NDBI',           'mean'),
    mean_NDVI          = ('NDVI',           'mean'),
    mean_road_density  = ('road_density',   'mean'),
    mean_paved_prop    = ('paved_proportion','mean')
).reset_index()

# Rank by ensemble SCMI — higher predicted change = higher readiness
settlement_scores = settlement_scores.sort_values(
    'mean_ensemble_scmi', ascending=False
).reset_index(drop=True)
settlement_scores['readiness_rank'] = settlement_scores.index + 1

# Contextualise against KISIP treated settlement SCMI
kisip_settlement_means = train_data.groupby('settlement')['SCMI'].mean()
kisip_overall_mean     = y_train.mean()
kisip_overall_std      = y_train.std()

print('\n' + '=' * 65)
print('MUKURU SETTLEMENT READINESS PROFILES')
print('=' * 65)
print(settlement_scores[[
    'readiness_rank', 'settlement', 'zones',
    'mean_ensemble_scmi', 'std_ensemble_scmi',
    'mean_ridge_scmi', 'mean_rf_scmi', 'mean_xgb_scmi'
]].to_string(index=False))

print(f'\nKISIP treated settlements — mean SCMI: {kisip_overall_mean:.4f}')
print(f'KISIP treated settlements — std SCMI:  {kisip_overall_std:.4f}')
print(f'\nKISIP settlement-level means:')
print(kisip_settlement_means.sort_values(ascending=False).round(4).to_string())

# ================================================================
# SECTION 6: TREESHAP READINESS DECOMPOSITION
# ================================================================
print('\nComputing TreeSHAP for Mukuru zones...')

explainer        = shap.TreeExplainer(xgb_model)
X_mukuru_df      = pd.DataFrame(X_mukuru, columns=FEATURE_COLS)
shap_values      = explainer.shap_values(X_mukuru_df)

shap_df               = pd.DataFrame(shap_values, columns=FEATURE_COLS)
shap_df['settlement'] = mukuru_data['settlement'].values
shap_df['zone_id']    = mukuru_data['zone_id'].values

# Mean absolute SHAP per feature per settlement
shap_by_settlement = shap_df.groupby('settlement')[FEATURE_COLS].apply(
    lambda x: x.abs().mean()
).round(6)

print('\nTreeSHAP Feature Attribution by Settlement:')
print(shap_by_settlement.to_string())

# ================================================================
# SECTION 7: DETAILED READINESS PROFILE PER SETTLEMENT
# ================================================================
print('\n' + '=' * 65)
print('DETAILED READINESS PROFILES WITH SHAP DRIVERS')
print('=' * 65)

for _, row in settlement_scores.iterrows():
    settlement = row['settlement']

    # Readiness tier relative to KISIP mean
    ensemble_scmi = row['mean_ensemble_scmi']
    if ensemble_scmi >= kisip_overall_mean + kisip_overall_std:
        tier = 'HIGH — above KISIP treated mean + 1 std'
    elif ensemble_scmi >= kisip_overall_mean:
        tier = 'MODERATE — at or above KISIP treated mean'
    elif ensemble_scmi >= kisip_overall_mean - kisip_overall_std:
        tier = 'LOW — below KISIP treated mean'
    else:
        tier = 'VERY LOW — more than 1 std below KISIP treated mean'

    print(f'\n{settlement} (Readiness Rank: {int(row["readiness_rank"])})')
    print(f'  Readiness tier:       {tier}')
    print(f'  Ensemble SCMI:        {ensemble_scmi:.4f} ± {row["std_ensemble_scmi"]:.4f}')
    print(f'  Ridge prediction:     {row["mean_ridge_scmi"]:.4f}')
    print(f'  Random Forest pred:   {row["mean_rf_scmi"]:.4f}')
    print(f'  XGBoost prediction:   {row["mean_xgb_scmi"]:.4f}')
    print(f'  Zones assessed:       {int(row["zones"])}')
    print(f'  Mean NDBI:            {row["mean_NDBI"]:.4f}')
    print(f'  Mean NDVI:            {row["mean_NDVI"]:.4f}')
    print(f'  Mean road density:    {row["mean_road_density"]:.1f} m/km²')
    print(f'  KISIP overall mean:   {kisip_overall_mean:.4f}')
    print(f'  Difference from mean: {ensemble_scmi - kisip_overall_mean:+.4f}')

    if settlement in shap_by_settlement.index:
        top_drivers = shap_by_settlement.loc[settlement].sort_values(
            ascending=False
        ).head(3)
        print(f'  Top 3 SHAP drivers:')
        for feat, val in top_drivers.items():
            print(f'    {feat:<20} {val:.6f}')

# ================================================================
# SECTION 8: VISUALISATIONS
# ================================================================
print('\nGenerating plots...')

SETTLEMENT_COLOURS = {
    'Milimani_village': '#2ecc71',
    'area_48_village':  '#3498db',
    'wapewape_village': '#e67e22',
    'Sisal_village':    '#e74c3c'
}

# --- Plot 1: Settlement Readiness Bar Chart with KISIP Reference ---
fig, ax = plt.subplots(figsize=(10, 6))

colours = [SETTLEMENT_COLOURS.get(s, 'grey')
           for s in settlement_scores['settlement']]

bars = ax.bar(
    settlement_scores['settlement'],
    settlement_scores['mean_ensemble_scmi'],
    color=colours,
    alpha=0.85,
    yerr=settlement_scores['std_ensemble_scmi'],
    capsize=6,
    edgecolor='white',
    linewidth=0.5
)

# KISIP reference lines
ax.axhline(
    kisip_overall_mean,
    color='black', linestyle='--', linewidth=1.5,
    label=f'KISIP treated mean ({kisip_overall_mean:.3f})'
)
ax.axhline(
    kisip_overall_mean + kisip_overall_std,
    color='grey', linestyle=':', linewidth=1,
    label=f'KISIP mean ± 1 std'
)
ax.axhline(
    kisip_overall_mean - kisip_overall_std,
    color='grey', linestyle=':', linewidth=1
)

ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=9)
ax.set_xlabel('Mukuru Settlement', fontsize=11)
ax.set_ylabel('Predicted SCMI (Ensemble)', fontsize=11)
ax.set_title(
    'Mukuru Settlement Readiness Profiles\n'
    'Predicted SCMI relative to KISIP treated settlement baseline',
    fontsize=12
)
ax.legend(fontsize=9)
ax.set_xticklabels(
    settlement_scores['settlement'],
    rotation=15, fontsize=9
)
plt.tight_layout()
plt.savefig('kisip_mukuru_readiness_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: kisip_mukuru_readiness_bar.png')

# --- Plot 2: Three Model Predictions per Settlement ---
fig, ax = plt.subplots(figsize=(10, 6))

x      = np.arange(len(settlement_scores))
width  = 0.25

ax.bar(x - width, settlement_scores['mean_ridge_scmi'],
       width, label='Ridge', color='steelblue', alpha=0.85)
ax.bar(x,         settlement_scores['mean_rf_scmi'],
       width, label='Random Forest', color='seagreen', alpha=0.85)
ax.bar(x + width, settlement_scores['mean_xgb_scmi'],
       width, label='XGBoost', color='tomato', alpha=0.85)

ax.axhline(
    kisip_overall_mean,
    color='black', linestyle='--', linewidth=1.5,
    label=f'KISIP treated mean ({kisip_overall_mean:.3f})'
)

ax.set_xticks(x)
ax.set_xticklabels(settlement_scores['settlement'], rotation=15, fontsize=9)
ax.set_ylabel('Predicted SCMI', fontsize=11)
ax.set_title(
    'Readiness Predictions per Model — Mukuru Settlements',
    fontsize=12
)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('kisip_mukuru_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: kisip_mukuru_model_comparison.png')

# --- Plot 3: SHAP Feature Attribution Heatmap ---
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(
    shap_by_settlement.values,
    cmap='YlOrRd',
    aspect='auto'
)
ax.set_xticks(range(len(FEATURE_COLS)))
ax.set_xticklabels(FEATURE_COLS, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(shap_by_settlement.index)))
ax.set_yticklabels(shap_by_settlement.index, fontsize=9)
ax.set_title(
    'TreeSHAP Feature Attribution — Mukuru Readiness Profiles\n'
    '(Mean |SHAP| per feature per settlement)',
    fontsize=11
)
plt.colorbar(im, ax=ax, label='Mean |SHAP|')

for i in range(len(shap_by_settlement.index)):
    for j in range(len(FEATURE_COLS)):
        ax.text(
            j, i,
            f'{shap_by_settlement.values[i,j]:.5f}',
            ha='center', va='center', fontsize=7
        )

plt.tight_layout()
plt.savefig('kisip_mukuru_shap_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: kisip_mukuru_shap_heatmap.png')

# --- Plot 4: Zone-level SCMI Distribution per Settlement ---
n_settlements = len(settlement_scores)
fig, axes     = plt.subplots(1, n_settlements, figsize=(14, 4), sharey=True)
fig.suptitle(
    'Zone-level Predicted SCMI Distribution — Mukuru Settlements',
    fontsize=12
)

for ax, (_, row) in zip(axes, settlement_scores.iterrows()):
    settlement = row['settlement']
    zone_scmi  = mukuru_data.loc[
        mukuru_data['settlement'] == settlement, 'ensemble_scmi'
    ]
    colour = SETTLEMENT_COLOURS.get(settlement, 'steelblue')

    ax.hist(zone_scmi, bins=15, color=colour, alpha=0.8, edgecolor='white')
    ax.axvline(
        zone_scmi.mean(), color='black',
        linestyle='--', linewidth=1.2,
        label=f'Mean: {zone_scmi.mean():.3f}'
    )
    ax.axvline(
        kisip_overall_mean, color='red',
        linestyle=':', linewidth=1,
        label=f'KISIP mean: {kisip_overall_mean:.3f}'
    )
    ax.set_title(
        f'{settlement}\nRank {int(row["readiness_rank"])}',
        fontsize=9
    )
    ax.set_xlabel('Predicted SCMI', fontsize=8)
    if ax == axes[0]:
        ax.set_ylabel('Zone count')
    ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig('kisip_mukuru_zone_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: kisip_mukuru_zone_distribution.png')

# --- Plot 5: SHAP Beeswarm for Mukuru ---
plt.figure(figsize=(9, 6))
shap.summary_plot(
    shap_values, X_mukuru_df,
    plot_type='dot',
    show=False
)
plt.title(
    'TreeSHAP Beeswarm — Mukuru Readiness Feature Impact',
    fontsize=12
)
plt.tight_layout()
plt.savefig('kisip_mukuru_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: kisip_mukuru_shap_beeswarm.png')

# ================================================================
# SECTION 9: EXPORT ALL RESULTS
# ================================================================
mukuru_data[[
    'zone_id', 'settlement',
    'ridge_scmi', 'rf_scmi', 'xgb_scmi', 'ensemble_scmi'
] + FEATURE_COLS].to_csv('kisip_mukuru_predictions.csv', index=False)

settlement_scores.to_csv(
    'kisip_mukuru_readiness_profiles.csv', index=False
)

shap_by_settlement.to_csv('kisip_mukuru_shap_attribution.csv')

shap_export = pd.DataFrame(
    shap_values,
    columns=[f'shap_{f}' for f in FEATURE_COLS]
)
shap_export['zone_id']    = mukuru_data['zone_id'].values
shap_export['settlement'] = mukuru_data['settlement'].values
shap_export['ensemble_scmi'] = mukuru_data['ensemble_scmi'].values
shap_export.to_csv('kisip_mukuru_shap_values.csv', index=False)

print('\n' + '=' * 65)
print('STAGE 6C COMPLETE')
print('=' * 65)
print('Output files:')
print('  kisip_mukuru_predictions.csv')
print('  kisip_mukuru_readiness_profiles.csv')
print('  kisip_mukuru_shap_attribution.csv')
print('  kisip_mukuru_shap_values.csv')
print('  kisip_mukuru_readiness_bar.png')
print('  kisip_mukuru_model_comparison.png')
print('  kisip_mukuru_shap_heatmap.png')
print('  kisip_mukuru_zone_distribution.png')
print('  kisip_mukuru_shap_beeswarm.png')