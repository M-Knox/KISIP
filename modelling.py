# ================================================================
# KISIP Study — Stage 5: Final Matrix Assembly & Modelling
# Models: Ridge Regression, Random Forest, XGBoost + TreeSHAP
# Validation: Leave-One-Settlement-Out Cross Validation (LOSO-CV)
# Target: SCMI (Settlement Change Magnitude Index)
# ================================================================

import os
os.chdir(r'C:\Users\USER\Documents\KISIP')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import Ridge
from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection      import permutation_importance
import xgboost as xgb
import shap

# ================================================================
# SECTION 1: LOAD AND MERGE DATA
# ================================================================
print('Loading data...')
features = pd.read_csv(r'data\kisip_baseline_features_9final.csv')
scmi_df  = pd.read_csv(r'data\kisip_zone_scmi.csv')

# Merge on zone_id
data = pd.merge(
    features,
    scmi_df[['zone_id', 'SCMI']],
    on='zone_id',
    how='inner'
)

print(f'Final dataset shape: {data.shape}')
print(f'Zones per settlement:')
print(data['settlement'].value_counts().to_string())

# ================================================================
# SECTION 2: DEFINE FEATURES AND TARGET
# ================================================================
FEATURE_COLS = [
    'NDVI', 'NDBI', 'MNDWI',
    'Contrast', 'Entropy', 'Homogeneity', 'Correlation',
    'road_density', 'paved_proportion'
]

TARGET = 'SCMI'

X = data[FEATURE_COLS].values
y = data[TARGET].values
settlements = data['settlement'].values

print(f'\nFeatures: {FEATURE_COLS}')
print(f'Target range: {y.min():.4f} — {y.max():.4f}')
print(f'Target mean:  {y.mean():.4f}')

# ================================================================
# SECTION 3: LOSO-CV FUNCTION
# Leave-One-Settlement-Out Cross Validation
# Trains on 4 settlements, tests on the held-out one
# Prevents spatial leakage between adjacent zones
# ================================================================
def loso_cv(model, X, y, settlements, scaler=None, model_name='Model'):
    unique_settlements = np.unique(settlements)
    all_preds   = np.zeros_like(y)
    fold_results = []

    print(f'\n{model_name} — LOSO-CV Results:')
    print('-' * 55)

    for test_settlement in unique_settlements:
        # Train/test split by settlement
        train_mask = settlements != test_settlement
        test_mask  = settlements == test_settlement

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Scale if scaler provided (Ridge only)
        if scaler is not None:
            sc      = scaler()
            X_train = sc.fit_transform(X_train)
            X_test  = sc.transform(X_test)

        # Train and predict
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        all_preds[test_mask] = preds

        # Fold metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)

        fold_results.append({
            'settlement': test_settlement,
            'n_test':     test_mask.sum(),
            'rmse':       rmse,
            'mae':        mae,
            'r2':         r2
        })

        print(f'  Test: {test_settlement:<15} '
              f'n={test_mask.sum():>4}  '
              f'RMSE={rmse:.4f}  '
              f'MAE={mae:.4f}  '
              f'R²={r2:.4f}')

    # Overall metrics across all folds
    overall_rmse = np.sqrt(mean_squared_error(y, all_preds))
    overall_mae  = mean_absolute_error(y, all_preds)
    overall_r2   = r2_score(y, all_preds)

    print('-' * 55)
    print(f'  Overall LOSO          '
          f'RMSE={overall_rmse:.4f}  '
          f'MAE={overall_mae:.4f}  '
          f'R²={overall_r2:.4f}')

    return {
        'model':        model,
        'predictions':  all_preds,
        'fold_results': pd.DataFrame(fold_results),
        'overall': {
            'rmse': overall_rmse,
            'mae':  overall_mae,
            'r2':   overall_r2
        }
    }

# ================================================================
# SECTION 4: MODEL 1 — RIDGE REGRESSION
# Linear baseline — scaled features required
# ================================================================
print('\n' + '=' * 55)
print('MODEL 1: RIDGE REGRESSION')
print('=' * 55)

ridge_model  = Ridge(alpha=1.0)
ridge_results = loso_cv(
    ridge_model, X, y, settlements,
    scaler=StandardScaler,
    model_name='Ridge Regression'
)

# Refit on full data for coefficient extraction
scaler_full   = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)
ridge_model.fit(X_scaled_full, y)

ridge_coefs = pd.DataFrame({
    'feature':     FEATURE_COLS,
    'coefficient': ridge_model.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print('\nRidge Coefficients (full data fit):')
print(ridge_coefs.to_string(index=False))

# ================================================================
# SECTION 5: MODEL 2 — RANDOM FOREST
# Non-linear ensemble — no scaling required
# ================================================================
print('\n' + '=' * 55)
print('MODEL 2: RANDOM FOREST')
print('=' * 55)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_results = loso_cv(
    rf_model, X, y, settlements,
    scaler=None,
    model_name='Random Forest'
)

# Refit on full data for feature importance
rf_model.fit(X, y)
rf_importance = pd.DataFrame({
    'feature':   FEATURE_COLS,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print('\nRandom Forest Feature Importance (full data fit):')
print(rf_importance.to_string(index=False))

# ================================================================
# SECTION 6: MODEL 3 — XGBOOST
# Primary model — TreeSHAP interpretation
# ================================================================
print('\n' + '=' * 55)
print('MODEL 3: XGBOOST')
print('=' * 55)

xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

xgb_results = loso_cv(
    xgb_model, X, y, settlements,
    scaler=None,
    model_name='XGBoost'
)

# Refit on full data for SHAP
xgb_model.fit(X, y)

# ================================================================
# SECTION 7: MODEL COMPARISON TABLE
# ================================================================
print('\n' + '=' * 55)
print('MODEL COMPARISON SUMMARY')
print('=' * 55)

comparison = pd.DataFrame([
    {
        'Model': 'Ridge Regression',
        'RMSE':  ridge_results['overall']['rmse'],
        'MAE':   ridge_results['overall']['mae'],
        'R²':    ridge_results['overall']['r2']
    },
    {
        'Model': 'Random Forest',
        'RMSE':  rf_results['overall']['rmse'],
        'MAE':   rf_results['overall']['mae'],
        'R²':    rf_results['overall']['r2']
    },
    {
        'Model': 'XGBoost',
        'RMSE':  xgb_results['overall']['rmse'],
        'MAE':   xgb_results['overall']['mae'],
        'R²':    xgb_results['overall']['r2']
    }
])

print(comparison.to_string(index=False))
comparison.to_csv('kisip_model_comparison.csv', index=False)

# ================================================================
# SECTION 8: TREESHAP ANALYSIS
# Global and local SHAP values from XGBoost
# ================================================================
print('\n' + '=' * 55)
print('TREESHAP ANALYSIS')
print('=' * 55)

# Compute SHAP values
explainer  = shap.TreeExplainer(xgb_model)
X_df       = pd.DataFrame(X, columns=FEATURE_COLS)
shap_values = explainer.shap_values(X_df)

# Global mean absolute SHAP per feature
shap_importance = pd.DataFrame({
    'feature':          FEATURE_COLS,
    'mean_abs_shap':    np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print('\nGlobal TreeSHAP Feature Importance:')
print(shap_importance.to_string(index=False))
shap_importance.to_csv('kisip_shap_importance.csv', index=False)

# SHAP values per settlement — mean absolute SHAP per feature per settlement
print('\nMean absolute SHAP by settlement:')
shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLS)
shap_df['settlement'] = settlements

shap_by_settlement = shap_df.groupby('settlement')[FEATURE_COLS].apply(
    lambda x: x.abs().mean()
).round(6)
print(shap_by_settlement.to_string())
shap_by_settlement.to_csv('kisip_shap_by_settlement.csv')

# ================================================================
# SECTION 9: VISUALISATIONS
# ================================================================
print('\nGenerating plots...')

# --- Plot 1: LOSO-CV R² comparison across models and settlements ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('LOSO-CV R² per Settlement — Model Comparison', fontsize=13)

for ax, results, name in zip(
    axes,
    [ridge_results, rf_results, xgb_results],
    ['Ridge Regression', 'Random Forest', 'XGBoost']
):
    fold_df = results['fold_results']
    bars    = ax.barh(fold_df['settlement'], fold_df['r2'], color='steelblue')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('R²')
    ax.set_title(name)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)

plt.tight_layout()
plt.savefig('kisip_loso_r2_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: kisip_loso_r2_comparison.png')

# --- Plot 2: SHAP Summary Bar Plot ---
plt.figure(figsize=(9, 6))
shap.summary_plot(
    shap_values, X_df,
    plot_type='bar',
    show=False,
    color='steelblue'
)
plt.title('Global TreeSHAP Feature Importance — XGBoost', fontsize=12)
plt.tight_layout()
plt.savefig('kisip_shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: kisip_shap_bar.png')

# --- Plot 3: SHAP Beeswarm Plot ---
plt.figure(figsize=(9, 6))
shap.summary_plot(
    shap_values, X_df,
    plot_type='dot',
    show=False
)
plt.title('TreeSHAP Beeswarm — Feature Impact on SCMI', fontsize=12)
plt.tight_layout()
plt.savefig('kisip_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: kisip_shap_beeswarm.png')

# --- Plot 4: Predicted vs Actual SCMI — XGBoost ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Predicted vs Actual SCMI — LOSO-CV', fontsize=13)

colours = {
    'Mathare':       '#e41a1c',
    'Kayole_Soweto': '#377eb8',
    'Kahawa_Soweto': '#4daf4a',
    'KCC':           '#984ea3',
    'Kambi_Moto':    '#ff7f00'
}

for ax, results, name in zip(
    axes,
    [ridge_results, rf_results, xgb_results],
    ['Ridge Regression', 'Random Forest', 'XGBoost']
):
    for settlement in np.unique(settlements):
        mask = settlements == settlement
        ax.scatter(
            y[mask],
            results['predictions'][mask],
            label=settlement,
            alpha=0.4,
            s=8,
            color=colours.get(settlement, 'grey')
        )

    lims = [y.min(), y.max()]
    ax.plot(lims, lims, 'k--', linewidth=1, label='Perfect fit')
    ax.set_xlabel('Actual SCMI')
    ax.set_ylabel('Predicted SCMI')
    ax.set_title(f'{name}\nR²={results["overall"]["r2"]:.3f}  '
                 f'RMSE={results["overall"]["rmse"]:.4f}')
    if ax == axes[0]:
        ax.legend(fontsize=7, markerscale=2)

plt.tight_layout()
plt.savefig('kisip_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: kisip_predicted_vs_actual.png')

# --- Plot 5: Ridge Coefficients ---
plt.figure(figsize=(8, 5))
colours_coef = ['steelblue' if c > 0 else 'tomato'
                 for c in ridge_coefs['coefficient']]
plt.barh(ridge_coefs['feature'], ridge_coefs['coefficient'], color=colours_coef)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('Coefficient Value')
plt.title('Ridge Regression Coefficients\n(Blue = positive, Red = negative)', fontsize=11)
plt.tight_layout()
plt.savefig('kisip_ridge_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: kisip_ridge_coefficients.png')

# ================================================================
# SECTION 10: EXPORT ALL PREDICTIONS
# ================================================================
data['ridge_pred'] = ridge_results['predictions']
data['rf_pred']    = rf_results['predictions']
data['xgb_pred']   = xgb_results['predictions']

data[[
    'zone_id', 'settlement', 'SCMI',
    'ridge_pred', 'rf_pred', 'xgb_pred'
] + FEATURE_COLS].to_csv('kisip_model_predictions.csv', index=False)

print('\nExported: kisip_model_predictions.csv')

# ================================================================
# SECTION 11: SHAP VALUES EXPORT
# For spatial mapping in GEE or external GIS
# ================================================================
shap_export = pd.DataFrame(shap_values, columns=[f'shap_{f}' for f in FEATURE_COLS])
shap_export['zone_id']    = data['zone_id'].values
shap_export['settlement'] = data['settlement'].values
shap_export['SCMI']       = y
shap_export['xgb_pred']   = xgb_results['predictions']

shap_export.to_csv('kisip_shap_values.csv', index=False)
print('Exported: kisip_shap_values.csv')

print('\n' + '=' * 55)
print('STAGE 5 COMPLETE')
print('=' * 55)
print('Output files:')
print('  kisip_model_comparison.csv')
print('  kisip_model_predictions.csv')
print('  kisip_shap_importance.csv')
print('  kisip_shap_by_settlement.csv')
print('  kisip_shap_values.csv')
print('  kisip_loso_r2_comparison.png')
print('  kisip_shap_bar.png')
print('  kisip_shap_beeswarm.png')
print('  kisip_predicted_vs_actual.png')
print('  kisip_ridge_coefficients.png')