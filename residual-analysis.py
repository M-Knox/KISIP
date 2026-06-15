# ================================================================
# KISIP — Residual Analysis
# Analyses prediction errors from LOSO-CV across all three models
# Required for Slide 11: Discussion & Error Analysis
# ================================================================

import os
os.chdir(r'C:\Users\USER\Documents\KISIP')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model  import Ridge
from sklearn.ensemble      import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# ================================================================
# SECTION 1: LOAD DATA AND RETRAIN MODELS WITH LOSO-CV
# Collect predictions and residuals per fold
# ================================================================
print('Loading data...')

features = pd.read_csv(r'data\kisip_baseline_features_9final.csv')
scmi_df  = pd.read_csv(r'data\kisip_zone_scmi.csv')

data = pd.merge(
    features,
    scmi_df[['zone_id', 'SCMI']],
    on='zone_id',
    how='inner'
)

FEATURE_COLS = [
    'NDVI', 'NDBI', 'MNDWI',
    'Contrast', 'Entropy', 'Homogeneity', 'Correlation',
    'road_density', 'paved_proportion'
]

X           = data[FEATURE_COLS].values
y           = data['SCMI'].values
settlements = data['settlement'].values

print(f'Dataset: {len(data)} zones')
print(f'SCMI range: {y.min():.4f} — {y.max():.4f}')

# ================================================================
# SECTION 2: LOSO-CV WITH RESIDUAL COLLECTION
# ================================================================
def loso_cv_residuals(model, X, y, settlements,
                      scaler=None, model_name='Model'):
    unique_settlements = np.unique(settlements)
    all_preds          = np.zeros_like(y, dtype=float)
    fold_results       = []

    for test_settlement in unique_settlements:
        train_mask = settlements != test_settlement
        test_mask  = settlements == test_settlement

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if scaler is not None:
            sc      = scaler()
            X_train = sc.fit_transform(X_train)
            X_test  = sc.transform(X_test)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        all_preds[test_mask] = preds

        residuals  = y_test - preds
        fold_results.append({
            'settlement':    test_settlement,
            'n_test':        int(test_mask.sum()),
            'rmse':          float(np.sqrt(mean_squared_error(y_test, preds))),
            'mae':           float(mean_absolute_error(y_test, preds)),
            'r2':            float(r2_score(y_test, preds)),
            'mean_residual': float(residuals.mean()),
            'std_residual':  float(residuals.std()),
            'max_error':     float(np.abs(residuals).max()),
            'pct_within_005': float((np.abs(residuals) <= 0.05).mean() * 100),
            'pct_within_010': float((np.abs(residuals) <= 0.10).mean() * 100)
        })

    residuals_all = y - all_preds

    return {
        'model':        model,
        'predictions':  all_preds,
        'residuals':    residuals_all,
        'fold_results': pd.DataFrame(fold_results),
        'overall': {
            'rmse': float(np.sqrt(mean_squared_error(y, all_preds))),
            'mae':  float(mean_absolute_error(y, all_preds)),
            'r2':   float(r2_score(y, all_preds))
        }
    }

# Run all three models
print('\nRunning LOSO-CV with residual collection...')

ridge_res = loso_cv_residuals(
    Ridge(alpha=1.0), X, y, settlements,
    scaler=StandardScaler, model_name='Ridge'
)
print('Ridge complete')

rf_res = loso_cv_residuals(
    RandomForestRegressor(
        n_estimators=300, max_depth=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
    X, y, settlements, model_name='Random Forest'
)
print('Random Forest complete')

xgb_res = loso_cv_residuals(
    xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, verbosity=0
    ),
    X, y, settlements, model_name='XGBoost'
)
print('XGBoost complete')

# ================================================================
# SECTION 3: RESIDUAL SUMMARY STATISTICS
# ================================================================
print('\n' + '=' * 65)
print('RESIDUAL ANALYSIS SUMMARY')
print('=' * 65)

models_results = {
    'Ridge':         ridge_res,
    'Random Forest': rf_res,
    'XGBoost':       xgb_res
}

for model_name, res in models_results.items():
    r = res['residuals']
    print(f'\n{model_name}:')
    print(f'  Mean residual:          {r.mean():.6f}  (bias — should be near 0)')
    print(f'  Std residual:           {r.std():.4f}')
    print(f'  Max absolute error:     {np.abs(r).max():.4f}')
    print(f'  % within ±0.05 SCMI:   {(np.abs(r) <= 0.05).mean()*100:.1f}%')
    print(f'  % within ±0.10 SCMI:   {(np.abs(r) <= 0.10).mean()*100:.1f}%')

    # Shapiro-Wilk normality test on residuals
    # Sample 500 if too large for the test
    r_sample = r if len(r) <= 5000 else np.random.choice(r, 5000, replace=False)
    stat, p  = stats.shapiro(r_sample[:500])
    print(f'  Shapiro-Wilk p-value:   {p:.4f}  '
          f'({"normal" if p > 0.05 else "non-normal"} residuals)')

print('\nFold-level residual stats — Ridge:')
print(ridge_res['fold_results'][[
    'settlement', 'mean_residual', 'std_residual',
    'max_error', 'pct_within_005', 'pct_within_010'
]].to_string(index=False))

print('\nFold-level residual stats — Random Forest:')
print(rf_res['fold_results'][[
    'settlement', 'mean_residual', 'std_residual',
    'max_error', 'pct_within_005', 'pct_within_010'
]].to_string(index=False))

print('\nFold-level residual stats — XGBoost:')
print(xgb_res['fold_results'][[
    'settlement', 'mean_residual', 'std_residual',
    'max_error', 'pct_within_005', 'pct_within_010'
]].to_string(index=False))

# ================================================================
# SECTION 4: VISUALISATIONS
# ================================================================
print('\nGenerating residual plots...')

COLOURS = {
    'Ridge':         'steelblue',
    'Random Forest': 'seagreen',
    'XGBoost':       'tomato'
}

SETTLEMENT_COLOURS = {
    'Mathare':       '#e41a1c',
    'Kayole_Soweto': '#377eb8',
    'Kahawa_Soweto': '#4daf4a',
    'KCC':           '#984ea3',
    'Kambi_Moto':    '#ff7f00'
}

# --- Plot 1: Predicted vs Actual with Residual Colouring ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    'Predicted vs Actual SCMI — LOSO-CV\n'
    'Coloured by settlement',
    fontsize=13, fontweight='bold'
)

for ax, (model_name, res) in zip(axes, models_results.items()):
    preds = res['predictions']

    for settlement in np.unique(settlements):
        mask = settlements == settlement
        ax.scatter(
            y[mask], preds[mask],
            label=settlement,
            alpha=0.4, s=6,
            color=SETTLEMENT_COLOURS.get(settlement, 'grey')
        )

    # Perfect prediction line
    lims = [y.min(), y.max()]
    ax.plot(lims, lims, 'k--', linewidth=1.2, label='Perfect fit')

    # ±0.05 error bands
    ax.fill_between(
        lims,
        [l - 0.05 for l in lims],
        [l + 0.05 for l in lims],
        alpha=0.1, color='grey', label='±0.05 band'
    )

    ax.set_xlabel('Actual SCMI', fontsize=10)
    ax.set_ylabel('Predicted SCMI', fontsize=10)
    ax.set_title(
        f'{model_name}\n'
        f'R²={res["overall"]["r2"]:.3f}  '
        f'RMSE={res["overall"]["rmse"]:.4f}',
        fontsize=10
    )

    if ax == axes[0]:
        ax.legend(fontsize=6, markerscale=2)

plt.tight_layout()
plt.savefig('residual_plot1_predicted_vs_actual.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Saved: residual_plot1_predicted_vs_actual.png')

# --- Plot 2: Residual Distribution per Model ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    'Residual Distributions — LOSO-CV\n'
    '(Actual minus Predicted SCMI)',
    fontsize=13, fontweight='bold'
)

for ax, (model_name, res) in zip(axes, models_results.items()):
    r      = res['residuals']
    colour = COLOURS[model_name]

    ax.hist(r, bins=50, color=colour, alpha=0.8,
            edgecolor='white', density=True)

    # Overlay normal distribution fit
    x_range = np.linspace(r.min(), r.max(), 200)
    ax.plot(
        x_range,
        stats.norm.pdf(x_range, r.mean(), r.std()),
        'k-', linewidth=1.5, label='Normal fit'
    )

    ax.axvline(0, color='black', linestyle='--',
               linewidth=1.2, label='Zero residual')
    ax.axvline(r.mean(), color='red', linestyle='-',
               linewidth=1, label=f'Mean: {r.mean():.5f}')

    ax.set_xlabel('Residual (Actual − Predicted)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(
        f'{model_name}\n'
        f'Mean={r.mean():.5f}  Std={r.std():.4f}',
        fontsize=10
    )
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('residual_plot2_distribution.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Saved: residual_plot2_distribution.png')

# --- Plot 3: Residuals vs Predicted (Heteroscedasticity Check) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    'Residuals vs Predicted SCMI\n'
    '(Heteroscedasticity Check — random scatter = good)',
    fontsize=13, fontweight='bold'
)

for ax, (model_name, res) in zip(axes, models_results.items()):
    preds  = res['predictions']
    r      = res['residuals']
    colour = COLOURS[model_name]

    for settlement in np.unique(settlements):
        mask = settlements == settlement
        ax.scatter(
            preds[mask], r[mask],
            alpha=0.3, s=6,
            color=SETTLEMENT_COLOURS.get(settlement, 'grey')
        )

    ax.axhline(0,     color='black', linestyle='--', linewidth=1.2)
    ax.axhline(0.05,  color='grey',  linestyle=':',  linewidth=0.8)
    ax.axhline(-0.05, color='grey',  linestyle=':',  linewidth=0.8)

    # Lowess trend line to detect systematic bias
    from scipy.stats import pearsonr
    corr, _ = pearsonr(preds, r)

    ax.set_xlabel('Predicted SCMI', fontsize=10)