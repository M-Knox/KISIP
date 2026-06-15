# ================================================================
# KISIP — Hyperparameter Tuning
# Nested LOSO-CV for unbiased hyperparameter selection
# Focus: Ridge alpha, XGBoost regularisation parameters
# ================================================================

import os
os.chdir(r'C:\Users\USER\Documents\KISIP')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model  import Ridge
from sklearn.ensemble      import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------
# SECTION 1: LOAD DATA
# ----------------------------------------------------------------
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
unique_s    = np.unique(settlements)

# ----------------------------------------------------------------
# SECTION 2: NESTED LOSO-CV TUNING FUNCTION
# Outer loop: hold out one settlement for evaluation
# Inner loop: tune on remaining 4 settlements via 4-fold LOSO
# ----------------------------------------------------------------
def nested_loso_tune(param_grid, model_fn, X, y,
                     settlements, scaler=None):
    unique_s   = np.unique(settlements)
    all_preds  = np.zeros_like(y, dtype=float)
    best_params_per_fold = []

    for test_s in unique_s:
        # Outer split
        outer_train_mask = settlements != test_s
        outer_test_mask  = settlements == test_s

        X_outer_train = X[outer_train_mask]
        y_outer_train = y[outer_train_mask]
        X_outer_test  = X[outer_test_mask]
        y_outer_test  = y[outer_test_mask]
        s_outer_train = settlements[outer_train_mask]

        # Inner LOSO over the 4 remaining settlements
        best_params = None
        best_inner_r2 = -np.inf

        for params in param_grid:
            inner_preds = np.zeros_like(y_outer_train, dtype=float)
            inner_s     = np.unique(s_outer_train)

            for val_s in inner_s:
                inner_train = s_outer_train != val_s
                inner_val   = s_outer_train == val_s

                Xi_train = X_outer_train[inner_train]
                yi_train = y_outer_train[inner_train]
                Xi_val   = X_outer_train[inner_val]

                if scaler is not None:
                    sc       = scaler()
                    Xi_train = sc.fit_transform(Xi_train)
                    Xi_val   = sc.transform(Xi_val)

                m = model_fn(**params)
                m.fit(Xi_train, yi_train)
                inner_preds[inner_val] = m.predict(Xi_val)

            inner_r2 = r2_score(y_outer_train, inner_preds)
            if inner_r2 > best_inner_r2:
                best_inner_r2 = inner_r2
                best_params   = params

        best_params_per_fold.append({
            'held_out':    test_s,
            'best_params': best_params,
            'inner_r2':    round(best_inner_r2, 4)
        })

        # Retrain on full outer training set with best params
        X_tr = X_outer_train.copy()
        X_te = X_outer_test.copy()

        if scaler is not None:
            sc   = scaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

        best_model = model_fn(**best_params)
        best_model.fit(X_tr, y_outer_train)
        all_preds[outer_test_mask] = best_model.predict(X_te)

        print(f'  Held out: {test_s:<15} '
              f'Best params: {best_params}  '
              f'Inner R²: {best_inner_r2:.4f}')

    overall_r2   = r2_score(y, all_preds)
    overall_rmse = np.sqrt(mean_squared_error(y, all_preds))

    return {
        'predictions':  all_preds,
        'overall_r2':   overall_r2,
        'overall_rmse': overall_rmse,
        'fold_params':  best_params_per_fold
    }

# ----------------------------------------------------------------
# SECTION 3: RIDGE ALPHA TUNING
# ----------------------------------------------------------------
print('=' * 65)
print('RIDGE ALPHA TUNING — Nested LOSO-CV')
print('=' * 65)

ridge_param_grid = [
    {'alpha': 0.01},
    {'alpha': 0.1},
    {'alpha': 1.0},
    {'alpha': 10.0},
    {'alpha': 100.0}
]

ridge_tuning = nested_loso_tune(
    ridge_param_grid,
    Ridge,
    X, y, settlements,
    scaler=StandardScaler
)

print(f'\nRidge tuned — Overall R²:   {ridge_tuning["overall_r2"]:.4f}')
print(f'Ridge tuned — Overall RMSE: {ridge_tuning["overall_rmse"]:.4f}')
print(f'Ridge baseline R²:          0.3032')
print(f'Improvement:                {ridge_tuning["overall_r2"] - 0.3032:+.4f}')

print('\nBest alpha per fold:')
for fold in ridge_tuning['fold_params']:
    print(f'  {fold["held_out"]:<15} alpha={fold["best_params"]["alpha"]}')

# ----------------------------------------------------------------
# SECTION 4: XGBOOST REGULARISATION TUNING
# Focused grid on regularisation params most likely to help
# ----------------------------------------------------------------
print('\n' + '=' * 65)
print('XGBOOST REGULARISATION TUNING — Nested LOSO-CV')
print('=' * 65)

xgb_param_grid = [
    {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05,
     'subsample': 0.8, 'colsample_bytree': 0.8,
     'min_child_weight': 5,  'reg_alpha': 0.1, 'reg_lambda': 1.0,
     'random_state': 42, 'verbosity': 0},

    {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05,
     'subsample': 0.8, 'colsample_bytree': 0.8,
     'min_child_weight': 10, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
     'random_state': 42, 'verbosity': 0},

    {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
     'subsample': 0.8, 'colsample_bytree': 0.8,
     'min_child_weight': 5,  'reg_alpha': 0.1, 'reg_lambda': 1.0,
     'random_state': 42, 'verbosity': 0},

    {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.03,
     'subsample': 0.7, 'colsample_bytree': 0.7,
     'min_child_weight': 10, 'reg_alpha': 1.0, 'reg_lambda': 2.0,
     'random_state': 42, 'verbosity': 0},

    {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.02,
     'subsample': 0.8, 'colsample_bytree': 0.8,
     'min_child_weight': 10, 'reg_alpha': 0.5, 'reg_lambda': 1.0,
     'random_state': 42, 'verbosity': 0}
]

xgb_tuning = nested_loso_tune(
    xgb_param_grid,
    xgb.XGBRegressor,
    X, y, settlements,
    scaler=None
)

print(f'\nXGBoost tuned — Overall R²:   {xgb_tuning["overall_r2"]:.4f}')
print(f'XGBoost tuned — Overall RMSE: {xgb_tuning["overall_rmse"]:.4f}')
print(f'XGBoost baseline R²:          0.2115')
print(f'Improvement:                  {xgb_tuning["overall_r2"] - 0.2115:+.4f}')

print('\nBest params per fold:')
for fold in xgb_tuning['fold_params']:
    p = fold['best_params']
    print(f'  {fold["held_out"]:<15} '
          f'depth={p["max_depth"]} '
          f'lr={p["learning_rate"]} '
          f'mcw={p["min_child_weight"]} '
          f'alpha={p["reg_alpha"]}')

# ----------------------------------------------------------------
# SECTION 5: RANDOM FOREST TUNING
# Focused on min_samples_leaf — key regularisation parameter
# ----------------------------------------------------------------
print('\n' + '=' * 65)
print('RANDOM FOREST TUNING — Nested LOSO-CV')
print('=' * 65)

rf_param_grid = [
    {'n_estimators': 300, 'max_depth': 8,
     'min_samples_leaf': 5,  'random_state': 42, 'n_jobs': -1},
    {'n_estimators': 300, 'max_depth': 10,
     'min_samples_leaf': 5,  'random_state': 42, 'n_jobs': -1},
    {'n_estimators': 300, 'max_depth': 10,
     'min_samples_leaf': 10, 'random_state': 42, 'n_jobs': -1},
    {'n_estimators': 300, 'max_depth': 15,
     'min_samples_leaf': 5,  'random_state': 42, 'n_jobs': -1},
    {'n_estimators': 500, 'max_depth': 10,
     'min_samples_leaf': 10, 'random_state': 42, 'n_jobs': -1}
]

rf_tuning = nested_loso_tune(
    rf_param_grid,
    RandomForestRegressor,
    X, y, settlements,
    scaler=None
)

print(f'\nRandom Forest tuned — Overall R²:   {rf_tuning["overall_r2"]:.4f}')
print(f'Random Forest tuned — Overall RMSE: {rf_tuning["overall_rmse"]:.4f}')
print(f'Random Forest baseline R²:          0.2979')
print(f'Improvement:                        {rf_tuning["overall_r2"] - 0.2979:+.4f}')

print('\nBest params per fold:')
for fold in rf_tuning['fold_params']:
    p = fold['best_params']
    print(f'  {fold["held_out"]:<15} '
          f'depth={p["max_depth"]} '
          f'leaf={p["min_samples_leaf"]}')

# ----------------------------------------------------------------
# SECTION 6: COMPARISON TABLE
# ----------------------------------------------------------------
print('\n' + '=' * 65)
print('TUNING COMPARISON SUMMARY')
print('=' * 65)

comparison = pd.DataFrame([
    {'Model': 'Ridge',
     'Baseline R²': 0.3032,
     'Tuned R²':    round(ridge_tuning['overall_r2'], 4),
     'Baseline RMSE': 0.0378,
     'Tuned RMSE':  round(ridge_tuning['overall_rmse'], 4)},
    {'Model': 'Random Forest',
     'Baseline R²': 0.2979,
     'Tuned R²':    round(rf_tuning['overall_r2'], 4),
     'Baseline RMSE': 0.0377,
     'Tuned RMSE':  round(rf_tuning['overall_rmse'], 4)},
    {'Model': 'XGBoost',
     'Baseline R²': 0.2115,
     'Tuned R²':    round(xgb_tuning['overall_r2'], 4),
     'Baseline RMSE': 0.0397,
     'Tuned RMSE':  round(xgb_tuning['overall_rmse'], 4)}
])

comparison['R² Change']   = (comparison['Tuned R²'] -
                              comparison['Baseline R²']).round(4)
comparison['RMSE Change'] = (comparison['Tuned RMSE'] -
                              comparison['Baseline RMSE']).round(4)

print(comparison.to_string(index=False))
comparison.to_csv('kisip_hyperparameter_tuning_results.csv', index=False)

# ----------------------------------------------------------------
# SECTION 7: VISUALISE TUNING RESULTS
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    'Hyperparameter Tuning — Baseline vs Tuned Performance\n'
    'Nested LOSO-CV',
    fontsize=13, fontweight='bold'
)

models    = comparison['Model'].tolist()
x         = np.arange(len(models))
width     = 0.35

# R² comparison
ax = axes[0]
ax.bar(x - width/2, comparison['Baseline R²'],
       width, label='Baseline', color='steelblue', alpha=0.85)
ax.bar(x + width/2, comparison['Tuned R²'],
       width, label='Tuned', color='tomato', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel('R²')
ax.set_title('R² — Baseline vs Tuned')
ax.legend()
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

for i, (b, t) in enumerate(zip(
    comparison['Baseline R²'], comparison['Tuned R²']
)):
    ax.text(i - width/2, b + 0.005, f'{b:.3f}',
            ha='center', fontsize=8)
    ax.text(i + width/2, t + 0.005, f'{t:.3f}',
            ha='center', fontsize=8)

# RMSE comparison
ax = axes[1]
ax.bar(x - width/2, comparison['Baseline RMSE'],
       width, label='Baseline', color='steelblue', alpha=0.85)
ax.bar(x + width/2, comparison['Tuned RMSE'],
       width, label='Tuned', color='tomato', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel('RMSE')
ax.set_title('RMSE — Baseline vs Tuned')
ax.legend()

for i, (b, t) in enumerate(zip(
    comparison['Baseline RMSE'], comparison['Tuned RMSE']
)):
    ax.text(i - width/2, b + 0.0005, f'{b:.4f}',
            ha='center', fontsize=8)
    ax.text(i + width/2, t + 0.0005, f'{t:.4f}',
            ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('kisip_hyperparameter_tuning.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved: kisip_hyperparameter_tuning.png')

print('\n' + '=' * 65)
print('HYPERPARAMETER TUNING COMPLETE')
print('=' * 65)