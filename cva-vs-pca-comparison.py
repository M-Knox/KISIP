# ================================================================
# KISIP — SCMI Method Comparison: CVA vs PCA
# Trains Ridge, RF and XGBoost on both SCMI versions
# Compares LOSO-CV performance to validate CVA choice
# ================================================================

import os
os.chdir(r'C:\Users\USER\Documents\KISIP')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model  import Ridge
from sklearn.ensemble      import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------
# SECTION 1: LOAD DATA
# ----------------------------------------------------------------
features  = pd.read_csv(r'data\kisip_baseline_features_9final.csv')
scmi_both = pd.read_csv(r'data\kisip_zone_scmi_both.csv')

# Merge
data = pd.merge(
    features,
    scmi_both[['zone_id', 'SCMI', 'SCMI_PCA']],
    on='zone_id',
    how='inner'
)

print(f'Dataset shape: {data.shape}')
print(f'\nSCMI CVA stats:\n{data["SCMI"].describe().round(4)}')
print(f'\nSCMI PCA stats:\n{data["SCMI_PCA"].describe().round(4)}')

# ----------------------------------------------------------------
# SECTION 2: CORRELATION BETWEEN THE TWO SCMI VERSIONS
# ----------------------------------------------------------------
corr = data['SCMI'].corr(data['SCMI_PCA'])
print(f'\nCorrelation between CVA-SCMI and PCA-SCMI: {corr:.4f}')

# ----------------------------------------------------------------
# SECTION 3: FEATURE SETUP
# ----------------------------------------------------------------
FEATURE_COLS = [
    'NDVI', 'NDBI', 'MNDWI',
    'Contrast', 'Entropy', 'Homogeneity', 'Correlation',
    'road_density', 'paved_proportion'
]

X           = data[FEATURE_COLS].values
settlements = data['settlement'].values
y_cva       = data['SCMI'].values
y_pca       = data['SCMI_PCA'].values

# ----------------------------------------------------------------
# SECTION 4: LOSO-CV FUNCTION
# ----------------------------------------------------------------
def loso_cv(model, X, y, settlements, scaler=None):
    unique_settlements = np.unique(settlements)
    all_preds          = np.zeros_like(y)

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
        all_preds[test_mask] = model.predict(X_test)

    return {
        'rmse': np.sqrt(mean_squared_error(y, all_preds)),
        'mae':  mean_absolute_error(y, all_preds),
        'r2':   r2_score(y, all_preds)
    }

# ----------------------------------------------------------------
# SECTION 5: RUN COMPARISON
# Both SCMI versions across all three models
# ----------------------------------------------------------------
models = {
    'Ridge':         (Ridge(alpha=1.0),                      StandardScaler),
    'Random Forest': (RandomForestRegressor(n_estimators=300, max_depth=10,
                      min_samples_leaf=5, random_state=42), None),
    'XGBoost':       (xgb.XGBRegressor(n_estimators=300, max_depth=6,
                      learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=5,
                      random_state=42, verbosity=0),          None)
}

results = []

for model_name, (model, scaler) in models.items():
    print(f'Running {model_name}...')

    cva_metrics = loso_cv(model, X, y_cva, settlements, scaler)
    pca_metrics = loso_cv(model, X, y_pca, settlements, scaler)

    results.append({
        'Model':       model_name,
        'SCMI_Method': 'CVA',
        'RMSE':        round(cva_metrics['rmse'], 4),
        'MAE':         round(cva_metrics['mae'],  4),
        'R2':          round(cva_metrics['r2'],   4)
    })
    results.append({
        'Model':       model_name,
        'SCMI_Method': 'PCA',
        'RMSE':        round(pca_metrics['rmse'], 4),
        'MAE':         round(pca_metrics['mae'],  4),
        'R2':          round(pca_metrics['r2'],   4)
    })

results_df = pd.DataFrame(results)
print('\n' + '=' * 60)
print('CVA vs PCA SCMI — MODEL COMPARISON')
print('=' * 60)
print(results_df.to_string(index=False))
results_df.to_csv('kisip_cva_vs_pca_comparison.csv', index=False)

# ----------------------------------------------------------------
# SECTION 6: VISUALISE COMPARISON
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('CVA vs PCA SCMI — LOSO-CV Model Performance', fontsize=13)

metrics  = ['RMSE', 'MAE', 'R2']
titles   = ['RMSE (lower = better)', 'MAE (lower = better)', 'R² (higher = better)']
colours  = {'CVA': 'steelblue', 'PCA': 'tomato'}

for ax, metric, title in zip(axes, metrics, titles):
    for method in ['CVA', 'PCA']:
        subset = results_df[results_df['SCMI_Method'] == method]
        ax.bar(
            [f'{m}\n({method})' for m in subset['Model']],
            subset[metric],
            color=colours[method],
            alpha=0.8,
            label=method
        )
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=15)
    if ax == axes[0]:
        ax.legend()

plt.tight_layout()
plt.savefig('kisip_cva_vs_pca_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved: kisip_cva_vs_pca_comparison.png')

# ----------------------------------------------------------------
# SECTION 7: SCATTER PLOT OF CVA vs PCA SCMI VALUES
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colours_s = {
    'Mathare':       '#e41a1c',
    'Kayole_Soweto': '#377eb8',
    'Kahawa_Soweto': '#4daf4a',
    'KCC':           '#984ea3',
    'Kambi_Moto':    '#ff7f00'
}

# CVA vs PCA scatter
ax = axes[0]
for settlement in np.unique(settlements):
    mask = data['settlement'] == settlement
    ax.scatter(
        data.loc[mask, 'SCMI'],
        data.loc[mask, 'SCMI_PCA'],
        label=settlement,
        alpha=0.4,
        s=8,
        color=colours_s.get(settlement, 'grey')
    )
ax.set_xlabel('CVA-SCMI')
ax.set_ylabel('PCA-SCMI')
ax.set_title(f'CVA vs PCA SCMI Values\n(r = {corr:.3f})')
ax.legend(fontsize=7, markerscale=2)

# Settlement mean comparison
ax = axes[1]
means = data.groupby('settlement')[['SCMI', 'SCMI_PCA']].mean().reset_index()
x     = np.arange(len(means))
width = 0.35

ax.bar(x - width/2, means['SCMI'],     width, label='CVA-SCMI', color='steelblue', alpha=0.8)
ax.bar(x + width/2, means['SCMI_PCA'], width, label='PCA-SCMI', color='tomato',    alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(means['settlement'], rotation=15, fontsize=8)
ax.set_ylabel('Mean SCMI')
ax.set_title('Settlement Mean SCMI — CVA vs PCA')
ax.legend()

plt.tight_layout()
plt.savefig('kisip_scmi_cva_vs_pca_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: kisip_scmi_cva_vs_pca_scatter.png')
print('\nSCMI comparison complete.')