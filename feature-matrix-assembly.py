# ================================================================
# KISIP Study — Stage 2D: Feature Matrix Assembly
# Merges 9 spectral/texture features (GEE) with
# 2 road network features (OSMnx) into full 11-feature matrix
# ================================================================

import os
os.chdir(r'C:\Users\USER\Documents\KISIP')

import pandas as pd
import numpy as np

# ----------------------------------------------------------------
# SECTION 1: LOAD BOTH FEATURE CSVs
# ----------------------------------------------------------------
print('Loading feature files...')

spectral_texture = pd.read_csv(r'data\kisip_zone_features_pre.csv')
road_features    = pd.read_csv(r'data\kisip_road_features.csv')

print(f'Spectral/texture shape: {spectral_texture.shape}')
print(f'Road features shape:    {road_features.shape}')
print(f'\nSpectral columns:  {spectral_texture.columns.tolist()}')
print(f'Road columns:      {road_features.columns.tolist()}')

# ----------------------------------------------------------------
# SECTION 2: INSPECT BEFORE MERGING
# Check zone_id format matches between both files
# ----------------------------------------------------------------
print(f'\nSpectral zone_id sample: {spectral_texture["zone_id"].head(3).tolist()}')
print(f'Road zone_id sample:     {road_features["zone_id"].head(3).tolist()}')

print(f'\nSpectral settlements: {spectral_texture["settlement"].unique()}')
print(f'Road settlements:     {road_features["settlement"].unique()}')

# ----------------------------------------------------------------
# SECTION 3: MERGE ON ZONE ID
# Inner join ensures only zones present in both datasets
# ----------------------------------------------------------------
feature_matrix = pd.merge(
    spectral_texture,
    road_features[['zone_id', 'road_density', 'paved_proportion']],
    on='zone_id',
    how='inner'
)

print(f'\nMerged matrix shape: {feature_matrix.shape}')
print(f'Columns: {feature_matrix.columns.tolist()}')

# Check for merge losses
if len(feature_matrix) < len(spectral_texture):
    lost = len(spectral_texture) - len(feature_matrix)
    print(f'WARNING: {lost} zones lost in merge — zone_id mismatch between files')

# ----------------------------------------------------------------
# SECTION 4: DEFINE FEATURE COLUMNS
# ----------------------------------------------------------------
FEATURE_COLS = [
    # Spectral indices
    'NDVI', 'NDBI', 'MNDWI', 'SAVI', 'UI',
    # Texture metrics
    'Contrast', 'Entropy', 'Homogeneity', 'Correlation',
    # Road network
    'road_density', 'paved_proportion'
]

# Verify all 11 features are present
missing_features = [f for f in FEATURE_COLS if f not in feature_matrix.columns]
if missing_features:
    print(f'\nERROR: Missing feature columns: {missing_features}')
else:
    print(f'\nAll 11 features confirmed present')

# ----------------------------------------------------------------
# SECTION 5: HANDLE MISSING VALUES
# ----------------------------------------------------------------
print('\nMissing values per feature column:')
print(feature_matrix[FEATURE_COLS].isnull().sum())

# Fill nulls with settlement-level mean
# Handles zones where cloud cover caused null spectral values
for col in FEATURE_COLS:
    null_count = feature_matrix[col].isnull().sum()
    if null_count > 0:
        settlement_means = feature_matrix.groupby('settlement')[col].transform('mean')
        feature_matrix[col] = feature_matrix[col].fillna(settlement_means)
        print(f'Filled {null_count} nulls in {col} with settlement mean')

# Check if any nulls remain after fill
remaining_nulls = feature_matrix[FEATURE_COLS].isnull().sum().sum()
if remaining_nulls > 0:
    print(f'WARNING: {remaining_nulls} nulls remain — filling with global mean')
    feature_matrix[FEATURE_COLS] = feature_matrix[FEATURE_COLS].fillna(
        feature_matrix[FEATURE_COLS].mean()
    )

# ----------------------------------------------------------------
# SECTION 6: SUMMARY STATISTICS
# ----------------------------------------------------------------
print('\n' + '=' * 60)
print('FEATURE MATRIX SUMMARY')
print('=' * 60)

print(f'\nZones per settlement:')
print(feature_matrix['settlement'].value_counts().to_string())

print(f'\nFeature statistics:')
print(feature_matrix[FEATURE_COLS].describe().round(4).to_string())

print(f'\nFeature correlations with each other:')
print(feature_matrix[FEATURE_COLS].corr().round(3).to_string())

# ----------------------------------------------------------------
# SECTION 7: FLAG HIGH CORRELATIONS
# Features correlated > 0.90 may cause redundancy in modelling
# ----------------------------------------------------------------
print('\nHighly correlated feature pairs (|r| > 0.90):')
corr_matrix = feature_matrix[FEATURE_COLS].corr().abs()
high_corr    = []

for i in range(len(FEATURE_COLS)):
    for j in range(i + 1, len(FEATURE_COLS)):
        r = corr_matrix.iloc[i, j]
        if r > 0.90:
            high_corr.append({
                'feature_1': FEATURE_COLS[i],
                'feature_2': FEATURE_COLS[j],
                'correlation': round(r, 4)
            })

if high_corr:
    print(pd.DataFrame(high_corr).to_string(index=False))
    print('Note: Consider reviewing highly correlated pairs before modelling')
else:
    print('No feature pairs exceed 0.90 correlation threshold')

# ----------------------------------------------------------------
# SECTION 7B: DROP REDUNDANT FEATURES
# UI dropped — identical to NDBI (correlation = 1.0)
# SAVI dropped — near-identical to NDVI (correlation = 0.9935)
# Retains 9 independent features for modelling
# ----------------------------------------------------------------
FEATURES_TO_DROP = ['UI', 'SAVI']

print('\nDropping redundant features:')
for f in FEATURES_TO_DROP:
    print(f'  Dropping {f}')

feature_matrix = feature_matrix.drop(columns=FEATURES_TO_DROP)

# Update feature column list
FEATURE_COLS_FINAL = [f for f in FEATURE_COLS if f not in FEATURES_TO_DROP]

print(f'\nFinal feature set ({len(FEATURE_COLS_FINAL)} features):')
for i, f in enumerate(FEATURE_COLS_FINAL, 1):
    print(f'  {i}. {f}')

# Verify no remaining high correlations
print('\nPost-drop correlation check (|r| > 0.80):')
corr_matrix_final = feature_matrix[FEATURE_COLS_FINAL].corr().abs()
remaining_high    = []

for i in range(len(FEATURE_COLS_FINAL)):
    for j in range(i + 1, len(FEATURE_COLS_FINAL)):
        r = corr_matrix_final.iloc[i, j]
        if r > 0.80:
            remaining_high.append({
                'feature_1': FEATURE_COLS_FINAL[i],
                'feature_2': FEATURE_COLS_FINAL[j],
                'correlation': round(r, 4)
            })

if remaining_high:
    print(pd.DataFrame(remaining_high).to_string(index=False))
else:
    print('No feature pairs exceed 0.80 threshold — feature set is clean')

# ----------------------------------------------------------------
# SECTION 8: EXPORT FINAL FEATURE MATRIX (9 features)
# ----------------------------------------------------------------
output_cols    = ['zone_id', 'settlement'] + FEATURE_COLS_FINAL
feature_matrix = feature_matrix[output_cols]

feature_matrix.to_csv('kisip_baseline_features_9final.csv', index=False)

print('\n' + '=' * 60)
print('EXPORT COMPLETE')
print('=' * 60)
print(f'File: kisip_baseline_features_9final.csv')
print(f'Shape: {feature_matrix.shape}')
print(f'Features retained: {", ".join(FEATURE_COLS_FINAL)}')


