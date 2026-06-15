# ================================================================
# KISIP — Stage 7: Spatial Mapping
# Joins SCMI, predictions and SHAP values to zone geometries
# Produces choropleth maps for:
#   - KISIP treated settlements (SCMI + dominant SHAP feature)
#   - Mukuru untreated settlements (predicted SCMI + readiness)
# ================================================================

import os
os.chdir(r'C:\Users\USER\Documents\KISIP')

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# SECTION 1: LOAD ZONE GEOMETRIES
# ================================================================
print('Loading zone geometries...')

# KISIP zones
kisip_zones = gpd.read_file(r'data\kisip_analysis_zones_50m.geojson')
kisip_zones = kisip_zones.to_crs(epsg=4326)
print(f'KISIP zones loaded: {len(kisip_zones)}')
print(f'KISIP columns: {kisip_zones.columns.tolist()}')

# Mukuru zones
mukuru_zones = gpd.read_file(r'data\mukuru_analysis_zones.geojson')
mukuru_zones = mukuru_zones.to_crs(epsg=4326)
print(f'Mukuru zones loaded: {len(mukuru_zones)}')

# ================================================================
# SECTION 2: LOAD ANALYTICAL RESULTS
# ================================================================
print('\nLoading analytical results...')

# KISIP — SCMI and predictions
kisip_scmi    = pd.read_csv(r'data\kisip_zone_scmi.csv')
kisip_preds   = pd.read_csv(r'data\kisip_model_predictions.csv')
kisip_shap    = pd.read_csv(r'data\kisip_shap_values.csv')
kisip_features = pd.read_csv(r'data\kisip_baseline_features_9final.csv')

# Mukuru — predictions and SHAP
mukuru_preds  = pd.read_csv(r'data\kisip_mukuru_predictions.csv')
mukuru_shap   = pd.read_csv(r'data\kisip_mukuru_shap_values.csv')

print('All result files loaded')

# ================================================================
# SECTION 3: MERGE RESULTS WITH GEOMETRIES (Fixed)
# ================================================================
print('\nMerging results with geometries...')

# Step 1 — Start with zone geometries
# Keep only essential geometry columns
kisip_merged = kisip_zones[['zone_id', 'settlement', 'geometry']].copy()

# Step 2 — Merge SCMI and CVA Direction
kisip_merged = kisip_merged.merge(
    kisip_scmi[['zone_id', 'SCMI', 'CVA_Direction']],
    on='zone_id',
    how='left'
)

# Step 3 — Merge predictions only (avoid re-importing SCMI and settlement)
kisip_merged = kisip_merged.merge(
    kisip_preds[['zone_id', 'xgb_pred', 'ridge_pred', 'rf_pred']],
    on='zone_id',
    how='left'
)

# Step 4 — Merge SHAP values only
shap_cols = [c for c in kisip_shap.columns
             if c.startswith('shap_') or c == 'zone_id']
kisip_merged = kisip_merged.merge(
    kisip_shap[shap_cols],
    on='zone_id',
    how='left'
)

print(f'KISIP merged shape:   {kisip_merged.shape}')
print(f'KISIP merged columns: {kisip_merged.columns.tolist()}')
print(f'SCMI null count:      {kisip_merged["SCMI"].isnull().sum()}')
print(f'Null geometries:      {kisip_merged.geometry.isna().sum()}')

# Mukuru — same clean approach
mukuru_merged = mukuru_zones[['zone_id', 'settlement', 'geometry']].copy()

mukuru_merged = mukuru_merged.merge(
    mukuru_preds[['zone_id', 'ensemble_scmi',
                  'ridge_scmi', 'rf_scmi', 'xgb_scmi']],
    on='zone_id',
    how='left'
)

mukuru_shap_cols = [c for c in mukuru_shap.columns
                    if c.startswith('shap_') or c == 'zone_id']
mukuru_merged = mukuru_merged.merge(
    mukuru_shap[mukuru_shap_cols],
    on='zone_id',
    how='left'
)

print(f'\nMukuru merged shape:   {mukuru_merged.shape}')
print(f'Mukuru merged columns: {mukuru_merged.columns.tolist()}')
print(f'ensemble_scmi nulls:   {mukuru_merged["ensemble_scmi"].isnull().sum()}')
print(f'Null geometries:       {mukuru_merged.geometry.isna().sum()}')

# ================================================================
# SECTION 4: DOMINANT SHAP FEATURE PER ZONE (unchanged)
# ================================================================
FEATURE_COLS = [
    'NDVI', 'NDBI', 'MNDWI',
    'Contrast', 'Entropy', 'Homogeneity', 'Correlation',
    'road_density', 'paved_proportion'
]

shap_cols_kisip  = [f'shap_{f}' for f in FEATURE_COLS]
shap_cols_mukuru = [f'shap_{f}' for f in FEATURE_COLS]

kisip_merged['dominant_feature'] = (
    kisip_merged[shap_cols_kisip].abs().idxmax(axis=1)
    .str.replace('shap_', '', regex=False)
)

mukuru_merged['dominant_feature'] = (
    mukuru_merged[shap_cols_mukuru].abs().idxmax(axis=1)
    .str.replace('shap_', '', regex=False)
)

print('\nDominant feature distribution — KISIP zones:')
print(kisip_merged['dominant_feature'].value_counts().to_string())
print('\nDominant feature distribution — Mukuru zones:')
print(mukuru_merged['dominant_feature'].value_counts().to_string())


# ================================================================
# SECTION 5: COLOUR MAPS AND FEATURE COLOURS
# ================================================================
FEATURE_COLOURS = {
    'NDVI':             '#2ecc71',
    'NDBI':             '#e74c3c',
    'MNDWI':            '#3498db',
    'Contrast':         '#9b59b6',
    'Entropy':          '#f39c12',
    'Homogeneity':      '#1abc9c',
    'Correlation':      '#e67e22',
    'road_density':     '#34495e',
    'paved_proportion': '#95a5a6'
}

SETTLEMENT_COLOURS_KISIP = {
    'Mathare':       '#e41a1c',
    'Kayole_Soweto': '#377eb8',
    'Kahawa_Soweto': '#4daf4a',
    'KCC':           '#984ea3',
    'Kambi_Moto':    '#ff7f00'
}

SETTLEMENT_COLOURS_MUKURU = {
    'Milimani_village': '#2ecc71',
    'area_48_village':  '#3498db',
    'wapewape_village': '#e67e22',
    'Sisal_village':    '#e74c3c'
}

# ================================================================
# SECTION 6: MAP 1 — KISIP SCMI CHOROPLETH PER SETTLEMENT
# One subplot per settlement showing zone-level SCMI
# ================================================================
print('\nGenerating Map 1: KISIP SCMI Choropleth...')

kisip_settlements = ['Mathare', 'Kayole_Soweto',
                     'Kahawa_Soweto', 'KCC', 'Kambi_Moto']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    'KISIP Treated Settlements — Zone-level SCMI\n'
    '(Settlement Change Magnitude Index, CVA)',
    fontsize=14, fontweight='bold'
)
axes = axes.flatten()

# Global SCMI range for consistent colour scale
scmi_min = kisip_merged['SCMI'].quantile(0.02)
scmi_max = kisip_merged['SCMI'].quantile(0.98)

for i, settlement in enumerate(kisip_settlements):
    ax     = axes[i]
    subset = kisip_merged[kisip_merged['settlement'] == settlement]

    if len(subset) == 0:
        ax.set_visible(False)
        continue

    subset.plot(
        column='SCMI',
        ax=ax,
        cmap='YlOrRd',
        vmin=scmi_min,
        vmax=scmi_max,
        edgecolor='white',
        linewidth=0.2,
        legend=False
    )

    mean_scmi = subset['SCMI'].mean()
    n_zones   = len(subset)

    ax.set_title(
        f'{settlement.replace("_", " ")}\n'
        f'Mean SCMI: {mean_scmi:.4f}  |  Zones: {n_zones}',
        fontsize=10
    )
    ax.set_axis_off()

# Hide unused subplot
axes[-1].set_visible(False)

# Shared colorbar
sm = plt.cm.ScalarMappable(
    cmap='YlOrRd',
    norm=plt.Normalize(vmin=scmi_min, vmax=scmi_max)
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, shrink=0.4,
                    orientation='vertical', pad=0.02)
cbar.set_label('SCMI (Change Magnitude)', fontsize=11)

plt.tight_layout()
plt.savefig('map1_kisip_scmi_choropleth.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: map1_kisip_scmi_choropleth.png')

# ================================================================
# SECTION 7: MAP 2 — DOMINANT SHAP FEATURE PER ZONE (KISIP)
# Colour each zone by which feature most influenced its prediction
# ================================================================
print('Generating Map 2: Dominant SHAP Feature Map...')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    'KISIP Treated Settlements — Dominant TreeSHAP Feature per Zone',
    fontsize=14, fontweight='bold'
)
axes = axes.flatten()

for i, settlement in enumerate(kisip_settlements):
    ax     = axes[i]
    subset = kisip_merged[
        kisip_merged['settlement'] == settlement
    ].copy()

    if len(subset) == 0:
        ax.set_visible(False)
        continue

    # Plot each feature group separately for legend
    plotted_features = subset['dominant_feature'].unique()

    for feature in plotted_features:
        feature_zones = subset[subset['dominant_feature'] == feature]
        feature_zones.plot(
            ax=ax,
            color=FEATURE_COLOURS.get(feature, 'grey'),
            edgecolor='white',
            linewidth=0.2,
            label=feature
        )

    ax.set_title(
        f'{settlement.replace("_", " ")}',
        fontsize=10
    )
    ax.set_axis_off()

axes[-1].set_visible(False)

# Legend
legend_patches = [
    mpatches.Patch(
        color=FEATURE_COLOURS[f], label=f
    )
    for f in FEATURE_COLS
    if f in kisip_merged['dominant_feature'].values
]
fig.legend(
    handles=legend_patches,
    loc='lower right',
    fontsize=9,
    title='Dominant Feature',
    title_fontsize=10,
    framealpha=0.9
)

plt.tight_layout()
plt.savefig('map2_kisip_dominant_shap_feature.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Saved: map2_kisip_dominant_shap_feature.png')

# ================================================================
# SECTION 8: MAP 3 — CVA DIRECTION MAP (KISIP)
# Shows nature of change: built-up increase vs greening
# ================================================================
print('Generating Map 3: CVA Direction Map...')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    'KISIP Treated Settlements — CVA Direction\n'
    '(Positive = Greening/De-densification | '
    'Negative = Built-up Intensification)',
    fontsize=13, fontweight='bold'
)
axes = axes.flatten()

for i, settlement in enumerate(kisip_settlements):
    ax     = axes[i]
    subset = kisip_merged[kisip_merged['settlement'] == settlement]

    if len(subset) == 0:
        ax.set_visible(False)
        continue

    subset.plot(
        column='CVA_Direction',
        ax=ax,
        cmap='RdBu',
        vmin=-90,
        vmax=90,
        edgecolor='white',
        linewidth=0.2,
        legend=False
    )

    ax.set_title(
        f'{settlement.replace("_", " ")}',
        fontsize=10
    )
    ax.set_axis_off()

axes[-1].set_visible(False)

sm2 = plt.cm.ScalarMappable(
    cmap='RdBu',
    norm=plt.Normalize(vmin=-90, vmax=90)
)
sm2.set_array([])
cbar2 = fig.colorbar(sm2, ax=axes, shrink=0.4,
                     orientation='vertical', pad=0.02)
cbar2.set_label('CVA Direction (degrees)', fontsize=11)

plt.tight_layout()
plt.savefig('map3_kisip_cva_direction.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Saved: map3_kisip_cva_direction.png')

# ================================================================
# SECTION 9: MAP 4 — MUKURU PREDICTED SCMI CHOROPLETH
# ================================================================
print('Generating Map 4: Mukuru Readiness Choropleth...')

mukuru_settlements = [
    'Sisal_village', 'Milimani_village',
    'area_48_village', 'wapewape_village'
]

fig, axes = plt.subplots(1, 4, figsize=(20, 6))
fig.suptitle(
    'Mukuru Settlements — Predicted SCMI Readiness Profiles\n'
    '(Ensemble Model: Ridge + Random Forest + XGBoost)',
    fontsize=13, fontweight='bold'
)

# Colour scale anchored to KISIP training range
kisip_mean = 0.0928
pred_min   = mukuru_merged['ensemble_scmi'].quantile(0.02)
pred_max   = mukuru_merged['ensemble_scmi'].quantile(0.98)

for i, settlement in enumerate(mukuru_settlements):
    ax     = axes[i]
    subset = mukuru_merged[
        mukuru_merged['settlement'] == settlement
    ]

    if len(subset) == 0:
        ax.set_visible(False)
        continue

    subset.plot(
        column='ensemble_scmi',
        ax=ax,
        cmap='RdYlGn',
        vmin=pred_min,
        vmax=pred_max,
        edgecolor='white',
        linewidth=0.2,
        legend=False
    )

    mean_pred = subset['ensemble_scmi'].mean()
    diff      = mean_pred - kisip_mean
    rank      = mukuru_settlements.index(settlement) + 1

    ax.set_title(
        f'{settlement.replace("_", " ")}\n'
        f'Rank {rank} | SCMI: {mean_pred:.4f} '
        f'({diff:+.4f} vs KISIP mean)',
        fontsize=9
    )
    ax.set_axis_off()

sm3 = plt.cm.ScalarMappable(
    cmap='RdYlGn',
    norm=plt.Normalize(vmin=pred_min, vmax=pred_max)
)
sm3.set_array([])
cbar3 = fig.colorbar(sm3, ax=axes,
                     shrink=0.6, orientation='vertical', pad=0.02)
cbar3.set_label('Predicted SCMI', fontsize=11)

# Add KISIP mean reference annotation
fig.text(
    0.5, 0.01,
    f'Reference: KISIP treated settlement mean SCMI = {kisip_mean:.4f}',
    ha='center', fontsize=10, style='italic', color='darkred'
)

plt.tight_layout()
plt.savefig('map4_mukuru_readiness_choropleth.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Saved: map4_mukuru_readiness_choropleth.png')

# ================================================================
# SECTION 10: MAP 5 — MUKURU DOMINANT SHAP FEATURE
# ================================================================
print('Generating Map 5: Mukuru Dominant SHAP Feature...')

fig, axes = plt.subplots(1, 4, figsize=(20, 6))
fig.suptitle(
    'Mukuru Settlements — Dominant TreeSHAP Feature per Zone',
    fontsize=13, fontweight='bold'
)

for i, settlement in enumerate(mukuru_settlements):
    ax     = axes[i]
    subset = mukuru_merged[
        mukuru_merged['settlement'] == settlement
    ].copy()

    if len(subset) == 0:
        ax.set_visible(False)
        continue

    for feature in subset['dominant_feature'].unique():
        feature_zones = subset[subset['dominant_feature'] == feature]
        feature_zones.plot(
            ax=ax,
            color=FEATURE_COLOURS.get(feature, 'grey'),
            edgecolor='white',
            linewidth=0.2,
            label=feature
        )

    ax.set_title(
        f'{settlement.replace("_", " ")}',
        fontsize=9
    )
    ax.set_axis_off()

legend_patches_mukuru = [
    mpatches.Patch(color=FEATURE_COLOURS[f], label=f)
    for f in FEATURE_COLS
    if f in mukuru_merged['dominant_feature'].values
]
fig.legend(
    handles=legend_patches_mukuru,
    loc='lower right',
    fontsize=9,
    title='Dominant Feature',
    title_fontsize=10,
    framealpha=0.9
)

plt.tight_layout()
plt.savefig('map5_mukuru_dominant_shap_feature.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Saved: map5_mukuru_dominant_shap_feature.png')

# ================================================================
# SECTION 11: MAP 6 — COMBINED STUDY AREA OVERVIEW
# All 9 settlements (5 KISIP + 4 Mukuru) on one map
# Colour coded by settlement status
# ================================================================
print('Generating Map 6: Combined Study Area Overview...')

# Add status column
kisip_merged['status']  = 'KISIP Treated'
mukuru_merged['status'] = 'Untreated (Mukuru)'

# Align columns for concatenation
kisip_plot  = kisip_merged[['zone_id', 'settlement',
                             'status', 'geometry']].copy()
mukuru_plot = mukuru_merged[['zone_id', 'settlement',
                              'status', 'geometry']].copy()

combined = gpd.GeoDataFrame(
    pd.concat([kisip_plot, mukuru_plot], ignore_index=True),
    crs='EPSG:4326'
)

fig, ax = plt.subplots(figsize=(12, 10))

status_colours = {
    'KISIP Treated':       '#e74c3c',
    'Untreated (Mukuru)':  '#3498db'
}

for status, colour in status_colours.items():
    subset = combined[combined['status'] == status]
    subset.plot(
        ax=ax,
        color=colour,
        alpha=0.6,
        edgecolor='white',
        linewidth=0.1,
        label=status
    )

ax.set_title(
    'KISIP Study Area — All Settlements\n'
    'Treated (KISIP) vs Untreated (Mukuru) Informal Settlements, Nairobi',
    fontsize=13, fontweight='bold'
)
ax.legend(fontsize=11, loc='upper left')
ax.set_axis_off()

# Settlement name annotations
for settlement in kisip_settlements + mukuru_settlements:
    subset = combined[combined['settlement'] == settlement]
    if len(subset) > 0:
        centroid = subset.geometry.unary_union.centroid
        label    = settlement.replace('_', '\n')
        ax.annotate(
            label,
            xy=(centroid.x, centroid.y),
            fontsize=7,
            ha='center',
            va='center',
            color='black',
            fontweight='bold'
        )

plt.tight_layout()
plt.savefig('map6_combined_study_area.png',
            dpi=150, bbox_inches='tight')
plt.close()
print('Saved: map6_combined_study_area.png')

# ================================================================
# SECTION 12: EXPORT SPATIAL DATA AS GEOJSON
# For use in Streamlit app and GEE visualisation
# ================================================================
print('\nExporting spatial data...')

# KISIP zones with all attributes
kisip_export_cols = [
    'zone_id', 'settlement', 'SCMI', 'CVA_Direction',
    'xgb_pred', 'dominant_feature', 'geometry'
] + shap_cols_kisip

kisip_export = kisip_merged[[
    c for c in kisip_export_cols
    if c in kisip_merged.columns
]]
kisip_export.to_file(
    'kisip_zones_spatial.geojson',
    driver='GeoJSON'
)
print('Saved: kisip_zones_spatial.geojson')

# Mukuru zones with all attributes
mukuru_export_cols = [
    'zone_id', 'settlement', 'ensemble_scmi',
    'dominant_feature', 'geometry'
] + shap_cols_mukuru

mukuru_export = mukuru_merged[[
    c for c in mukuru_export_cols
    if c in mukuru_merged.columns
]]
mukuru_export.to_file(
    'mukuru_zones_spatial.geojson',
    driver='GeoJSON'
)
print('Saved: mukuru_zones_spatial.geojson')

print('\n' + '=' * 65)
print('STAGE 7 COMPLETE')
print('=' * 65)
print('Maps generated:')
print('  map1_kisip_scmi_choropleth.png')
print('  map2_kisip_dominant_shap_feature.png')
print('  map3_kisip_cva_direction.png')
print('  map4_mukuru_readiness_choropleth.png')
print('  map5_mukuru_dominant_shap_feature.png')
print('  map6_combined_study_area.png')
print('\nSpatial exports:')
print('  kisip_zones_spatial.geojson')
print('  mukuru_zones_spatial.geojson')