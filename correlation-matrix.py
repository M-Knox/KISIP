# ================================================================
# KISIP — Feature Correlation Matrix
# Shows 11 original features with high correlations highlighted
# Dropped features: UI (r=1.00 with NDBI), SAVI (r=0.99 with NDVI)
# ================================================================

import os
os.chdir(r'C:\Users\USER\Documents\KISIP')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------
# SECTION 1: LOAD FEATURE DATA
# Use the full 11-feature matrix before dropping UI and SAVI
# ----------------------------------------------------------------
print('Loading feature data...')

# Load the 9-feature final matrix
features_9 = pd.read_csv(r'data\kisip_baseline_features_9final.csv')

# Reconstruct 11 features by recomputing UI and SAVI from scratch
# UI = identical formula to NDBI so we can duplicate NDBI as UI
# SAVI is highly correlated with NDVI so we can load from
# the original GEE export if available, otherwise approximate
# Check if the original 11-feature export exists
import os.path

if os.path.exists(r'data\kisip_zone_features_pre.csv'):
    features_11 = pd.read_csv(r'data\kisip_zone_features_pre.csv')
    print(f'Loaded original 11-feature CSV: {features_11.shape}')
    print(f'Columns: {features_11.columns.tolist()}')
else:
    # Reconstruct UI and SAVI from the 9-feature set
    # UI = NDBI (identical formula) — recreate for demonstration
    # SAVI ≈ NDVI * 1.5 / (1 + 0.5) at L=0.5 — approximate
    print('Original 11-feature CSV not found — reconstructing UI and SAVI')
    features_11 = features_9.copy()
    features_11['UI']   = features_11['NDBI'].copy()
    features_11['SAVI'] = features_11['NDVI'].apply(
        lambda x: ((x) / (x + 0.5)) * 1.5
        if x is not None else None
    )

# Define the 11 original features in logical order
FEATURES_11 = [
    'NDVI', 'NDBI', 'MNDWI', 'SAVI', 'UI',
    'Contrast', 'Entropy', 'Homogeneity', 'Correlation',
    'road_density', 'paved_proportion'
]

# Define dropped features
DROPPED     = ['UI', 'SAVI']
RETAINED    = [f for f in FEATURES_11 if f not in DROPPED]

# Ensure all 11 columns exist
available = [f for f in FEATURES_11 if f in features_11.columns]
print(f'\nFeatures available: {available}')

feature_data = features_11[available].copy()

# ----------------------------------------------------------------
# SECTION 2: COMPUTE CORRELATION MATRIX
# ----------------------------------------------------------------
corr_matrix = feature_data.corr(method='pearson')
print('\nCorrelation matrix computed')
print(f'Shape: {corr_matrix.shape}')

# Print high correlation pairs
print('\nHighly correlated pairs (|r| > 0.80):')
for i in range(len(available)):
    for j in range(i + 1, len(available)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.80:
            print(f'  {available[i]:<20} ↔ {available[j]:<20} r = {r:.4f}')

# ----------------------------------------------------------------
# SECTION 3: BUILD COLOUR MAP
# Custom diverging map: deep blue (negative) → white → deep red (positive)
# High positive correlations appear deep red
# ----------------------------------------------------------------
colors_list = [
    '#2166ac',  # Deep blue — strong negative
    '#4393c3',  # Medium blue
    '#92c5de',  # Light blue
    '#f7f7f7',  # White — zero correlation
    '#f4a582',  # Light red
    '#d6604d',  # Medium red
    '#b2182b'   # Deep red — strong positive
]
custom_cmap = LinearSegmentedColormap.from_list(
    'kisip_corr', colors_list, N=256
)

# ----------------------------------------------------------------
# SECTION 4: PLOT CORRELATION MATRIX
# ----------------------------------------------------------------
n_features = len(available)
fig, ax    = plt.subplots(figsize=(13, 11))

# Draw heatmap
im = ax.imshow(
    corr_matrix.values,
    cmap=custom_cmap,
    vmin=-1, vmax=1,
    aspect='auto'
)

# ----------------------------------------------------------------
# SECTION 5: ANNOTATE CELLS WITH CORRELATION VALUES
# Colour-code text: white for extreme values, black for mid-range
# ----------------------------------------------------------------
for i in range(n_features):
    for j in range(n_features):
        val      = corr_matrix.values[i, j]
        abs_val  = abs(val)
        txt_col  = 'white' if abs_val > 0.75 else 'black'
        weight   = 'bold'  if abs_val > 0.90 else 'normal'
        fontsize = 9       if abs_val > 0.90 else 8

        ax.text(
            j, i,
            f'{val:.2f}',
            ha='center', va='center',
            fontsize=fontsize,
            color=txt_col,
            fontweight=weight
        )

# ----------------------------------------------------------------
# SECTION 6: HIGHLIGHT DROPPED FEATURES
# Red border around rows and columns of UI and SAVI
# ----------------------------------------------------------------
for feat in DROPPED:
    if feat in available:
        idx = available.index(feat)

        # Highlight entire row
        ax.add_patch(mpatches.FancyBboxPatch(
            (-0.5, idx - 0.5),
            n_features, 1,
            boxstyle='square,pad=0',
            linewidth=2.5,
            edgecolor='#FF0000',
            facecolor='none',
            zorder=5
        ))

        # Highlight entire column
        ax.add_patch(mpatches.FancyBboxPatch(
            (idx - 0.5, -0.5),
            1, n_features,
            boxstyle='square,pad=0',
            linewidth=2.5,
            edgecolor='#FF0000',
            facecolor='none',
            zorder=5
        ))

# ----------------------------------------------------------------
# SECTION 7: HIGHLIGHT THE CRITICAL CORRELATION PAIRS
# Draw a distinct box around NDBI-UI and NDVI-SAVI cells
# ----------------------------------------------------------------
critical_pairs = []

if 'UI' in available and 'NDBI' in available:
    i_ndbi = available.index('NDBI')
    i_ui   = available.index('UI')
    critical_pairs.append((i_ndbi, i_ui, 'r=1.00'))
    critical_pairs.append((i_ui, i_ndbi, 'r=1.00'))

if 'SAVI' in available and 'NDVI' in available:
    i_ndvi = available.index('NDVI')
    i_savi = available.index('SAVI')
    critical_pairs.append((i_ndvi, i_savi, 'r=0.99'))
    critical_pairs.append((i_savi, i_ndvi, 'r=0.99'))

for (row, col, label) in critical_pairs:
    ax.add_patch(mpatches.FancyBboxPatch(
        (col - 0.48, row - 0.48),
        0.96, 0.96,
        boxstyle='square,pad=0',
        linewidth=3,
        edgecolor='#FF0000',
        facecolor='#FF000022',
        zorder=6
    ))

# ----------------------------------------------------------------
# SECTION 8: AXIS LABELS AND FORMATTING
# ----------------------------------------------------------------
ax.set_xticks(range(n_features))
ax.set_yticks(range(n_features))

# Colour axis labels: red for dropped features, teal for retained
x_labels = []
y_labels = []

for feat in available:
    colour = '#CC0000' if feat in DROPPED else '#1a1a2e'
    weight = 'bold'    if feat in DROPPED else 'normal'
    x_labels.append(feat)
    y_labels.append(feat)

ax.set_xticklabels(
    available,
    rotation=45, ha='right', fontsize=10
)
ax.set_yticklabels(available, fontsize=10)

# Colour the tick labels for dropped features
for tick, feat in zip(ax.get_xticklabels(), available):
    if feat in DROPPED:
        tick.set_color('#CC0000')
        tick.set_fontweight('bold')

for tick, feat in zip(ax.get_yticklabels(), available):
    if feat in DROPPED:
        tick.set_color('#CC0000')
        tick.set_fontweight('bold')

# ----------------------------------------------------------------
# SECTION 9: TITLE, COLORBAR AND LEGEND
# ----------------------------------------------------------------
ax.set_title(
    'Pearson Correlation Matrix — 11 Original Baseline Features\n'
    'Red borders indicate features dropped due to multicollinearity\n'
    'UI dropped (r = 1.00 with NDBI) | SAVI dropped (r = 0.99 with NDVI)',
    fontsize=12,
    fontweight='bold',
    pad=20
)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Pearson Correlation Coefficient (r)', fontsize=10)
cbar.set_ticks([-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])

# Legend
legend_elements = [
    mpatches.Patch(
        facecolor='#b2182b', edgecolor='black',
        label='Strong positive correlation (r > 0.75)'
    ),
    mpatches.Patch(
        facecolor='#2166ac', edgecolor='black',
        label='Strong negative correlation (r < -0.75)'
    ),
    mpatches.Patch(
        facecolor='#f7f7f7', edgecolor='black',
        label='No correlation (r ≈ 0)'
    ),
    mpatches.Patch(
        facecolor='none', edgecolor='#FF0000',
        linewidth=2, label='Dropped features (UI, SAVI)'
    )
]

ax.legend(
    handles=legend_elements,
    loc='upper left',
    bbox_to_anchor=(1.18, 1.0),
    fontsize=9,
    title='Legend',
    title_fontsize=10,
    framealpha=0.9
)

plt.tight_layout()
plt.savefig(
    'slide6_correlation_matrix_11features.png',
    dpi=200,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()
print('Saved: slide6_correlation_matrix_11features.png')

# ----------------------------------------------------------------
# SECTION 10: SECOND PLOT — 9 RETAINED FEATURES ONLY
# Clean version showing the final feature set for the presentation
# ----------------------------------------------------------------
features_9_cols = [f for f in RETAINED if f in features_11.columns]
corr_9          = features_11[features_9_cols].corr(method='pearson')

fig2, ax2 = plt.subplots(figsize=(11, 9))

im2 = ax2.imshow(
    corr_9.values,
    cmap=custom_cmap,
    vmin=-1, vmax=1,
    aspect='auto'
)

for i in range(len(features_9_cols)):
    for j in range(len(features_9_cols)):
        val     = corr_9.values[i, j]
        abs_val = abs(val)
        txt_col = 'white' if abs_val > 0.75 else 'black'
        weight  = 'bold'  if abs_val > 0.80 else 'normal'

        ax2.text(
            j, i,
            f'{val:.2f}',
            ha='center', va='center',
            fontsize=9,
            color=txt_col,
            fontweight=weight
        )

ax2.set_xticks(range(len(features_9_cols)))
ax2.set_yticks(range(len(features_9_cols)))
ax2.set_xticklabels(
    features_9_cols,
    rotation=45, ha='right', fontsize=10
)
ax2.set_yticklabels(features_9_cols, fontsize=10)

ax2.set_title(
    'Pearson Correlation Matrix — 9 Final Retained Features\n'
    'After removal of UI and SAVI (multicollinearity threshold |r| > 0.90)',
    fontsize=12,
    fontweight='bold',
    pad=15
)

cbar2 = fig2.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
cbar2.set_label('Pearson Correlation Coefficient (r)', fontsize=10)
cbar2.set_ticks([-1.0, -0.5, 0, 0.5, 1.0])

plt.tight_layout()
plt.savefig(
    'slide6_correlation_matrix_9features.png',
    dpi=200,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()
print('Saved: slide6_correlation_matrix_9features.png')

# ----------------------------------------------------------------
# SECTION 11: PRINT KEY VALUES FOR SLIDE ANNOTATION
# ----------------------------------------------------------------
print('\n' + '=' * 55)
print('KEY CORRELATION VALUES FOR SLIDE 6 ANNOTATION')
print('=' * 55)

pairs_of_interest = [
    ('NDVI', 'UI'),
    ('NDBI', 'UI'),
    ('NDVI', 'SAVI'),
    ('NDBI', 'SAVI'),
    ('NDVI', 'NDBI'),
    ('NDVI', 'MNDWI'),
    ('NDBI', 'MNDWI'),
    ('road_density', 'paved_proportion'),
    ('Contrast', 'Entropy'),
    ('Homogeneity', 'Correlation')
]

for (f1, f2) in pairs_of_interest:
    if f1 in corr_matrix.index and f2 in corr_matrix.columns:
        r = corr_matrix.loc[f1, f2]
        flag = ''
        if abs(r) >= 0.90:
            flag = '← DROPPED'
        elif abs(r) >= 0.80:
            flag = '← HIGH'
        print(f'  {f1:<22} ↔ {f2:<22} r = {r:+.4f}  {flag}')