# ================================================================
# KISIP — Feature Distribution Histograms
# Generates Figure 4.3 / Figure 5.1
# ================================================================

import os
os.chdir(r'C:\Users\USER\Documents\KISIP')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

features = pd.read_csv(r'data\kisip_baseline_features_9final.csv')

FEATURE_COLS = [
    'NDVI', 'NDBI', 'MNDWI',
    'Contrast', 'Entropy', 'Homogeneity', 'Correlation',
    'road_density', 'paved_proportion'
]

FEATURE_LABELS = {
    'NDVI':             'NDVI\n(Vegetation Index)',
    'NDBI':             'NDBI\n(Built-up Index)',
    'MNDWI':            'MNDWI\n(Water Index)',
    'Contrast':         'Contrast\n(GLCM Texture)',
    'Entropy':          'Entropy\n(GLCM Texture)',
    'Homogeneity':      'Homogeneity\n(GLCM Texture)',
    'Correlation':      'Correlation\n(GLCM Texture)',
    'road_density':     'Road Density\n(m/km²)',
    'paved_proportion': 'Paved Proportion\n(0–1)'
}

SETTLEMENT_COLOURS = {
    'Mathare':       '#e41a1c',
    'Kayole_Soweto': '#377eb8',
    'Kahawa_Soweto': '#4daf4a',
    'KCC':           '#984ea3',
    'Kambi_Moto':    '#ff7f00'
}

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle(
    'Distribution of 9 Baseline Features Across 2,445 Analysis Zones\n'
    'Coloured by KISIP Settlement',
    fontsize=14, fontweight='bold', y=1.02
)

axes = axes.flatten()

for i, feature in enumerate(FEATURE_COLS):
    ax = axes[i]

    # Overall distribution
    ax.hist(
        features[feature].dropna(),
        bins=40,
        color='lightgrey',
        edgecolor='white',
        alpha=0.6,
        density=True,
        label='All zones'
    )

    # Per-settlement overlay
    for settlement, colour in SETTLEMENT_COLOURS.items():
        subset = features[features['settlement'] == settlement][feature].dropna()
        if len(subset) > 0:
            ax.hist(
                subset,
                bins=30,
                color=colour,
                alpha=0.4,
                density=True,
                label=settlement.replace('_', ' ')
            )

    # Mean line
    mean_val = features[feature].mean()
    ax.axvline(
        mean_val,
        color='black',
        linestyle='--',
        linewidth=1.2,
        label=f'Mean: {mean_val:.3f}'
    )

    ax.set_title(FEATURE_LABELS[feature], fontsize=10, fontweight='bold')
    ax.set_xlabel('Feature Value', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.tick_params(labelsize=8)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='lower center',
    ncol=7,
    fontsize=9,
    bbox_to_anchor=(0.5, -0.02),
    title='Settlement',
    title_fontsize=10,
    framealpha=0.9
)

plt.tight_layout()
plt.savefig(
    'figure_feature_distributions.png',
    dpi=200,
    bbox_inches='tight',
    facecolor='white'
)
plt.close()
print('Saved: figure_feature_distributions.png')