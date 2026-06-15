# KISIP Settlement Intelligence — Streamlit Demo

XGBoost ensemble model demo for quantifying physical surface change in Nairobi's informal settlements.

## Setup

### 1. Copy this folder into your project root
Your directory should look like:
```
your_project/
├── kisip_demo/          ← this folder
│   ├── app.py
│   ├── pages/
│   ├── requirements.txt
│   └── .streamlit/
├── data/                ← your existing data folder
└── ...
```

### 2. Install dependencies (inside your venv)
```bash
# Activate your existing venv
source KISIP.venv/bin/activate   # macOS/Linux
# or
KISIP.venv\Scripts\activate      # Windows

# Install Streamlit additions
pip install streamlit folium streamlit-folium plotly
```

### 3. Run the app
```bash
# From your project root (NOT from inside kisip_demo/)
streamlit run kisip_demo/app.py
```
The app reads `data/` relative to where you launch it — so always run from the project root.

---

## Data files used

| File | Used by |
|------|---------|
| `data/kisip_zones_spatial.geojson` | Pages 1, 2 |
| `data/mukuru_zones_spatial.geojson` | Pages 1, 3 |
| `data/kisip_model_predictions.csv` | Page 2 |
| `data/kisip_mukuru_predictions.csv` | Pages 1, 3 |
| `data/kisip_shap_by_settlement.csv` | Page 2 |
| `data/kisip_mukuru_shap_attribution.csv` | Page 3 |
| `data/kisip_mukuru_readiness_profiles.csv` | Page 3 |
| `data/kisip_model_comparison.csv` | Page 4 |
| `data/kisip_cva_vs_pca_comparison.csv` | Page 4 |
| `data/kisip_shap_importance.csv` | Page 4 |
| `data/kisip_baseline_features_9final.csv` | Page 2 (feature table) |
| `data/kisip_zone_scmi_both.csv` | Page 2 (CVA direction) |

---

## Recording the demo

```bash
# Clean URL (hides toolbar for recording)
http://localhost:8501?embed=true
```

Recommended walkthrough order:
1. **Landing page** — introduce the project, point out the 5 metrics
2. **Page 1** — show the full settlement map, hover a few zones
3. **Page 2** — select Mathare, walk through SCMI choropleth → SHAP → feature table
4. **Page 3** — select a Mukuru settlement, explain the readiness tier + gap to baseline
5. **Page 4** — CVA vs PCA chart, key findings

Total walkthrough: ~6–8 minutes for a presentation demo.

---

## Troubleshooting

**`ModuleNotFoundError: streamlit_folium`**
```bash
pip install streamlit-folium
```

**Map shows blank / wrong location**
Check that GeoJSON CRS is EPSG:4326. The app calls `.to_crs("EPSG:4326")` automatically.

**`KeyError` on a column name**
Run the head commands in the README to confirm column names match what's in your CSVs. Column names are loaded from the exact files — no hardcoded assumptions beyond what you provided.

**`data/` not found**
You must run `streamlit run kisip_demo/app.py` from the project root, not from inside `kisip_demo/`.
