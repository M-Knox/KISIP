import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import json

st.set_page_config(page_title="Study Area", page_icon="🗺️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.page-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.18em;
    color: #4FC3A1; text-transform: uppercase; margin-bottom: 0.4rem;
}
.page-title { font-size: 1.8rem; font-weight: 700; color: #E8EAF0; margin-bottom: 0.2rem; }
.page-sub { font-size: 0.9rem; color: #8B95A8; margin-bottom: 1.5rem; }
.legend-box {
    background: #1A1F2E; border: 1px solid #252D3D;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem;
}
.legend-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #4FC3A1;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.6rem;
}
.legend-item { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem; font-size: 0.85rem; color: #C5CAD6; }
.dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
.stat-pill {
    background: rgba(79,195,161,0.1); border: 1px solid rgba(79,195,161,0.25);
    border-radius: 8px; padding: 0.6rem 1rem; text-align: center; margin-bottom: 0.5rem;
}
.stat-pill .sv { font-size: 1.4rem; font-weight: 700; color: #4FC3A1; }
.stat-pill .sl { font-size: 0.7rem; color: #8B95A8; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-eyebrow">01 / Study Area</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Nairobi Settlement Map</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">All 9 analysis settlements — 5 KISIP-treated (observed SCMI) and 4 Mukuru sites (predicted SCMI). Click any zone for details.</div>', unsafe_allow_html=True)

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_kisip_zones():
    gdf = gpd.read_file("data/kisip_zones_spatial.geojson").to_crs("EPSG:4326")
    gdf["type"] = "KISIP"
    return gdf

@st.cache_data
def load_mukuru_preds():
    return pd.read_csv("data/kisip_mukuru_predictions.csv")

@st.cache_data
def load_mukuru_zones():
    gdf = gpd.read_file("data/mukuru_zones_spatial.geojson").to_crs("EPSG:4326")
    gdf["type"] = "Mukuru"
    return gdf

kisip_gdf  = load_kisip_zones()
mukuru_gdf = load_mukuru_zones()
mukuru_preds = load_mukuru_preds()

# Merge Mukuru predictions into zones
# Merge Mukuru predictions into zones.
# Drop 'settlement' from preds CSV first — the GDF already has it,
# and a duplicate causes pandas to create settlement_x / settlement_y.
mukuru_merged = mukuru_gdf.merge(
    mukuru_preds[["zone_id", "ensemble_scmi"]],
    on="zone_id", how="left"
)
# kisip already has SCMI column — rename to ensemble_scmi for uniform treatment
kisip_plot = kisip_gdf[["zone_id","settlement","SCMI","geometry","type"]].copy()
kisip_plot = kisip_plot.rename(columns={"SCMI":"ensemble_scmi"})
mukuru_plot = mukuru_merged[["zone_id","settlement","ensemble_scmi","geometry","type"]].copy()

# Combine for study-area view
all_zones = pd.concat([kisip_plot, mukuru_plot], ignore_index=True)

KISIP_SETTLEMENTS  = sorted(kisip_gdf["settlement"].unique().tolist())
MUKURU_SETTLEMENTS = sorted(mukuru_preds["settlement"].unique().tolist())

col_map, col_info = st.columns([3, 1])

with col_map:
    # ── Folium map ────────────────────────────────────────────────────────────
    m = folium.Map(
        location=[-1.300, 36.855],
        zoom_start=12,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    # KISIP zones — teal palette
    kisip_layer = folium.FeatureGroup(name="KISIP Treated Settlements")
    for _, row in kisip_plot.iterrows():
        scmi_val = row["ensemble_scmi"] if pd.notna(row["ensemble_scmi"]) else 0
        opacity  = 0.4 + 0.5 * min(scmi_val / 0.4, 1.0)
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda f, op=opacity: {
                "fillColor": "#4FC3A1",
                "color": "#2D8A74",
                "weight": 0.6,
                "fillOpacity": op,
            },
            tooltip=folium.Tooltip(
                f"<b>{row['settlement']}</b><br>Zone: {row['zone_id']}<br>SCMI: {scmi_val:.4f}",
                sticky=False,
            ),
        ).add_to(kisip_layer)
    kisip_layer.add_to(m)

    # Mukuru zones — amber palette
    mukuru_layer = folium.FeatureGroup(name="Mukuru Sites (Predicted)")
    for _, row in mukuru_plot.iterrows():
        scmi_val = row["ensemble_scmi"] if pd.notna(row["ensemble_scmi"]) else 0
        opacity  = 0.35 + 0.55 * min(scmi_val / 0.15, 1.0)
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda f, op=opacity: {
                "fillColor": "#F5A623",
                "color": "#C47D0E",
                "weight": 0.6,
                "fillOpacity": op,
            },
            tooltip=folium.Tooltip(
                f"<b>{row['settlement']}</b><br>Zone: {row['zone_id']}<br>Predicted SCMI: {scmi_val:.4f}",
                sticky=False,
            ),
        ).add_to(mukuru_layer)
    mukuru_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=560, returned_objects=[])

with col_info:
    st.markdown("""
    <div class="legend-box">
        <div class="legend-title">Layer legend</div>
        <div class="legend-item"><div class="dot" style="background:#4FC3A1"></div>KISIP treated (observed SCMI)</div>
        <div class="legend-item"><div class="dot" style="background:#F5A623"></div>Mukuru sites (predicted SCMI)</div>
        <div class="legend-item" style="margin-top:0.5rem; font-size:0.75rem; color:#8B95A8;">
            Zone opacity scales with SCMI magnitude
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Per-settlement summary stats
    st.markdown('<div class="legend-title" style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;color:#4FC3A1;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.6rem;">KISIP settlements</div>', unsafe_allow_html=True)
    for s in KISIP_SETTLEMENTS:
        subset = kisip_plot[kisip_plot["settlement"] == s]["ensemble_scmi"]
        st.markdown(f"""
        <div class="stat-pill">
            <div class="sv">{subset.mean():.3f}</div>
            <div class="sl">{s} · mean SCMI</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="legend-title" style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;color:#F5A623;text-transform:uppercase;letter-spacing:0.12em;margin-top:0.8rem;margin-bottom:0.6rem;">Mukuru sites</div>', unsafe_allow_html=True)
    for s in MUKURU_SETTLEMENTS:
        subset = mukuru_preds[mukuru_preds["settlement"] == s]["ensemble_scmi"]
        st.markdown(f"""
        <div class="stat-pill" style="background:rgba(245,166,35,0.08);border-color:rgba(245,166,35,0.25)">
            <div class="sv" style="color:#F5A623">{subset.mean():.3f}</div>
            <div class="sl">{s.replace('_', ' ')} · mean pred. SCMI</div>
        </div>""", unsafe_allow_html=True)
