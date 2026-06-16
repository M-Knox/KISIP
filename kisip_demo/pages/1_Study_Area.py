import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from styles import inject_css

st.set_page_config(page_title="Study Area", page_icon="🗺️", layout="wide")
inject_css()

@st.cache_data
def load_kisip_zones():
    gdf = gpd.read_file("data/kisip_zones_spatial.geojson").to_crs("EPSG:4326")
    gdf["type"] = "KISIP"
    return gdf

@st.cache_data
def load_mukuru_zones():
    gdf = gpd.read_file("data/mukuru_zones_spatial.geojson").to_crs("EPSG:4326")
    gdf["type"] = "Mukuru"
    return gdf

@st.cache_data
def load_mukuru_preds():
    return pd.read_csv("data/kisip_mukuru_predictions.csv")

kisip_gdf    = load_kisip_zones()
mukuru_gdf   = load_mukuru_zones()
mukuru_preds = load_mukuru_preds()

kisip_plot  = kisip_gdf[["zone_id","settlement","SCMI","geometry","type"]].copy().rename(columns={"SCMI":"ensemble_scmi"})
mukuru_plot = mukuru_gdf.copy()  # ensemble_scmi already present

KISIP_SETTLEMENTS  = sorted(kisip_gdf["settlement"].unique().tolist())
MUKURU_SETTLEMENTS = sorted(mukuru_preds["settlement"].unique().tolist())

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="pg-label">01 / Study Area</div>', unsafe_allow_html=True)
st.markdown('<div class="pg-title">Nairobi Settlement Map</div>', unsafe_allow_html=True)
st.markdown('<div class="pg-sub">All 9 analysis settlements. Teal = KISIP treated zones (observed SCMI). Amber = Mukuru sites (predicted SCMI). Zone opacity scales with change magnitude — hover any zone for details.</div>', unsafe_allow_html=True)

col_map, col_info = st.columns([4, 1], gap="large")

with col_map:
    m = folium.Map(
        location=[-1.300, 36.855],
        zoom_start=12,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    kisip_layer = folium.FeatureGroup(name="KISIP Treated")
    for _, row in kisip_plot.iterrows():
        sv      = row["ensemble_scmi"] if pd.notna(row["ensemble_scmi"]) else 0
        opacity = 0.35 + 0.55 * min(sv / 0.4, 1.0)
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda f, op=opacity: {
                "fillColor": "#4FC3A1", "color": "#2A7A63",
                "weight": 0.5, "fillOpacity": op,
            },
            tooltip=folium.Tooltip(
                f"<b>{row['settlement']}</b><br>Zone {row['zone_id']}<br>SCMI {sv:.4f}",
                sticky=False,
            ),
        ).add_to(kisip_layer)
    kisip_layer.add_to(m)

    mukuru_layer = folium.FeatureGroup(name="Mukuru Sites")
    for _, row in mukuru_plot.iterrows():
        sv      = row["ensemble_scmi"] if pd.notna(row["ensemble_scmi"]) else 0
        opacity = 0.35 + 0.55 * min(sv / 0.15, 1.0)
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda f, op=opacity: {
                "fillColor": "#F5A623", "color": "#B87A10",
                "weight": 0.5, "fillOpacity": op,
            },
            tooltip=folium.Tooltip(
                f"<b>{row['settlement'].replace('_',' ')}</b><br>Zone {row['zone_id']}<br>Pred. SCMI {sv:.4f}",
                sticky=False,
            ),
        ).add_to(mukuru_layer)
    mukuru_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=580, returned_objects=[])

with col_info:
    st.markdown('<div class="section-rule">Legend</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="leg-row"><div class="leg-dot" style="background:#4FC3A1"></div>KISIP treated</div>
    <div class="leg-row"><div class="leg-dot" style="background:#F5A623"></div>Mukuru sites</div>
    <div style="font-size:0.75rem;color:#8B95A8;margin-top:0.5rem;line-height:1.5;">
        Opacity encodes SCMI magnitude. Use layer control on the map to toggle groups.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-rule">KISIP mean SCMI</div>', unsafe_allow_html=True)
    for s in KISIP_SETTLEMENTS:
        val = kisip_plot[kisip_plot["settlement"] == s]["ensemble_scmi"].mean()
        st.markdown(f"""
        <div class="rank-item">
          <div class="rank-name" style="font-size:0.8rem;">{s.replace('_',' ')}</div>
          <div class="rank-score">{val:.3f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-rule">Mukuru pred. SCMI</div>', unsafe_allow_html=True)
    for s in MUKURU_SETTLEMENTS:
        val = mukuru_preds[mukuru_preds["settlement"] == s]["ensemble_scmi"].mean()
        st.markdown(f"""
        <div class="rank-item">
          <div class="rank-name" style="font-size:0.8rem;">{s.replace('_',' ')}</div>
          <div class="rank-score" style="color:#F5A623">{val:.3f}</div>
        </div>""", unsafe_allow_html=True)
