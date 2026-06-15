import sys
from pathlib import Path

_DEMO = Path(__file__).resolve().parent.parent
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from utils.data_loaders import load_kisip_zones, load_mukuru_predictions, load_mukuru_zones
from utils.interpret import is_technical, metric_block, page_explainer, scmi_interpretation
from utils.maps import add_study_area_layers, base_map
from utils.sidebar import render_sidebar
from utils.styling import inject_styles, page_header

st.set_page_config(page_title="Study Area", page_icon="🗺️", layout="wide")

inject_styles()
render_sidebar()

lead = (
    "All nine study sites — five KISIP-treated (measured change) and four Mukuru "
    "sites (model-estimated change). Hover a zone for details."
    if not is_technical()
    else "Combined 50 m zones — KISIP observed SCMI and Mukuru ensemble_scmi. "
    "Opacity scales with magnitude."
)

page_header("01 · Study area", "Nairobi settlement map", lead)
page_explainer("study_area")

kisip_gdf = load_kisip_zones()
mukuru_gdf = load_mukuru_zones()
mukuru_preds = load_mukuru_predictions()

kisip_plot = kisip_gdf[["zone_id", "settlement", "SCMI", "geometry", "type"]].copy()
kisip_plot = kisip_plot.rename(columns={"SCMI": "ensemble_scmi"})
mukuru_plot = mukuru_gdf[["zone_id", "settlement", "ensemble_scmi", "geometry", "type"]].copy()

KISIP_SETTLEMENTS = sorted(kisip_gdf["settlement"].unique().tolist())
MUKURU_SETTLEMENTS = sorted(mukuru_preds["settlement"].unique().tolist())

col_map, col_info = st.columns([3, 1], gap="large")

with col_map:
    m = base_map((-1.300, 36.855), zoom=12)
    add_study_area_layers(m, kisip_plot, mukuru_plot)
    st_folium(m, width=None, height=520, returned_objects=[])

with col_info:
    st.markdown('<div class="section-title">Legend</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="list-row"><span class="status-dot dot-green"></span> KISIP treated</div>
<div class="list-row"><span class="status-dot dot-amber"></span> Mukuru predicted</div>
<p class="section-caption">Darker fill = stronger change signal</p>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Mean change by site</div>', unsafe_allow_html=True)
    for s in KISIP_SETTLEMENTS:
        mean_val = kisip_plot[kisip_plot["settlement"] == s]["ensemble_scmi"].mean()
        hint = scmi_interpretation(mean_val) if not is_technical() else f"SCMI {mean_val:.4f}"
        st.markdown(metric_block(s.replace("_", " "), f"{mean_val:.3f}", hint), unsafe_allow_html=True)

    for s in MUKURU_SETTLEMENTS:
        mean_val = mukuru_preds[mukuru_preds["settlement"] == s]["ensemble_scmi"].mean()
        label = s.replace("_", " ")
        hint = scmi_interpretation(mean_val) if not is_technical() else f"pred. {mean_val:.4f}"
        st.markdown(
            metric_block(label, f"{mean_val:.3f}", hint),
            unsafe_allow_html=True,
        )
