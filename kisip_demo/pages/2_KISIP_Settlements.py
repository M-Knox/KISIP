import sys
from pathlib import Path

_DEMO = Path(__file__).resolve().parent.parent
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from utils.constants import FEATURES, MODEL_COLUMNS
from utils.data_loaders import load_kisip_page2_data
from utils.interpret import (
    feature_label,
    is_technical,
    metric_block,
    page_explainer,
    scmi_interpretation,
    term_label,
)
from utils.maps import base_map, direction_color, scmi_color
from utils.sidebar import render_sidebar
from utils.styling import inject_styles, page_header, plotly_layout

st.set_page_config(page_title="KISIP Settlements", page_icon="🏘️", layout="wide")

inject_styles()
render_sidebar()

model_name = st.session_state["selected_model"]
pred_col = MODEL_COLUMNS[model_name]["kisip"]

gdf, preds, shap_set, features, _scmi_df = load_kisip_page2_data()
KISIP_SETTLEMENTS = sorted(gdf["settlement"].unique().tolist())

lead = (
    "Measured physical change in each treated settlement, and which factors "
    "the model associated with that change."
    if not is_technical()
    else "Zone-level observed SCMI (CVA), model predictions, and XGBoost SHAP attribution."
)

page_header("02 · KISIP treated", "Change by settlement", lead)
page_explainer("kisip")

selected = st.selectbox(
    "Settlement",
    KISIP_SETTLEMENTS,
    format_func=lambda x: x.replace("_", " "),
)

s_gdf = gdf[gdf["settlement"] == selected].copy()
s_preds = preds[preds["settlement"] == selected].copy()
s_shap = shap_set[shap_set["settlement"] == selected].copy()
s_feats = features[features["settlement"] == selected].copy()

s_gdf = s_gdf.merge(
    s_preds[["zone_id", pred_col]],
    on="zone_id",
    how="left",
)

mean_obs = s_gdf["SCMI"].mean()
mean_pred = s_gdf[pred_col].mean()
n_zones = len(s_gdf)
dominant_feat = (
    s_gdf["dominant_feature"].value_counts().idxmax()
    if "dominant_feature" in s_gdf.columns
    else "—"
)

map_options = (
    ["Observed change", f"Model prediction ({model_name})", "Surface change direction"]
    if not is_technical()
    else ["Observed SCMI (CVA)", f"Prediction ({pred_col})", "CVA direction (°)"]
)
map_layer = st.radio("Map view", map_options, horizontal=True, label_visibility="collapsed")

col_map, col_right = st.columns([3, 2], gap="large")

with col_map:
    centroid_lat = s_gdf.geometry.centroid.y.mean()
    centroid_lon = s_gdf.geometry.centroid.x.mean()
    m = base_map((centroid_lat, centroid_lon), zoom=15)

    if map_layer.startswith("Observed") or map_layer.startswith("Observed SCMI"):
        value_col = "SCMI"
        vmin, vmax = s_gdf["SCMI"].min(), s_gdf["SCMI"].max()

        def color_fn(v):
            return scmi_color(v, vmin, vmax)

        def tooltip(row, val):
            return (
                f"<b>{row['zone_id']}</b><br>"
                f"Observed: <b>{val:.4f}</b><br>"
                f"{scmi_interpretation(val)}"
            )

    elif "direction" in map_layer.lower() or "CVA direction" in map_layer:
        value_col = "CVA_Direction"
        s_gdf[value_col] = s_gdf.get("CVA_Direction", 0).fillna(0)

        def color_fn(v):
            return direction_color(v)

        def tooltip(row, val):
            return f"<b>{row['zone_id']}</b><br>Direction: <b>{val:.1f}°</b>"

    else:
        value_col = pred_col
        vmin, vmax = s_gdf[pred_col].min(), s_gdf[pred_col].max()

        def color_fn(v):
            return scmi_color(v, vmin, vmax)

        def tooltip(row, val):
            return (
                f"<b>{row['zone_id']}</b><br>"
                f"Predicted: <b>{val:.4f}</b><br>"
                f"Model: {model_name}"
            )

    for _, row in s_gdf.iterrows():
        val = row[value_col] if pd.notna(row.get(value_col)) else 0
        folium_style = color_fn(val)
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda f, c=folium_style: {
                "fillColor": c,
                "color": "#0F1117",
                "weight": 0.5,
                "fillOpacity": 0.78,
            },
            tooltip=folium.Tooltip(tooltip(row, val), sticky=False),
        ).add_to(m)

    st_folium(m, width=None, height=440, returned_objects=[])

    st.markdown('<div class="section-title">Change across zones</div>', unsafe_allow_html=True)
    hist_fig = px.histogram(
        s_gdf,
        x="SCMI",
        nbins=20,
        labels={"SCMI": term_label("SCMI") if is_technical() else "Observed change"},
    )
    hist_fig.update_layout(**plotly_layout(height=180, showlegend=False, bargap=0.05))
    hist_fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    hist_fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    hist_fig.update_traces(marker_color="#4FC3A1")
    st.plotly_chart(hist_fig, use_container_width=True)

with col_right:
    obs_label = term_label("SCMI") if is_technical() else "Observed change"
    pred_label = f"Predicted ({model_name})" if is_technical() else "Model estimate"

    st.markdown(
        metric_block(obs_label, f"{mean_obs:.4f}", scmi_interpretation(mean_obs)),
        unsafe_allow_html=True,
    )
    st.markdown(
        metric_block(pred_label, f"{mean_pred:.4f}", scmi_interpretation(mean_pred)),
        unsafe_allow_html=True,
    )
    st.markdown(metric_block("Zones", str(n_zones), "50 m grid"), unsafe_allow_html=True)

    driver_label = feature_label(dominant_feat) if dominant_feat != "—" else "—"
    driver_title = "Strongest driver" if not is_technical() else "Top SHAP feature"
    st.markdown(metric_block(driver_title, driver_label), unsafe_allow_html=True)

    shap_title = (
        "What drove the prediction?"
        if not is_technical()
        else "Mean |SHAP| — XGBoost"
    )
    st.markdown(f'<div class="section-title">{shap_title}</div>', unsafe_allow_html=True)

    if not s_shap.empty:
        shap_vals = s_shap[FEATURES].mean().abs().sort_values(ascending=True)
    else:
        shap_cols = [c for c in s_gdf.columns if c.startswith("shap_")]
        shap_vals = s_gdf[shap_cols].mean().abs()
        shap_vals.index = shap_vals.index.str.replace("shap_", "")
        shap_vals = shap_vals.sort_values(ascending=True)

    y_labels = [feature_label(i) for i in shap_vals.index]
    top_n = 3
    colors = [
        "#4FC3A1" if i >= len(shap_vals) - top_n else "rgba(79,195,161,0.45)"
        for i in range(len(shap_vals))
    ]

    shap_fig = go.Figure(
        go.Bar(
            x=shap_vals.values,
            y=y_labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    shap_fig.update_layout(
        **plotly_layout(
            height=280,
            xaxis_title="Influence" if not is_technical() else "Mean |SHAP|",
        )
    )
    shap_fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    st.plotly_chart(shap_fig, use_container_width=True)

    if is_technical():
        st.caption("SHAP computed for XGBoost. Sidebar model controls predictions only.")

if not s_preds.empty and "SCMI" in s_preds.columns and pred_col in s_preds.columns:
    st.markdown('<div class="section-title">Observed vs predicted</div>', unsafe_allow_html=True)
    fig_ov = px.scatter(
        s_preds,
        x="SCMI",
        y=pred_col,
        hover_data=["zone_id"],
        labels={
            "SCMI": term_label("SCMI") if is_technical() else "Observed",
            pred_col: "Predicted",
        },
    )
    fig_ov.add_shape(
        type="line",
        x0=s_preds["SCMI"].min(),
        x1=s_preds["SCMI"].max(),
        y0=s_preds["SCMI"].min(),
        y1=s_preds["SCMI"].max(),
        line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dash"),
    )
    fig_ov.update_layout(**plotly_layout(height=300))
    fig_ov.update_traces(marker_color="#4FC3A1")
    fig_ov.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig_ov.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    st.plotly_chart(fig_ov, use_container_width=True)

with st.expander("Zone feature data", expanded=False):
    display_cols = ["zone_id"] + [c for c in FEATURES if c in s_feats.columns]
    rename = {c: feature_label(c) for c in FEATURES if c in s_feats.columns}
    show = s_feats[display_cols].rename(columns=rename).set_index("zone_id").round(5)
    st.dataframe(show, use_container_width=True, height=260)
