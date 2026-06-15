import sys
from pathlib import Path

_DEMO = Path(__file__).resolve().parent.parent
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from utils.constants import MODEL_COLUMNS, TIER_COLORS
from utils.data_loaders import load_mukuru_page3_data
from utils.interpret import (
    feature_label,
    is_technical,
    metric_block,
    page_explainer,
    readiness_tier,
    scmi_interpretation,
    tier_dot_class,
    term_label,
)
from utils.maps import base_map, scmi_color
from utils.sidebar import render_sidebar
from utils.styling import inject_styles, page_header, plotly_layout

st.set_page_config(page_title="Mukuru Readiness", page_icon="📍", layout="wide")

inject_styles()
render_sidebar()

model_name = st.session_state["selected_model"]
pred_col = MODEL_COLUMNS[model_name]["mukuru"]

profiles, shap_df, mukuru_z, kisip_z, mukuru_preds = load_mukuru_page3_data()
kisip_baseline = kisip_z["SCMI"].mean()

lead = (
    "Estimated physical change at four Mukuru sites, ranked against the average "
    "change seen where KISIP work was completed."
    if not is_technical()
    else f"Model predictions ({pred_col}) benchmarked to KISIP mean SCMI {kisip_baseline:.4f}."
)

page_header("03 · Mukuru readiness", "Intervention readiness", lead)
page_explainer("mukuru")


def mukuru_profile_mean_col(pred_col: str) -> str:
    col = f"mean_{pred_col}"
    return col if col in profiles.columns else "mean_ensemble_scmi"


def settlement_option(row, mean_col: str) -> str:
    name = row["settlement"].replace("_", " ")
    val = row[mean_col]
    tier_label, tier_key = readiness_tier(val, kisip_baseline)
    emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}[tier_key]
    rank = int(row["readiness_rank"])
    if is_technical():
        return f"#{rank} · {name} · {tier_label} {emoji}"
    short_tier = tier_label.replace(" Readiness", "")
    return f"#{rank} · {name} · {emoji} {short_tier}"


mean_col = mukuru_profile_mean_col(pred_col)
prof_sorted = profiles.sort_values("readiness_rank")
option_labels = [settlement_option(r, mean_col) for _, r in prof_sorted.iterrows()]
settlement_keys = prof_sorted["settlement"].tolist()

col_rank, col_main = st.columns([1, 3], gap="large")

with col_rank:
    st.markdown('<div class="section-title">Ranking</div>', unsafe_allow_html=True)
    rows_html = ""
    for _, r in prof_sorted.iterrows():
        tier_label, tier_key = readiness_tier(r[mean_col], kisip_baseline)
        dot = tier_dot_class(tier_key)
        val = r[mean_col]
        rows_html += f"""
        <div class="list-row">
            <span class="list-rank">#{int(r['readiness_rank'])}</span>
            <span class="status-dot {dot}"></span>
            <span>{r['settlement'].replace('_', ' ')}</span>
            <span class="list-value">{val:.3f}</span>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)

    baseline_label = "KISIP average change" if not is_technical() else "KISIP baseline SCMI"
    st.markdown(
        metric_block(baseline_label, f"{kisip_baseline:.4f}"),
        unsafe_allow_html=True,
    )

with col_main:
    selected_idx = st.selectbox(
        "Settlement",
        range(len(settlement_keys)),
        format_func=lambda i: option_labels[i],
    )
    selected = settlement_keys[selected_idx]
    s_profile = profiles[profiles["settlement"] == selected].iloc[0]
    mean_pred = s_profile[mean_col]

    tier_label, tier_key = readiness_tier(mean_pred, kisip_baseline)
    dot = tier_dot_class(tier_key)
    st.markdown(
        f'<div class="status-line">'
        f'<span class="status-dot {dot}"></span>'
        f'<span>{tier_label}</span></div>',
        unsafe_allow_html=True,
    )

    s_zones = mukuru_z[mukuru_z["settlement"] == selected].copy()
    pred_data = mukuru_preds[mukuru_preds["settlement"] == selected][["zone_id", pred_col]]
    s_zones = s_zones.drop(columns=[pred_col], errors="ignore").merge(pred_data, on="zone_id")

    gap = kisip_baseline - mean_pred
    gap_text = f"{'Below' if gap > 0 else 'Above'} baseline by {abs(gap):.4f}"

    k1, k2, k3 = st.columns(3)
    with k1:
        pred_title = "Predicted change" if not is_technical() else f"Mean {pred_col}"
        st.markdown(
            metric_block(pred_title, f"{mean_pred:.4f}", scmi_interpretation(mean_pred)),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            metric_block("Zones", str(int(s_profile["zones"])), "50 m grid"),
            unsafe_allow_html=True,
        )
    with k3:
        gap_title = "Vs KISIP average" if not is_technical() else "Gap to baseline"
        st.markdown(metric_block(gap_title, f"{abs(gap):.4f}", gap_text), unsafe_allow_html=True)

    mc1, mc2 = st.columns([3, 2], gap="medium")

    with mc1:
        st.markdown('<div class="section-title">Predicted change map</div>', unsafe_allow_html=True)
        if not s_zones.empty and pred_col in s_zones.columns:
            centroid = s_zones.to_crs("EPSG:32737").geometry.centroid.to_crs("EPSG:4326")
            mm = base_map((centroid.y.mean(), centroid.x.mean()), zoom=15)
            vmin, vmax = s_zones[pred_col].min(), s_zones[pred_col].max()
            for _, row in s_zones.iterrows():
                sv = row[pred_col] if pd.notna(row[pred_col]) else 0
                c = scmi_color(sv, vmin, vmax)
                folium.GeoJson(
                    row["geometry"].__geo_interface__,
                    style_function=lambda f, col=c: {
                        "fillColor": col,
                        "color": "#0F1117",
                        "weight": 0.5,
                        "fillOpacity": 0.78,
                    },
                    tooltip=folium.Tooltip(
                        f"<b>{row['zone_id']}</b><br>Predicted: <b>{sv:.4f}</b>",
                        sticky=False,
                    ),
                ).add_to(mm)
            st_folium(mm, width=None, height=360, returned_objects=[])
        else:
            st.info("No zone geometries available.")

    with mc2:
        shap_title = (
            "What drives the prediction?"
            if not is_technical()
            else "Mean |SHAP| — XGBoost"
        )
        st.markdown(f'<div class="section-title">{shap_title}</div>', unsafe_allow_html=True)
        shap_cols = [c for c in s_zones.columns if c.startswith("shap_")]
        if shap_cols:
            shap_vals = s_zones[shap_cols].mean().abs().sort_values(ascending=True)
            shap_vals.index = shap_vals.index.str.replace("shap_", "", regex=False)
        elif not shap_df.empty:
            feat_cols = [c for c in shap_df.columns if c != "settlement"]
            shap_vals = shap_df[shap_df["settlement"] == selected][feat_cols].mean().abs()
            shap_vals = shap_vals.sort_values(ascending=True)
        else:
            shap_vals = pd.Series(dtype=float)

        if not shap_vals.empty:
            y_labels = [feature_label(i) for i in shap_vals.index]
            colors = [
                "#F5A623" if i >= len(shap_vals) - 3 else "rgba(245,166,35,0.45)"
                for i in range(len(shap_vals))
            ]
            fig_shap = go.Figure(
                go.Bar(
                    x=shap_vals.values,
                    y=y_labels,
                    orientation="h",
                    marker_color=colors,
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                )
            )
            fig_shap.update_layout(**plotly_layout(height=320))
            fig_shap.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig_shap, use_container_width=True)

st.markdown('<div class="section-title">All sites vs KISIP average</div>', unsafe_allow_html=True)

comp = profiles.copy()
comp_col = mean_col
comp["label"] = comp["settlement"].str.replace("_", " ")
comp = comp.sort_values(comp_col, ascending=True)

bar_colors = []
for v in comp[comp_col]:
    tier_name, _ = readiness_tier(v, kisip_baseline)
    bar_colors.append(TIER_COLORS.get(tier_name, "#4FC3A1"))

fig_comp = go.Figure(
    go.Bar(
        y=comp["label"],
        x=comp[comp_col],
        orientation="h",
        marker_color=bar_colors,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    )
)
fig_comp.add_vline(
    x=kisip_baseline,
    line=dict(color="#4FC3A1", width=1.5, dash="dash"),
)
fig_comp.update_layout(
    **plotly_layout(
        height=220,
        xaxis_title="Predicted change" if not is_technical() else comp_col,
    )
)
fig_comp.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
st.plotly_chart(fig_comp, use_container_width=True)
st.caption(
    "Dashed line = KISIP treated average. "
    + ("Dot colour indicates readiness tier." if not is_technical() else "Tier thresholds: ≥75%, ≥45% of baseline.")
)
