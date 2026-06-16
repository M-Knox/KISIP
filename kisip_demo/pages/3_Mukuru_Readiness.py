import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from styles import inject_css

st.set_page_config(page_title="Mukuru Readiness", page_icon="📍", layout="wide")
inject_css()

@st.cache_data
def load_data():
    profiles = pd.read_csv("data/kisip_mukuru_readiness_profiles.csv")
    shap_df  = pd.read_csv("data/kisip_mukuru_shap_attribution.csv")
    mukuru_z = gpd.read_file("data/mukuru_zones_spatial.geojson").to_crs("EPSG:4326")
    kisip_z  = gpd.read_file("data/kisip_zones_spatial.geojson").to_crs("EPSG:4326")
    return profiles, shap_df, mukuru_z, kisip_z

profiles, shap_df, mukuru_z, kisip_z = load_data()

MUKURU_SETTLEMENTS  = sorted(profiles["settlement"].tolist())
kisip_baseline_scmi = kisip_z["SCMI"].mean()

TIER_COLORS = {"High": "#4FC3A1", "Moderate": "#F5A623", "Low": "#E05A4B"}

def readiness_tier(score, baseline):
    ratio = score / baseline if baseline > 0 else 0
    if ratio >= 0.75:   return "High",     "High Readiness"
    elif ratio >= 0.45: return "Moderate", "Moderate Readiness"
    else:               return "Low",      "Low Readiness"

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="pg-label">03 / Mukuru Readiness Profiles</div>', unsafe_allow_html=True)
st.markdown('<div class="pg-title">Predicted SCMI & Intervention Readiness</div>', unsafe_allow_html=True)
st.markdown('<div class="pg-sub">Ensemble model (Ridge + Random Forest + XGBoost) applied to four untreated Mukuru sites. Readiness tier is determined by benchmarking predicted SCMI against the mean of the five KISIP treated settlements.</div>', unsafe_allow_html=True)

col_sel, col_rank = st.columns([3, 1], gap="large")

with col_rank:
    st.markdown('<div class="section-rule">Readiness ranking</div>', unsafe_allow_html=True)
    prof_sorted = profiles.sort_values("readiness_rank")
    rank_html = ""
    for _, r in prof_sorted.iterrows():
        tier_key, _ = readiness_tier(r["mean_ensemble_scmi"], kisip_baseline_scmi)
        dot_color   = TIER_COLORS[tier_key]
        rank_html += f"""
        <div class="rank-item">
          <div class="rank-n">#{int(r['readiness_rank'])}</div>
          <div class="rank-name">{r['settlement'].replace('_',' ')}</div>
          <div class="rank-score">{r['mean_ensemble_scmi']:.3f}</div>
        </div>"""
    st.markdown(rank_html, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:1.25rem;padding-top:0.75rem;border-top:1px solid #252D3D;">
      <div style="font-size:0.7rem;color:#8B95A8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.2rem;font-family:'JetBrains Mono',monospace;">KISIP baseline</div>
      <div style="font-size:1.4rem;font-weight:700;color:#4FC3A1;">{kisip_baseline_scmi:.4f}</div>
      <div style="font-size:0.75rem;color:#8B95A8;">mean observed SCMI</div>
    </div>
    <div style="margin-top:1rem;font-size:0.78rem;color:#8B95A8;line-height:1.6;">
      <b style="color:#C5CAD6;">Tier thresholds</b><br>
      <span style="color:#4FC3A1;">● High</span> — ≥ 75 % of baseline<br>
      <span style="color:#F5A623;">● Moderate</span> — 45–75 %<br>
      <span style="color:#E05A4B;">● Low</span> — &lt; 45 %
    </div>
    """, unsafe_allow_html=True)

with col_sel:
    selected = st.selectbox(
        "Settlement",
        MUKURU_SETTLEMENTS,
        format_func=lambda x: x.replace("_", " "),
        label_visibility="collapsed",
    )

    s_profile = profiles[profiles["settlement"] == selected].iloc[0]
    s_zones   = mukuru_z[mukuru_z["settlement"] == selected].copy()
    s_shap    = shap_df[shap_df["settlement"] == selected].copy()

    mean_pred           = s_profile["mean_ensemble_scmi"]
    tier_key, tier_label = readiness_tier(mean_pred, kisip_baseline_scmi)
    tier_color          = TIER_COLORS[tier_key]
    gap                 = kisip_baseline_scmi - mean_pred

    # Tier — dot + text, no box
    st.markdown(f"""
    <div class="tier-line">
      <div class="tier-dot" style="background:{tier_color}"></div>
      <div class="tier-text">{tier_label}</div>
    </div>
    <div class="tier-sub">{selected.replace('_',' ')} · Readiness rank #{int(s_profile['readiness_rank'])} of 4</div>
    """, unsafe_allow_html=True)

    # Stat row
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-cell">
        <div class="sc-val" style="color:{tier_color}">{mean_pred:.4f}</div>
        <div class="sc-label">Mean predicted SCMI</div>
      </div>
      <div class="stat-cell">
        <div class="sc-val">{int(s_profile['zones'])}</div>
        <div class="sc-label">50 m grid zones</div>
      </div>
      <div class="stat-cell">
        <div class="sc-val" style="color:{'#E05A4B' if gap>0 else '#4FC3A1'}">{"−" if gap>0 else "+"}{abs(gap):.4f}</div>
        <div class="sc-label">Gap vs KISIP baseline</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    mc1, mc2 = st.columns([3, 2], gap="large")

    with mc1:
        st.markdown('<div class="section-rule">Predicted SCMI map</div>', unsafe_allow_html=True)
        if not s_zones.empty:
            centroid = s_zones.to_crs("EPSG:32737").geometry.centroid.to_crs("EPSG:4326")
            mm = folium.Map(location=[centroid.y.mean(), centroid.x.mean()],
                            zoom_start=15, tiles="CartoDB dark_matter")

            scmi_min = s_zones["ensemble_scmi"].min()
            scmi_rng = s_zones["ensemble_scmi"].max() - scmi_min or 1

            for _, row in s_zones.iterrows():
                sv = row["ensemble_scmi"] if pd.notna(row["ensemble_scmi"]) else 0
                t  = (sv - scmi_min) / scmi_rng
                color = f"#{int(245*t+79*(1-t)):02x}{int(166*t+195*(1-t)):02x}{int(35*t+161*(1-t)):02x}"
                folium.GeoJson(
                    row["geometry"].__geo_interface__,
                    style_function=lambda f, c=color: {
                        "fillColor": c, "color": "#0F1117", "weight": 0.5, "fillOpacity": 0.85
                    },
                    tooltip=folium.Tooltip(
                        f"<b>{row['zone_id']}</b><br>Pred. SCMI {sv:.4f}", sticky=False),
                ).add_to(mm)
            st_folium(mm, width=None, height=360, returned_objects=[])
        else:
            st.info("No zone geometries found for this settlement.")

    with mc2:
        st.markdown('<div class="section-rule">SHAP attribution</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.78rem;color:#8B95A8;margin-bottom:0.6rem;">Which pre-intervention features most influenced predicted change at this site.</div>', unsafe_allow_html=True)

        shap_cols = [c for c in s_zones.columns if c.startswith("shap_")]
        if shap_cols:
            shap_vals = s_zones[shap_cols].mean().abs().sort_values(ascending=True)
            shap_vals.index = shap_vals.index.str.replace("shap_", "", regex=False)
        elif not s_shap.empty:
            feat_cols = [c for c in s_shap.columns if c != "settlement"]
            shap_vals = s_shap[feat_cols].mean().abs().sort_values(ascending=True)
        else:
            shap_vals = pd.Series(dtype=float)

        if not shap_vals.empty:
            bar_colors = [tier_color if i >= len(shap_vals) - 3 else "#2A3A4A"
                          for i in range(len(shap_vals))]
            fig_shap = go.Figure(go.Bar(
                x=shap_vals.values, y=shap_vals.index,
                orientation="h", marker_color=bar_colors,
                hovertemplate="%{y}: %{x:.5f}<extra></extra>",
            ))
            fig_shap.update_layout(
                height=340,
                margin=dict(l=0, r=0, t=4, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(28,35,51,0.5)",
                font_color="#8B95A8",
                xaxis=dict(gridcolor="#252D3D", title="Mean |SHAP|", title_font_size=11),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont_size=11),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

# ── All settlements comparison ────────────────────────────────────────────────
st.markdown('<div class="section-rule" style="margin-top:0.5rem;">All Mukuru sites vs KISIP baseline</div>', unsafe_allow_html=True)

comp = profiles[["settlement","mean_ensemble_scmi"]].copy()
comp["label"] = comp["settlement"].str.replace("_", " ")
comp = comp.sort_values("mean_ensemble_scmi", ascending=True)

def bar_color(v):
    tier_key, _ = readiness_tier(v, kisip_baseline_scmi)
    return TIER_COLORS[tier_key]

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(
    y=comp["label"], x=comp["mean_ensemble_scmi"],
    orientation="h",
    marker_color=[bar_color(v) for v in comp["mean_ensemble_scmi"]],
    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
))
fig_comp.add_vline(
    x=kisip_baseline_scmi,
    line=dict(color="#4FC3A1", width=1.5, dash="dash"),
    annotation_text=f"KISIP baseline {kisip_baseline_scmi:.3f}",
    annotation_position="top right",
    annotation_font_color="#4FC3A1",
    annotation_font_size=11,
)
fig_comp.update_layout(
    height=200,
    margin=dict(l=0, r=0, t=24, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(28,35,51,0.5)",
    font_color="#8B95A8",
    xaxis=dict(gridcolor="#252D3D", title="Mean ensemble SCMI", title_font_size=11),
    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    showlegend=False,
)
st.plotly_chart(fig_comp, use_container_width=True)
st.caption("Colour indicates readiness tier — green: high · amber: moderate · red: low")
