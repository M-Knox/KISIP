import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Mukuru Readiness", page_icon="📍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.page-eyebrow { font-family:'JetBrains Mono',monospace; font-size:0.65rem; letter-spacing:0.18em; color:#F5A623; text-transform:uppercase; margin-bottom:0.4rem; }
.page-title { font-size:1.8rem; font-weight:700; color:#E8EAF0; margin-bottom:0.2rem; }
.page-sub { font-size:0.9rem; color:#8B95A8; margin-bottom:1.5rem; }
.tier-badge {
    display: inline-block;
    border-radius: 8px;
    padding: 0.6rem 1.4rem;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-bottom: 1rem;
}
.tier-high   { background:rgba(79,195,161,0.15); border:2px solid #4FC3A1; color:#4FC3A1; }
.tier-medium { background:rgba(245,166,35,0.15);  border:2px solid #F5A623; color:#F5A623; }
.tier-low    { background:rgba(231,76,60,0.15);   border:2px solid #E74C3C; color:#E74C3C; }
.info-card { background:#1A1F2E; border:1px solid #252D3D; border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.8rem; }
.card-label { font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:#8B95A8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.2rem; }
.card-value { font-size:1.5rem; font-weight:700; }
.card-sub { font-size:0.75rem; color:#8B95A8; margin-top:0.1rem; }
.rank-row { display:flex; align-items:center; gap:0.8rem; padding:0.5rem 0; border-bottom:1px solid #252D3D; }
.rank-row:last-child { border-bottom:none; }
.rank-num { font-family:'JetBrains Mono',monospace; font-size:0.8rem; font-weight:700; width:24px; color:#F5A623; }
.rank-name { font-size:0.9rem; color:#C5CAD6; flex:1; }
.rank-score { font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#4FC3A1; }
</style>
""", unsafe_allow_html=True)

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    profiles = pd.read_csv("data/kisip_mukuru_readiness_profiles.csv")
    preds    = pd.read_csv("data/kisip_mukuru_predictions.csv")
    shap_df  = pd.read_csv("data/kisip_mukuru_shap_attribution.csv")
    mukuru_z = gpd.read_file("data/mukuru_zones_spatial.geojson").to_crs("EPSG:4326")
    kisip_z  = gpd.read_file("data/kisip_zones_spatial.geojson").to_crs("EPSG:4326")
    return profiles, preds, shap_df, mukuru_z, kisip_z

profiles, preds, shap_df, mukuru_z, kisip_z = load_data()

SHAP_FEATURES = ["NDVI","NDBI","MNDWI","Contrast","Entropy","Homogeneity","Correlation","road_density","paved_proportion"]
MUKURU_SETTLEMENTS = sorted(profiles["settlement"].tolist())

kisip_baseline_scmi = kisip_z["SCMI"].mean()

def readiness_tier(score, baseline):
    ratio = score / baseline if baseline > 0 else 0
    if ratio >= 0.75:   return "High Readiness",   "tier-high",   "🟢"
    elif ratio >= 0.45: return "Moderate Readiness","tier-medium", "🟡"
    else:               return "Low Readiness",     "tier-low",    "🔴"

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-eyebrow">03 / Mukuru Readiness Profiles</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Predicted SCMI & Intervention Readiness</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Ensemble model (Ridge + RF + XGBoost) applied to Mukuru sites. Readiness tier benchmarks predicted SCMI against the KISIP treated settlement baseline.</div>', unsafe_allow_html=True)

col_sel, col_rank = st.columns([3, 1])

with col_rank:
    st.markdown("**Readiness ranking**")
    prof_sorted = profiles.sort_values("readiness_rank")
    rank_html = ""
    for _, r in prof_sorted.iterrows():
        _, cls, emoji = readiness_tier(r["mean_ensemble_scmi"], kisip_baseline_scmi)
        rank_html += f"""
        <div class="rank-row">
            <div class="rank-num">#{int(r['readiness_rank'])}</div>
            <div class="rank-name">{r['settlement'].replace('_',' ')}</div>
            <div class="rank-score">{r['mean_ensemble_scmi']:.3f}</div>
        </div>"""
    st.markdown(f'<div class="info-card">{rank_html}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-card" style="margin-top:0.5rem">
        <div class="card-label">KISIP baseline (mean SCMI)</div>
        <div class="card-value" style="color:#4FC3A1">{kisip_baseline_scmi:.4f}</div>
        <div class="card-sub">benchmark for tier classification</div>
    </div>""", unsafe_allow_html=True)

with col_sel:
    selected = st.selectbox(
        "Select Mukuru settlement",
        MUKURU_SETTLEMENTS,
        format_func=lambda x: x.replace("_", " "),
    )

    s_profile = profiles[profiles["settlement"] == selected].iloc[0]
    s_preds   = preds[preds["settlement"] == selected].copy()
    s_shap    = shap_df[shap_df["settlement"] == selected].copy() if "settlement" in shap_df.columns else pd.DataFrame()
    s_zones = mukuru_z[mukuru_z["settlement"] == selected].copy()
    s_zones = s_zones.merge(s_preds[["zone_id","ensemble_scmi"]], on="zone_id", how="left")
    mean_pred = s_profile["mean_ensemble_scmi"]
    tier_label, tier_cls, tier_emoji = readiness_tier(mean_pred, kisip_baseline_scmi)

    st.markdown(f'<div class="tier-badge {tier_cls}">{tier_emoji} {tier_label}</div>', unsafe_allow_html=True)

    kc1, kc2, kc3 = st.columns(3)
    with kc1:
        st.markdown(f"""
        <div class="info-card">
            <div class="card-label">Mean pred. SCMI</div>
            <div class="card-value" style="color:#F5A623">{mean_pred:.4f}</div>
            <div class="card-sub">ensemble (Ridge+RF+XGB)</div>
        </div>""", unsafe_allow_html=True)
    with kc2:
        st.markdown(f"""
        <div class="info-card">
            <div class="card-label">Zones analysed</div>
            <div class="card-value" style="color:#F5A623">{int(s_profile['zones'])}</div>
            <div class="card-sub">50m grid cells</div>
        </div>""", unsafe_allow_html=True)
    with kc3:
        gap = kisip_baseline_scmi - mean_pred
        st.markdown(f"""
        <div class="info-card">
            <div class="card-label">Gap to baseline</div>
            <div class="card-value" style="color:{'#E74C3C' if gap>0 else '#4FC3A1'}">{"−" if gap>0 else "+"}{abs(gap):.4f}</div>
            <div class="card-sub">vs KISIP mean SCMI</div>
        </div>""", unsafe_allow_html=True)

    # ── Map + SHAP side by side ───────────────────────────────────────────────
    mc1, mc2 = st.columns([3, 2])

    with mc1:
        st.markdown(f"**Predicted SCMI map — {selected.replace('_',' ')}**")
        if not s_zones.empty:
            clat = s_zones.geometry.centroid.y.mean()
            clon = s_zones.geometry.centroid.x.mean()
            mm = folium.Map(location=[clat, clon], zoom_start=15, tiles="CartoDB dark_matter")

            scmi_min = s_zones["ensemble_scmi"].min()
            scmi_max = s_zones["ensemble_scmi"].max()
            scmi_rng = scmi_max - scmi_min if scmi_max != scmi_min else 1

            for _, row in s_zones.iterrows():
                sv = row["ensemble_scmi"] if pd.notna(row["ensemble_scmi"]) else 0
                t  = (sv - scmi_min) / scmi_rng
                r  = int(245 * t + 79 * (1 - t))
                g  = int(166 * t + 195 * (1 - t))
                b  = int(35 * t + 161 * (1 - t))
                color = f"#{r:02x}{g:02x}{b:02x}"
                folium.GeoJson(
                    row["geometry"].__geo_interface__,
                    style_function=lambda f, c=color: {
                        "fillColor": c, "color": "#0F1117", "weight": 0.5, "fillOpacity": 0.8
                    },
                    tooltip=folium.Tooltip(
                        f"<b>{row['zone_id']}</b><br>Pred. SCMI: <b>{sv:.4f}</b>",
                        sticky=False,
                    ),
                ).add_to(mm)
            st_folium(mm, width=None, height=380, returned_objects=[])
        else:
            st.info("No zone geometries available for this settlement.")

    with mc2:
        st.markdown("**SHAP attribution — settlement mean**")
        if not s_shap.empty:
            feat_cols = [c for c in SHAP_FEATURES if c in s_shap.columns]
            shap_vals = s_shap[feat_cols].mean().abs().sort_values(ascending=True)
        else:
            # Use settlement-level shap from mukuru_shap_attribution (no zone_id col)
            row_shap = shap_df[shap_df["settlement"] == selected]
            if not row_shap.empty:
                feat_cols = [c for c in SHAP_FEATURES if c in row_shap.columns]
                shap_vals = row_shap[feat_cols].mean().abs().sort_values(ascending=True)
            else:
                shap_vals = pd.Series(dtype=float)

        if not shap_vals.empty:
            colors = ["#F5A623" if i >= len(shap_vals) - 3 else "#8B5E14"
                      for i in range(len(shap_vals))]
            fig_shap = go.Figure(go.Bar(
                x=shap_vals.values,
                y=shap_vals.index,
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig_shap.update_layout(
                height=360,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,31,46,0.6)",
                font_color="#C5CAD6",
                xaxis_title="Mean |SHAP|",
                xaxis=dict(gridcolor="#252D3D"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

# ── Comparison bar — all settlements vs KISIP baseline ───────────────────────
st.markdown("---")
st.markdown("**All Mukuru settlements vs KISIP treated baseline**")

comp_data = profiles[["settlement","mean_ensemble_scmi"]].copy()
comp_data["label"] = comp_data["settlement"].str.replace("_", " ")
comp_data = comp_data.sort_values("mean_ensemble_scmi", ascending=True)

def bar_color(v, baseline):
    ratio = v / baseline if baseline > 0 else 0
    if ratio >= 0.75:   return "#4FC3A1"
    elif ratio >= 0.45: return "#F5A623"
    else:               return "#E74C3C"

bar_colors = [bar_color(v, kisip_baseline_scmi) for v in comp_data["mean_ensemble_scmi"]]

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(
    y=comp_data["label"],
    x=comp_data["mean_ensemble_scmi"],
    orientation="h",
    marker_color=bar_colors,
    name="Predicted SCMI",
    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
))
fig_comp.add_vline(
    x=kisip_baseline_scmi,
    line=dict(color="#4FC3A1", width=2, dash="dash"),
    annotation_text=f"KISIP baseline {kisip_baseline_scmi:.3f}",
    annotation_position="top right",
    annotation_font_color="#4FC3A1",
)
fig_comp.update_layout(
    height=240,
    margin=dict(l=0, r=0, t=20, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,31,46,0.6)",
    font_color="#C5CAD6",
    xaxis=dict(gridcolor="#252D3D", title="Mean ensemble SCMI"),
    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    showlegend=False,
)
st.plotly_chart(fig_comp, use_container_width=True)
st.caption("Dashed line = KISIP treated mean SCMI. Colour indicates readiness tier: 🟢 high · 🟡 moderate · 🔴 low")
