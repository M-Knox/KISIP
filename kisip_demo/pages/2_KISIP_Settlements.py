import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="KISIP Settlements", page_icon="🏘️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.page-eyebrow { font-family:'JetBrains Mono',monospace; font-size:0.65rem; letter-spacing:0.18em; color:#4FC3A1; text-transform:uppercase; margin-bottom:0.4rem; }
.page-title { font-size:1.8rem; font-weight:700; color:#E8EAF0; margin-bottom:0.2rem; }
.page-sub { font-size:0.9rem; color:#8B95A8; margin-bottom:1.5rem; }
.info-card { background:#1A1F2E; border:1px solid #252D3D; border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.8rem; }
.card-label { font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:#8B95A8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.2rem; }
.card-value { font-size:1.5rem; font-weight:700; color:#4FC3A1; }
.card-sub { font-size:0.75rem; color:#8B95A8; margin-top:0.1rem; }
.shap-legend { font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#8B95A8; margin-top:0.4rem; }
.feature-tag { display:inline-block; background:rgba(79,195,161,0.1); border:1px solid rgba(79,195,161,0.2); color:#4FC3A1; font-family:'JetBrains Mono',monospace; font-size:0.65rem; padding:2px 7px; border-radius:4px; margin:2px; }
</style>
""", unsafe_allow_html=True)

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    gdf      = gpd.read_file("data/kisip_zones_spatial.geojson").to_crs("EPSG:4326")
    preds    = pd.read_csv("data/kisip_model_predictions.csv")
    shap_set = pd.read_csv("data/kisip_shap_by_settlement.csv")
    features = pd.read_csv("data/kisip_baseline_features_9final.csv")
    scmi     = pd.read_csv("data/kisip_zone_scmi_both.csv")
    return gdf, preds, shap_set, features, scmi

gdf, preds, shap_set, features, scmi = load_data()

SHAP_FEATURES  = ["NDVI","NDBI","MNDWI","Contrast","Entropy","Homogeneity","Correlation","road_density","paved_proportion"]
KISIP_SETTLEMENTS = sorted(gdf["settlement"].unique().tolist())

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-eyebrow">02 / KISIP Treated Settlements</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Zone-Level SCMI & Feature Attribution</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Observed SCMI (Change Vector Analysis) per 50m grid zone. SHAP values show which features drove change magnitude in each settlement.</div>', unsafe_allow_html=True)

selected = st.selectbox(
    "Select settlement",
    KISIP_SETTLEMENTS,
    format_func=lambda x: x.replace("_", " "),
)

s_gdf   = gdf[gdf["settlement"] == selected].copy()
s_preds = preds[preds["settlement"] == selected].copy()
s_shap  = shap_set[shap_set["settlement"] == selected].copy()
s_feats = features[features["settlement"] == selected].copy()
s_scmi  = scmi[scmi["settlement"] == selected].copy()

mean_scmi    = s_gdf["SCMI"].mean()
mean_xgb     = s_gdf["xgb_pred"].mean()
n_zones      = len(s_gdf)
dominant_feat = s_gdf["dominant_feature"].value_counts().idxmax() if "dominant_feature" in s_gdf.columns else "—"

# ── Layout ────────────────────────────────────────────────────────────────────
col_map, col_right = st.columns([3, 2], gap="large")

with col_map:
    st.markdown(f"**Zone SCMI choropleth — {selected.replace('_',' ')}**")

    centroid_lat = s_gdf.geometry.centroid.y.mean()
    centroid_lon = s_gdf.geometry.centroid.x.mean()

    m = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=15, tiles="CartoDB dark_matter")

    scmi_min = s_gdf["SCMI"].min()
    scmi_max = s_gdf["SCMI"].max()
    scmi_range = scmi_max - scmi_min if scmi_max != scmi_min else 1

    def scmi_color(val):
        t = (val - scmi_min) / scmi_range
        r = int(255 * t)
        g = int(195 * (1 - t) + 80 * t)
        b = int(161 * (1 - t))
        return f"#{r:02x}{g:02x}{b:02x}"

    for _, row in s_gdf.iterrows():
        scmi_val = row["SCMI"] if pd.notna(row["SCMI"]) else 0
        xgb_val  = row["xgb_pred"] if pd.notna(row["xgb_pred"]) else 0
        dom      = row.get("dominant_feature", "—")
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda f, c=scmi_color(scmi_val): {
                "fillColor": c, "color": "#0F1117", "weight": 0.5, "fillOpacity": 0.8,
            },
            tooltip=folium.Tooltip(
                f"<b>{row['zone_id']}</b><br>"
                f"SCMI (obs): <b>{scmi_val:.4f}</b><br>"
                f"XGBoost pred: <b>{xgb_val:.4f}</b><br>"
                f"Dominant feature: <b>{dom}</b>",
                sticky=False,
            ),
        ).add_to(m)

    st_folium(m, width=None, height=480, returned_objects=[])

    # SCMI distribution
    st.markdown("**SCMI distribution across zones**")
    hist_fig = px.histogram(
        s_gdf, x="SCMI", nbins=20,
        color_discrete_sequence=["#4FC3A1"],
        labels={"SCMI": "Observed SCMI (CVA)"},
        height=200,
    )
    hist_fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,31,46,0.6)",
        font_color="#C5CAD6",
        bargap=0.05,
        showlegend=False,
    )
    hist_fig.update_xaxes(gridcolor="#252D3D")
    hist_fig.update_yaxes(gridcolor="#252D3D")
    st.plotly_chart(hist_fig, use_container_width=True)

with col_right:
    # ── KPI cards ─────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="info-card">
            <div class="card-label">Mean SCMI (obs)</div>
            <div class="card-value">{mean_scmi:.4f}</div>
            <div class="card-sub">CVA magnitude</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="info-card">
            <div class="card-label">Mean XGBoost pred</div>
            <div class="card-value">{mean_xgb:.4f}</div>
            <div class="card-sub">ensemble output</div>
        </div>""", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"""
        <div class="info-card">
            <div class="card-label">Zones</div>
            <div class="card-value">{n_zones}</div>
            <div class="card-sub">50m grid cells</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="info-card">
            <div class="card-label">Top driver</div>
            <div class="card-value" style="font-size:1rem;margin-top:0.3rem;">{dominant_feat}</div>
            <div class="card-sub">dominant SHAP feature</div>
        </div>""", unsafe_allow_html=True)

    # ── SHAP bar chart ────────────────────────────────────────────────────────
    st.markdown("**Mean SHAP attribution — settlement level**")
    if not s_shap.empty:
        shap_vals = s_shap[SHAP_FEATURES].mean().abs().sort_values(ascending=True)
    else:
        # Fall back to zone-level SHAP cols already in gdf
        shap_cols = [c for c in s_gdf.columns if c.startswith("shap_")]
        shap_vals = s_gdf[shap_cols].mean().abs()
        shap_vals.index = shap_vals.index.str.replace("shap_", "")
        shap_vals = shap_vals.sort_values(ascending=True)

    colors = ["#4FC3A1" if i >= len(shap_vals) - 3 else "#2D8A74"
              for i in range(len(shap_vals))]

    shap_fig = go.Figure(go.Bar(
        x=shap_vals.values,
        y=shap_vals.index,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    shap_fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,31,46,0.6)",
        font_color="#C5CAD6",
        xaxis_title="Mean |SHAP value|",
        xaxis=dict(gridcolor="#252D3D"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(shap_fig, use_container_width=True)
    st.markdown('<div class="shap-legend">Brighter bars = top 3 contributors to SCMI prediction</div>', unsafe_allow_html=True)

    # ── CVA direction summary ─────────────────────────────────────────────────
    if "CVA_Direction" in s_gdf.columns:
        st.markdown("**CVA direction (deg) — zone spread**")
        cva_fig = px.box(
            s_gdf, y="CVA_Direction",
            color_discrete_sequence=["#F5A623"],
            labels={"CVA_Direction": "CVA Direction (°)"},
            height=180,
        )
        cva_fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,31,46,0.6)",
            font_color="#C5CAD6",
            showlegend=False,
        )
        cva_fig.update_yaxes(gridcolor="#252D3D")
        st.plotly_chart(cva_fig, use_container_width=True)

# ── Feature table ─────────────────────────────────────────────────────────────
with st.expander("Zone feature matrix", expanded=False):
    display_cols = ["zone_id"] + [c for c in SHAP_FEATURES if c in s_feats.columns]
    st.dataframe(
        s_feats[display_cols].set_index("zone_id").round(5),
        use_container_width=True,
        height=280,
    )

# ── Obs vs Pred scatter ───────────────────────────────────────────────────────
if not s_preds.empty and "SCMI" in s_preds.columns and "xgb_pred" in s_preds.columns:
    with st.expander("Observed vs predicted SCMI scatter", expanded=False):
        fig_ov = px.scatter(
            s_preds, x="SCMI", y="xgb_pred",
            hover_data=["zone_id"],
            color_discrete_sequence=["#4FC3A1"],
            labels={"SCMI": "Observed SCMI", "xgb_pred": "XGBoost Prediction"},
            height=320,
        )
        fig_ov.add_shape(type="line",
            x0=s_preds["SCMI"].min(), x1=s_preds["SCMI"].max(),
            y0=s_preds["SCMI"].min(), y1=s_preds["SCMI"].max(),
            line=dict(color="#F5A623", width=1.5, dash="dash"))
        fig_ov.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,31,46,0.6)",
            font_color="#C5CAD6",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        fig_ov.update_xaxes(gridcolor="#252D3D")
        fig_ov.update_yaxes(gridcolor="#252D3D")
        st.plotly_chart(fig_ov, use_container_width=True)
        st.caption("Dashed line = perfect prediction. Points above = over-prediction; below = under-prediction.")
