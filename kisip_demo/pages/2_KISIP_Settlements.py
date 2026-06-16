import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from styles import inject_css

st.set_page_config(page_title="KISIP Settlements", page_icon="🏘️", layout="wide")
inject_css()

@st.cache_data
def load_data():
    gdf      = gpd.read_file("data/kisip_zones_spatial.geojson").to_crs("EPSG:4326")
    preds    = pd.read_csv("data/kisip_model_predictions.csv")
    features = pd.read_csv("data/kisip_baseline_features_9final.csv")
    return gdf, preds, features

gdf, preds, features = load_data()

KISIP_SETTLEMENTS = sorted(gdf["settlement"].unique().tolist())
SHAP_COLS = [c for c in gdf.columns if c.startswith("shap_")]
FEAT_COLS = ["NDVI","NDBI","MNDWI","Contrast","Entropy","Homogeneity",
             "Correlation","road_density","paved_proportion"]

# Plain-language interpretation per dominant feature
SHAP_INTERPRETATIONS = {
    "NDVI":             "Vegetation cover (NDVI) was the primary precondition — areas with more pre-intervention greenery showed stronger upgrading response, likely reflecting open land availability.",
    "NDBI":             "Built-up surface density (NDBI) was the primary driver — pre-intervention built-up intensity was the strongest precondition of physical change magnitude.",
    "MNDWI":            "Water surface index (MNDWI) dominated — proximity to water bodies or drainage channels influenced the degree of surface change detected.",
    "Contrast":         "Texture contrast was the primary driver — high spatial variation in pre-intervention surface materials predicted greater subsequent change.",
    "Entropy":          "Surface texture entropy was dominant — zones with more heterogeneous pre-intervention surfaces experienced greater measured change.",
    "Homogeneity":      "Texture homogeneity was the leading feature — uniformly surfaced pre-intervention zones responded more consistently to upgrading.",
    "Correlation":      "Texture correlation dominated — spatial co-occurrence patterns in pre-intervention imagery were the strongest change predictor.",
    "road_density":     "Road network density was the primary driver — zones with denser pre-intervention road access showed greater surface change post-intervention.",
    "paved_proportion": "Paved road proportion was dominant — pre-existing paved access correlated most strongly with the magnitude of surface upgrading.",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="pg-label">02 / KISIP Treated Settlements</div>', unsafe_allow_html=True)
st.markdown('<div class="pg-title">Zone-Level SCMI & Feature Attribution</div>', unsafe_allow_html=True)
st.markdown('<div class="pg-sub">Observed SCMI (Change Vector Analysis) per 50 m grid zone. SHAP values show which pre-intervention features drove the magnitude of surface change detected between the pre- and post-intervention epochs.</div>', unsafe_allow_html=True)

selected = st.selectbox(
    "Settlement",
    KISIP_SETTLEMENTS,
    format_func=lambda x: x.replace("_", " "),
    label_visibility="collapsed",
)

s_gdf   = gdf[gdf["settlement"] == selected].copy()
s_preds = preds[preds["settlement"] == selected].copy()
s_feats = features[features["settlement"] == selected].copy()

mean_scmi = s_gdf["SCMI"].mean()
mean_xgb  = s_gdf["xgb_pred"].mean()
n_zones   = len(s_gdf)

# Clean dominant_feature — strip shap_ prefix if present
if "dominant_feature" in s_gdf.columns:
    dom_feat = (s_gdf["dominant_feature"]
                .str.replace("shap_", "", regex=False)
                .value_counts().idxmax())
else:
    dom_feat = "—"

# ── Stat row ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stat-row">
  <div class="stat-cell">
    <div class="sc-val">{mean_scmi:.4f}</div>
    <div class="sc-label">Mean observed SCMI (CVA)</div>
  </div>
  <div class="stat-cell">
    <div class="sc-val">{mean_xgb:.4f}</div>
    <div class="sc-label">Mean XGBoost prediction</div>
  </div>
  <div class="stat-cell">
    <div class="sc-val">{n_zones}</div>
    <div class="sc-label">50 m grid zones</div>
  </div>
  <div class="stat-cell">
    <div class="sc-val" style="font-size:1.1rem;padding-top:0.3rem;">{dom_feat}</div>
    <div class="sc-label">Most frequent dominant feature</div>
  </div>
</div>
""", unsafe_allow_html=True)

col_map, col_right = st.columns([3, 2], gap="large")

with col_map:
    st.markdown('<div class="section-rule">SCMI choropleth</div>', unsafe_allow_html=True)

    centroid_lat = s_gdf.geometry.centroid.y.mean()
    centroid_lon = s_gdf.geometry.centroid.x.mean()
    m = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=15, tiles="CartoDB dark_matter")

    scmi_min   = s_gdf["SCMI"].min()
    scmi_range = s_gdf["SCMI"].max() - scmi_min or 1

    def scmi_color(val):
        t = (val - scmi_min) / scmi_range
        return f"#{int(255*t):02x}{int(195*(1-t)+80*t):02x}{int(161*(1-t)):02x}"

    for _, row in s_gdf.iterrows():
        sv  = row["SCMI"]     if pd.notna(row["SCMI"])     else 0
        xv  = row["xgb_pred"] if pd.notna(row["xgb_pred"]) else 0
        dom = str(row.get("dominant_feature", "—")).replace("shap_", "")
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda f, c=scmi_color(sv): {
                "fillColor": c, "color": "#0F1117", "weight": 0.5, "fillOpacity": 0.85,
            },
            tooltip=folium.Tooltip(
                f"<b>{row['zone_id']}</b><br>"
                f"SCMI {sv:.4f} &nbsp;|&nbsp; XGB {xv:.4f}<br>"
                f"Dominant feature: {dom}",
                sticky=False,
            ),
        ).add_to(m)

    st_folium(m, width=None, height=460, returned_objects=[])

    st.markdown('<div class="section-rule" style="margin-top:1rem;">SCMI distribution</div>', unsafe_allow_html=True)
    hist = px.histogram(
        s_gdf, x="SCMI", nbins=20,
        color_discrete_sequence=["#4FC3A1"],
        labels={"SCMI": "Observed SCMI (CVA)"},
        height=180,
    )
    hist.update_layout(
        margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(28,35,51,0.5)",
        font_color="#8B95A8",
        bargap=0.05, showlegend=False,
    )
    hist.update_xaxes(gridcolor="#252D3D", title_font_size=11)
    hist.update_yaxes(gridcolor="#252D3D")
    st.plotly_chart(hist, use_container_width=True)

with col_right:
    st.markdown('<div class="section-rule">SHAP feature attribution</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#8B95A8;margin-bottom:0.6rem;">Mean absolute SHAP value across all zones — shows which pre-intervention features most drove change magnitude.</div>', unsafe_allow_html=True)

    shap_vals = s_gdf[SHAP_COLS].mean().abs().sort_values(ascending=True)
    shap_vals.index = shap_vals.index.str.replace("shap_", "", regex=False)
    top_feature = shap_vals.idxmax()

    # Highlight only the single dominant feature
    bar_colors = ["#4FC3A1" if feat == top_feature else "#2A5A4A"
                  for feat in shap_vals.index]

    shap_fig = go.Figure(go.Bar(
        x=shap_vals.values,
        y=shap_vals.index,
        orientation="h",
        marker_color=bar_colors,
        hovertemplate="%{y}: %{x:.5f}<extra></extra>",
    ))
    shap_fig.update_layout(
        height=310,
        margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(28,35,51,0.5)",
        font_color="#8B95A8",
        xaxis=dict(gridcolor="#252D3D", title="Mean |SHAP|", title_font_size=11),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont_size=11),
    )
    st.plotly_chart(shap_fig, use_container_width=True)

    # Auto-generated interpretation sentence
    interpretation = SHAP_INTERPRETATIONS.get(
        top_feature,
        f"{top_feature} was the primary driver of SCMI in this settlement."
    )
    st.markdown(f'<div class="finding" style="margin-top:0rem;">{interpretation}</div>',
                unsafe_allow_html=True)

    if "CVA_Direction" in s_gdf.columns:
        st.markdown('<div class="section-rule">CVA direction spread</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.78rem;color:#8B95A8;margin-bottom:0.6rem;">Direction of surface change (degrees) across zones. Negative = spectral decrease; positive = increase.</div>', unsafe_allow_html=True)
        cva_fig = px.box(
            s_gdf, y="CVA_Direction",
            color_discrete_sequence=["#F5A623"],
            labels={"CVA_Direction": "CVA Direction (°)"},
            height=200,
        )
        cva_fig.update_layout(
            margin=dict(l=0, r=0, t=4, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(28,35,51,0.5)",
            font_color="#8B95A8",
            showlegend=False,
        )
        cva_fig.update_yaxes(gridcolor="#252D3D")
        st.plotly_chart(cva_fig, use_container_width=True)

# ── Expandable details ────────────────────────────────────────────────────────
with st.expander("Zone feature matrix", expanded=False):
    display_cols = ["zone_id"] + [c for c in FEAT_COLS if c in s_feats.columns]
    st.dataframe(s_feats[display_cols].set_index("zone_id").round(5),
                 use_container_width=True, height=260)

if not s_preds.empty and "SCMI" in s_preds.columns and "xgb_pred" in s_preds.columns:
    with st.expander("Observed vs predicted scatter", expanded=False):
        fig_sc = px.scatter(
            s_preds, x="SCMI", y="xgb_pred",
            hover_data=["zone_id"],
            color_discrete_sequence=["#4FC3A1"],
            labels={"SCMI": "Observed SCMI", "xgb_pred": "XGBoost prediction"},
            height=320,
        )
        fig_sc.add_shape(type="line",
            x0=s_preds["SCMI"].min(), x1=s_preds["SCMI"].max(),
            y0=s_preds["SCMI"].min(), y1=s_preds["SCMI"].max(),
            line=dict(color="#F5A623", width=1.5, dash="dash"))
        fig_sc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(28,35,51,0.5)",
            font_color="#8B95A8",
            margin=dict(l=0, r=0, t=4, b=0),
        )
        fig_sc.update_xaxes(gridcolor="#252D3D")
        fig_sc.update_yaxes(gridcolor="#252D3D")
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption("Dashed line = perfect prediction. Points above = over-prediction.")
