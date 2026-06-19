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

# ── Extra CSS specific to this page ──────────────────────────────────────────
st.markdown("""
<style>
/* Score card grid */
.scorecard-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.scorecard {
    background: #1C2333;
    border-radius: 6px;
    padding: 0.9rem 1rem;
}
.sc-feature {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #8B95A8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.2rem;
}
.sc-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #E2E5EC;
    letter-spacing: -0.02em;
    line-height: 1;
}
.sc-vs-kisip {
    font-size: 0.72rem;
    margin-top: 0.25rem;
}
.sc-above { color: #4FC3A1; }
.sc-below { color: #8B95A8; }
.sc-interp {
    font-size: 0.78rem;
    color: #C5CAD6;
    line-height: 1.55;
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid #252D3D;
}

/* Planning signal strip */
.signal-block {
    background: #1C2333;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.signal-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #E2E5EC;
    margin-bottom: 0.3rem;
}
.signal-body {
    font-size: 0.8rem;
    color: #8B95A8;
    line-height: 1.6;
}
.signal-body b { color: #C5CAD6; }

/* Baseline comparison — neutral, no red/green tier */
.baseline-pill {
    display: inline-block;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 2px 8px;
    border-radius: 3px;
    background: rgba(255,255,255,0.06);
    color: #8B95A8;
    margin-left: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    profiles = pd.read_csv("data/kisip_mukuru_readiness_profiles.csv")
    preds    = pd.read_csv("data/kisip_mukuru_predictions.csv")
    mukuru_z = gpd.read_file("data/mukuru_zones_spatial.geojson").to_crs("EPSG:4326")
    kisip_z  = gpd.read_file("data/kisip_zones_spatial.geojson").to_crs("EPSG:4326")
    kisip_f  = pd.read_csv("data/kisip_baseline_features_9final.csv")
    return profiles, preds, mukuru_z, kisip_z, kisip_f

profiles, preds, mukuru_z, kisip_z, kisip_f = load_data()

MUKURU_SETTLEMENTS  = sorted(profiles["settlement"].tolist())
kisip_baseline_scmi = kisip_z["SCMI"].mean()

# KISIP feature means for benchmarking scorecards
kisip_feat_means = kisip_f[["NDVI","NDBI","MNDWI","Contrast","Entropy",
                              "Homogeneity","Correlation",
                              "road_density","paved_proportion"]].mean()

# ── Planning interpretations per feature ─────────────────────────────────────
def feature_signal(feat, val, kisip_mean):
    """Return (short label, planning interpretation) for a given feature value."""
    above = val >= kisip_mean
    rel   = "above" if above else "below"

    signals = {
        "NDBI": (
            "Built-up surface density",
            f"Value {val:.3f} is {rel} the KISIP treated mean ({kisip_mean:.3f}). "
            + ("A densely consolidated built-up fabric. Road rehabilitation will involve "
               "significant existing structure negotiation rather than standard greenfield "
               "construction. Expect complex alignment decisions on site."
               if above else
               "Lower built-up consolidation than the KISIP baseline. Upgrading works "
               "are likely to involve more open space negotiation but fewer existing "
               "structural conflicts. Surveying burden for right-of-way is reduced.")
        ),
        "NDVI": (
            "Vegetation / green cover",
            f"Value {val:.3f} is {rel} the KISIP treated mean ({kisip_mean:.3f}). "
            + ("Above-average vegetation cover signals available open land or "
               "informal green space — likely easier clearance for drainage channels "
               "and road alignments, but ecological sensitivity should be assessed."
               if above else
               "Sparse vegetation indicates a largely consolidated surface. "
               "Vegetation-related clearance is unlikely to be a major surveying task.")
        ),
        "MNDWI": (
            "Water surface presence",
            f"Value {val:.3f} is {rel} the KISIP treated mean ({kisip_mean:.3f}). "
            + ("Elevated water index — drainage infrastructure should be a primary "
               "scoping concern. Flood risk mapping and stormwater routing should be "
               "conducted before detailed road design."
               if above else
               "Low water presence relative to baseline. Standard drainage scoping "
               "applies; no elevated flood risk signal from spectral data alone.")
        ),
        "Contrast": (
            "Surface texture contrast (GLCM)",
            f"Value {val:.0f} is {rel} the KISIP treated mean ({kisip_mean:.0f}). "
            + ("High surface heterogeneity detected. Drainage and road alignments will "
               "vary considerably within the settlement rather than following a uniform "
               "pattern. Expect higher within-settlement variability in scoping effort."
               if above else
               "Relatively uniform surface conditions. Road and drainage alignments "
               "are likely more consistent across zones, reducing scoping complexity.")
        ),
        "Entropy": (
            "Surface texture entropy (GLCM)",
            f"Value {val:.3f} is {rel} the KISIP treated mean ({kisip_mean:.3f}). "
            + ("High surface disorder — irregular material composition across zones. "
               "Material surveys will need to account for significant variation in "
               "existing surface types."
               if above else
               "More ordered surface composition than baseline. Material conditions "
               "are likely more consistent and predictable across zones.")
        ),
        "Homogeneity": (
            "Surface texture homogeneity (GLCM)",
            f"Value {val:.3f} is {rel} the KISIP treated mean ({kisip_mean:.3f}). "
            + ("High homogeneity indicates spatially consistent surface materials. "
               "Uniform conditions can reduce per-zone scoping time."
               if above else
               "Lower homogeneity than baseline — surface materials vary considerably. "
               "Site surveys should account for within-zone variability in material type.")
        ),
        "Correlation": (
            "Texture spatial correlation (GLCM)",
            f"Value {val:.3f} is {rel} the KISIP treated mean ({kisip_mean:.3f}). "
            + ("Strong spatial correlation in surface patterns — existing structures "
               "follow more regular spatial arrangements, potentially simplifying "
               "layout planning."
               if above else
               "Lower spatial regularity than baseline. Settlement layout is less "
               "ordered, which typically increases the complexity of route alignment.")
        ),
        "road_density": (
            "Road network density",
            f"Value {val:.0f} m/km² is {rel} the KISIP treated mean ({kisip_mean:.0f} m/km²). "
            + ("Dense informal pathway network already penetrates the settlement. "
               "This reduces the surveying burden for access route identification "
               "but signals that formalisation — not new construction — will be the "
               "primary engineering challenge."
               if above else
               "Sparser road network than the KISIP baseline. Access route creation "
               "will likely require more new construction rather than formalisation "
               "of existing paths. Budget and right-of-way implications are higher.")
        ),
        "paved_proportion": (
            "Paved road proportion",
            f"Value {val:.3f} is {rel} the KISIP treated mean ({kisip_mean:.3f}). "
            + ("A higher share of paved roads than the baseline suggests partial "
               "formalisation has already occurred. Intervention scope may focus on "
               "upgrading remaining unpaved sections rather than full network build-out."
               if above else
               "Predominantly unpaved network. Full road surfacing will likely form "
               "a significant component of the intervention scope and cost estimate.")
        ),
    }
    return signals.get(feat, (feat, f"Value {val:.3f} vs baseline {kisip_mean:.3f}."))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="pg-label">03 / Mukuru Readiness Profiles</div>', unsafe_allow_html=True)
st.markdown('<div class="pg-title">Pre-Intervention Planning Profiles</div>', unsafe_allow_html=True)
st.markdown("""
<div class="pg-sub">
These profiles are a <b style="color:#E2E5EC">planning input tool</b>, not a settlement selection or
ranking system. Each profile characterises what a surveyor will encounter on arrival and which
physical dimensions will demand the most intensive scoping attention — drawn from pre-intervention
satellite features benchmarked against the five KISIP treated settlements.
All four Mukuru sites require investment; the profiles describe <em>how</em> that investment should
be scoped, not <em>whether</em> it should occur.
</div>
""", unsafe_allow_html=True)

# ── Settlement selector ───────────────────────────────────────────────────────
selected = st.selectbox(
    "Select settlement to profile",
    MUKURU_SETTLEMENTS,
    format_func=lambda x: x.replace("_", " "),
    label_visibility="collapsed",
)

s_profile = profiles[profiles["settlement"] == selected].iloc[0]
s_preds   = preds[preds["settlement"] == selected].copy()
s_zones   = mukuru_z[mukuru_z["settlement"] == selected].copy()

mean_pred = s_profile["mean_ensemble_scmi"]
n_zones   = int(s_profile["zones"])
gap       = mean_pred - kisip_baseline_scmi   # positive = above, negative = below

# ── Top stat row ──────────────────────────────────────────────────────────────
direction = "above" if gap >= 0 else "below"
gap_str   = f"{'+'if gap>=0 else '−'}{abs(gap):.4f}"

st.markdown(f"""
<div class="stat-row">
  <div class="stat-cell">
    <div class="sc-val">{mean_pred:.4f}</div>
    <div class="sc-label">Predicted SCMI (ensemble)</div>
  </div>
  <div class="stat-cell">
    <div class="sc-val">{kisip_baseline_scmi:.4f}</div>
    <div class="sc-label">KISIP treated baseline</div>
  </div>
  <div class="stat-cell">
    <div class="sc-val">{gap_str}</div>
    <div class="sc-label">{direction.capitalize()} baseline · ensemble mean</div>
  </div>
  <div class="stat-cell">
    <div class="sc-val">{n_zones}</div>
    <div class="sc-label">50 m grid zones analysed</div>
  </div>
  <div class="stat-cell">
    <div class="sc-val" style="font-size:1rem;padding-top:0.3rem;">Ridge · RF · XGBoost</div>
    <div class="sc-label">All three models agree on ranking</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Context sentence
st.markdown(f"""
<div style="font-size:0.82rem;color:#8B95A8;margin-bottom:1.25rem;line-height:1.6;">
  A predicted SCMI of <b style="color:#E2E5EC">{mean_pred:.4f}</b> means that if this settlement
  underwent the same KISIP physical intervention, the ensemble model estimates surface change of
  this magnitude. The KISIP treated mean is <b style="color:#E2E5EC">{kisip_baseline_scmi:.4f}</b>.
  This settlement sits <b style="color:#E2E5EC">{direction} that reference</b> — use the feature
  scorecards below to understand what that means on the ground.
</div>
""", unsafe_allow_html=True)

col_cards, col_map = st.columns([3, 2], gap="large")

with col_cards:
    st.markdown('<div class="section-rule">Feature scorecards — planning signals</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#8B95A8;margin-bottom:0.75rem;">Each card shows the settlement\'s pre-intervention feature value against the KISIP treated mean and translates it into a concrete planning signal.</div>', unsafe_allow_html=True)

    # Primary spectral features
    FEAT_GROUPS = [
        ("Spectral indices", ["NDBI","NDVI","MNDWI"]),
        ("Texture features", ["Contrast","Entropy","Homogeneity","Correlation"]),
        ("Road network",     ["road_density","paved_proportion"]),
    ]

    for group_label, feats in FEAT_GROUPS:
        st.markdown(f'<div style="font-size:0.7rem;font-weight:600;color:#8B95A8;text-transform:uppercase;letter-spacing:0.1em;margin:0.6rem 0 0.4rem 0;">{group_label}</div>', unsafe_allow_html=True)

        for feat in feats:
            col_name = feat  # names match in profiles CSV
            # Get value from profiles if available, else from preds mean
            feat_col = f"mean_{feat.lower()}" if f"mean_{feat.lower()}" in s_profile.index else None
            # Try common naming patterns
            for candidate in [f"mean_{feat}", f"mean_{feat.lower()}", feat]:
                if candidate in s_profile.index:
                    val = s_profile[candidate]
                    break
            else:
                val = s_preds[feat].mean() if feat in s_preds.columns else None

            if val is None:
                continue

            kisip_mean = kisip_feat_means.get(feat, None)
            if kisip_mean is None:
                continue

            above   = val >= kisip_mean
            rel_txt = f"{'↑' if above else '↓'} {'above' if above else 'below'} KISIP mean ({kisip_mean:.3f if feat not in ['Contrast','road_density'] else '.0f'})"

            # Format display value
            if feat in ["Contrast", "road_density"]:
                val_display = f"{val:,.0f}"
                km_display  = f"{kisip_mean:,.0f}"
                rel_txt = f"{'↑' if above else '↓'} {'above' if above else 'below'} KISIP mean ({km_display})"
            else:
                val_display = f"{val:.3f}"
                km_display  = f"{kisip_mean:.3f}"
                rel_txt = f"{'↑' if above else '↓'} {'above' if above else 'below'} KISIP mean ({km_display})"

            _, interp = feature_signal(feat, val, kisip_mean)
            label, _  = feature_signal(feat, val, kisip_mean)

            st.markdown(f"""
            <div class="scorecard" style="margin-bottom:0.6rem;">
              <div class="sc-feature">{label}</div>
              <div style="display:flex;align-items:baseline;gap:0.75rem;">
                <div class="sc-value">{val_display}</div>
                <div class="sc-vs-kisip {'sc-above' if above else 'sc-below'}">{rel_txt}</div>
              </div>
              <div class="sc-interp">{interp}</div>
            </div>
            """, unsafe_allow_html=True)

with col_map:
    st.markdown('<div class="section-rule">Predicted SCMI — zone map</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#8B95A8;margin-bottom:0.5rem;">Spatial distribution of predicted change magnitude across 50 m zones. Darker zones indicate higher predicted SCMI.</div>', unsafe_allow_html=True)

    if not s_zones.empty:
        centroid = s_zones.to_crs("EPSG:32737").geometry.centroid.to_crs("EPSG:4326")
        mm = folium.Map(location=[centroid.y.mean(), centroid.x.mean()],
                        zoom_start=15, tiles="CartoDB dark_matter")

        scmi_min = s_zones["ensemble_scmi"].min()
        scmi_rng = s_zones["ensemble_scmi"].max() - scmi_min or 1

        for _, row in s_zones.iterrows():
            sv    = row["ensemble_scmi"] if pd.notna(row["ensemble_scmi"]) else 0
            t     = (sv - scmi_min) / scmi_rng
            # Neutral teal gradient — no red/green tier signal
            r_val = int(30  + 80  * t)
            g_val = int(120 + 75  * t)
            b_val = int(140 + 21  * t)
            color = f"#{r_val:02x}{g_val:02x}{b_val:02x}"
            folium.GeoJson(
                row["geometry"].__geo_interface__,
                style_function=lambda f, c=color: {
                    "fillColor": c, "color": "#0F1117",
                    "weight": 0.5, "fillOpacity": 0.85
                },
                tooltip=folium.Tooltip(
                    f"<b>{row['zone_id']}</b><br>Pred. SCMI {sv:.4f}",
                    sticky=False),
            ).add_to(mm)
        st_folium(mm, width=None, height=380, returned_objects=[])
    else:
        st.info("No zone geometries found for this settlement.")

    # SHAP attribution
    st.markdown('<div class="section-rule" style="margin-top:1rem;">SHAP attribution</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#8B95A8;margin-bottom:0.5rem;">Which features most influenced the predicted SCMI for this settlement.</div>', unsafe_allow_html=True)

    shap_cols = [c for c in s_zones.columns if c.startswith("shap_")]
    if shap_cols:
        shap_vals = s_zones[shap_cols].mean().abs().sort_values(ascending=True)
        shap_vals.index = shap_vals.index.str.replace("shap_", "", regex=False)
        top_feat  = shap_vals.idxmax()

        fig_shap = go.Figure(go.Bar(
            x=shap_vals.values, y=shap_vals.index,
            orientation="h",
            marker_color=["#4FC3A1" if f == top_feat else "#2A5A4A"
                          for f in shap_vals.index],
            hovertemplate="%{y}: %{x:.5f}<extra></extra>",
        ))
        fig_shap.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=4, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(28,35,51,0.5)",
            font_color="#8B95A8",
            xaxis=dict(gridcolor="#252D3D", title="Mean |SHAP|", title_font_size=11),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont_size=11),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

# ── All settlements baseline comparison ──────────────────────────────────────
st.markdown('<div class="section-rule">All Mukuru sites — predicted SCMI vs KISIP baseline</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:0.78rem;color:#8B95A8;margin-bottom:0.75rem;">All four sites require investment. This chart shows predicted change magnitude relative to the KISIP treated reference — not a readiness ranking. Sites below the baseline require a different scoping approach, not a lower investment priority.</div>', unsafe_allow_html=True)

comp = profiles[["settlement","mean_ensemble_scmi"]].copy()
comp["label"] = comp["settlement"].str.replace("_", " ")
comp = comp.sort_values("mean_ensemble_scmi", ascending=True)

# Uniform neutral colour — no tier signal
fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(
    y=comp["label"],
    x=comp["mean_ensemble_scmi"],
    orientation="h",
    marker_color=["#4FC3A1" if s == selected.replace("_"," ") else "#2A5A4A"
                  for s in comp["label"]],
    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
))
fig_comp.add_vline(
    x=kisip_baseline_scmi,
    line=dict(color="#8B95A8", width=1.5, dash="dash"),
    annotation_text=f"KISIP treated mean {kisip_baseline_scmi:.3f}",
    annotation_position="top right",
    annotation_font_color="#8B95A8",
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
st.caption("Selected settlement highlighted. Dashed line = KISIP treated mean. All sites below the line are characterised by lower predicted change magnitude — this informs scoping intensity, not investment priority.")