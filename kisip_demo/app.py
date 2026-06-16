import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from styles import inject_css

st.set_page_config(
    page_title="KISIP Settlement Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="pg-label">Kenya Informal Settlement Improvement Project</div>', unsafe_allow_html=True)
st.markdown('<div class="pg-title">Quantifying Physical Surface Change<br>in Nairobi\'s Informal Settlements</div>', unsafe_allow_html=True)
st.markdown("""
<div class="pg-sub">
Satellite remote sensing and machine learning applied to five KISIP Phase 1 treated
settlements and four Mukuru sites. Spectral indices, texture features, and road
network data are combined to predict the Settlement Change Magnitude Index (SCMI)
and generate readiness profiles for future intervention.
</div>
""", unsafe_allow_html=True)

# ── Headline stat row ─────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-row">
  <div class="stat-cell">
    <div class="sc-val">9</div>
    <div class="sc-label">Settlements analysed</div>
  </div>
  <div class="stat-cell">
    <div class="sc-val">5 <span style="font-size:1rem;font-weight:400;color:#8B95A8">KISIP treated</span></div>
    <div class="sc-label">Mathare · Korogocho · Mukuru kwa Njenga · Mukuru kwa Reuben · Kibera</div>
  </div>
  <div class="stat-cell">
    <div class="sc-val">4 <span style="font-size:1rem;font-weight:400;color:#8B95A8">Mukuru sites</span></div>
    <div class="sc-label">Readiness profiling targets</div>
  </div>
  <div class="stat-cell">
    <div class="sc-val sc-accent">CVA</div>
    <div class="sc-label">Change Vector Analysis · 50 m grid</div>
  </div>
  <div class="stat-cell">
    <div class="sc-val">9</div>
    <div class="sc-label">Input features per zone</div>
  </div>
  <div class="stat-cell">
    <div class="sc-val">0.30</div>
    <div class="sc-label">Best R² · Ridge Regression</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Two-column body ───────────────────────────────────────────────────────────
col_nav, col_right = st.columns([3, 2], gap="large")

with col_nav:
    st.markdown('<div class="section-rule">Pages</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="nav-item">
      <div class="nav-n">01</div>
      <div>
        <div class="nav-title">Study Area</div>
        <div class="nav-desc">All 9 settlements on an interactive Nairobi map. KISIP zones in teal, Mukuru sites in amber — zone opacity scales with SCMI magnitude.</div>
      </div>
    </div>
    <div class="nav-item">
      <div class="nav-n">02</div>
      <div>
        <div class="nav-title">KISIP Treated Settlements</div>
        <div class="nav-desc">Select a settlement to explore zone-level SCMI choropleths, SHAP feature attribution, and the observed vs predicted scatter.</div>
      </div>
    </div>
    <div class="nav-item">
      <div class="nav-n">03</div>
      <div>
        <div class="nav-title">Mukuru Readiness Profiles</div>
        <div class="nav-desc">Ensemble model applied to untreated Mukuru sites. Readiness tier and SCMI gap benchmarked against the KISIP treated baseline.</div>
      </div>
    </div>
    <div class="nav-item">
      <div class="nav-n">04</div>
      <div>
        <div class="nav-title">Model Comparison</div>
        <div class="nav-desc">CVA vs PCA SCMI method evaluation. Ridge, Random Forest, and XGBoost performance across LOSO cross-validation folds.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-rule">Study objectives</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.85rem; color:#C5CAD6; line-height:1.7;">
      <div style="margin-bottom:0.6rem;">
        <span style="color:#4FC3A1;font-family:'JetBrains Mono',monospace;font-size:0.7rem;">01 &nbsp;</span>
        Extract and describe spectral surface conditions, structural texture properties and road network features of the five KISIP Phase 1 treated settlements.
      </div>
      <div style="margin-bottom:0.6rem;">
        <span style="color:#4FC3A1;font-family:'JetBrains Mono',monospace;font-size:0.7rem;">02 &nbsp;</span>
        Quantify the spatial distribution and magnitude of surface-level change between pre-intervention (2009–2011) and post-intervention (2021–2023).
      </div>
      <div style="margin-bottom:0.6rem;">
        <span style="color:#4FC3A1;font-family:'JetBrains Mono',monospace;font-size:0.7rem;">03 &nbsp;</span>
        Identify which pre-intervention physical parameters are most strongly correlated with the magnitude of surface change detected.
      </div>
      <div>
        <span style="color:#4FC3A1;font-family:'JetBrains Mono',monospace;font-size:0.7rem;">04 &nbsp;</span>
        Apply the trained model to untreated informal settlements to generate preliminary readiness profiles with SHAP-translated feature attribution.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-rule" style="margin-top:1.5rem;">Feature set</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top:0.3rem">
      <span class="dtag">NDVI</span><span class="dtag">NDBI</span><span class="dtag">MNDWI</span>
      <span class="dtag">Contrast</span><span class="dtag">Entropy</span><span class="dtag">Homogeneity</span>
      <span class="dtag">Correlation</span><span class="dtag">road_density</span><span class="dtag">paved_proportion</span>
    </div>
    <div style="font-size:0.75rem;color:#8B95A8;margin-top:0.5rem;line-height:1.6;">
      Spectral (3) · GLCM texture (4) · Road network (2)
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div style="margin-top:2rem;font-size:0.78rem;color:#8B95A8;">← Select a page from the sidebar to begin.</div>', unsafe_allow_html=True)
