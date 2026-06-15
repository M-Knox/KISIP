import streamlit as st

st.set_page_config(
    page_title="KISIP Settlement Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0F1117 0%, #1A1F2E 40%, #0d2137 100%);
    border: 1px solid #4FC3A1;
    border-radius: 12px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(79,195,161,0.12) 0%, transparent 70%);
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    color: #4FC3A1;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #E8EAF0;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-size: 1rem;
    color: #8B95A8;
    max-width: 640px;
    line-height: 1.6;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
.metric-card {
    background: #1A1F2E;
    border: 1px solid #252D3D;
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    flex: 1;
    min-width: 140px;
}
.metric-card .m-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: #8B95A8;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-card .m-value {
    font-size: 2rem;
    font-weight: 700;
    color: #4FC3A1;
    line-height: 1;
}
.metric-card .m-unit {
    font-size: 0.8rem;
    color: #8B95A8;
    margin-top: 0.2rem;
}

/* Nav pills */
.nav-guide {
    background: #1A1F2E;
    border-radius: 10px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1rem;
}
.nav-guide h4 {
    color: #4FC3A1;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}
.nav-step {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #252D3D;
    color: #C5CAD6;
    font-size: 0.9rem;
}
.nav-step:last-child { border-bottom: none; }
.nav-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #4FC3A1;
    background: rgba(79,195,161,0.1);
    border-radius: 4px;
    padding: 2px 7px;
    flex-shrink: 0;
}

/* Data tag */
.data-tag {
    display: inline-block;
    background: rgba(79,195,161,0.1);
    border: 1px solid rgba(79,195,161,0.3);
    color: #4FC3A1;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px;
}

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #4FC3A1;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🛰️ &nbsp; Kenya Informal Settlement Improvement Project &nbsp;·&nbsp; Remote Sensing ML</div>
    <div class="hero-title">Quantifying Physical Surface Change<br>in Nairobi's Informal Settlements</div>
    <div class="hero-sub">
        XGBoost ensemble model trained on spectral indices (CVA/PCA), texture features,
        and road network density to predict Settlement Change Magnitude Index (SCMI)
        across KISIP-treated zones and Mukuru readiness sites.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Headline metrics ──────────────────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
    <div class="metric-card">
        <div class="m-label">Settlements</div>
        <div class="m-value">9</div>
        <div class="m-unit">5 KISIP · 4 Mukuru</div>
    </div>
    <div class="metric-card">
        <div class="m-label">Best Model R²</div>
        <div class="m-value">0.30</div>
        <div class="m-unit">Ridge Regression (CVA)</div>
    </div>
    <div class="metric-card">
        <div class="m-label">SCMI Method</div>
        <div class="m-value">CVA</div>
        <div class="m-unit">Change Vector Analysis</div>
    </div>
    <div class="metric-card">
        <div class="m-label">Zone Grid</div>
        <div class="m-value">50m</div>
        <div class="m-unit">Spatial resolution</div>
    </div>
    <div class="metric-card">
        <div class="m-label">Features</div>
        <div class="m-value">9</div>
        <div class="m-unit">Spectral · Texture · Road</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Navigation guide ──────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="nav-guide">
        <h4>Navigate the analysis</h4>
        <div class="nav-step"><span class="nav-num">01</span>Study Area — All 9 settlements on one map</div>
        <div class="nav-step"><span class="nav-num">02</span>KISIP Treated — Zone SCMI + SHAP attribution per settlement</div>
        <div class="nav-step"><span class="nav-num">03</span>Mukuru Readiness — Predicted change + readiness tier profiles</div>
        <div class="nav-step"><span class="nav-num">04</span>Model Comparison — CVA vs PCA, ensemble performance</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-label">Feature set</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top:0.3rem">
        <span class="data-tag">NDVI</span>
        <span class="data-tag">NDBI</span>
        <span class="data-tag">MNDWI</span>
        <span class="data-tag">Contrast</span>
        <span class="data-tag">Entropy</span>
        <span class="data-tag">Homogeneity</span>
        <span class="data-tag">Correlation</span>
        <span class="data-tag">road_density</span>
        <span class="data-tag">paved_proportion</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.2rem">Data pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#8B95A8; font-size:0.85rem; line-height:1.8; margin-top:0.3rem;">
        Google Earth Engine → Spectral extraction → Texture (GLCM) →
        Road network (OSMnx) → Feature matrix → XGBoost ensemble →
        SHAP attribution → SCMI prediction
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div style="color:#8B95A8; font-size:0.8rem;">Use the sidebar ← to navigate between pages.</div>',
    unsafe_allow_html=True
)
