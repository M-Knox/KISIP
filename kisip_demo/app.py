import sys
from pathlib import Path

_DEMO = Path(__file__).resolve().parent
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

import streamlit as st

from utils.data_loaders import load_home_metrics
from utils.interpret import is_technical, metric_block, page_explainer, term_label
from utils.sidebar import render_sidebar
from utils.styling import inject_styles, page_header

st.set_page_config(
    page_title="KISIP Settlement Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()
render_sidebar()

kisip_gdf, mukuru_gdf, model_comp = load_home_metrics()
best = model_comp.loc[model_comp["R²"].idxmax()]
n_kisip = kisip_gdf["settlement"].nunique()
n_mukuru = mukuru_gdf["settlement"].nunique()

page_header(
    "KISIP · Nairobi informal settlements",
    "Settlement change intelligence",
    "Satellite-based analysis of physical surface change after KISIP upgrading, "
    "with predictions for Mukuru readiness sites.",
)

page_explainer("home")

st.markdown(
    """
<div class="summary-box">
    <em>In plain terms:</em> We compared satellite images of Nairobi's informal
    settlements from before and after KISIP infrastructure work. Areas that changed
    most — new surfaces, less vegetation, more built-up land — score higher on our
    change index. The same approach is applied to four Mukuru sites to estimate how
    ready they may be for similar intervention.
</div>
""",
    unsafe_allow_html=True,
)

r2_label = "Best model fit (R²)" if is_technical() else "Model accuracy"
r2_hint = (
    f"{best['Model']} · leave-one-settlement-out validation"
    if is_technical()
    else f"How well the model predicts change it hasn't seen — {best['R²']:.0%} of variation explained"
)

metrics_html = f"""
<div class="metrics-row">
    {metric_block("Settlements analysed", str(n_kisip + n_mukuru), f"{n_kisip} treated · {n_mukuru} Mukuru")}
    {metric_block(r2_label, f"{best['R²']:.2f}", r2_hint)}
    {metric_block(term_label("SCMI") if is_technical() else "Change index method", "CVA" if is_technical() else "Change vector analysis", "50 m grid zones")}
    {metric_block("Grid resolution", "50 m", "Zone size")}
    {metric_block("Input features", "9", "Spectral · texture · roads")}
</div>
"""
st.markdown(metrics_html, unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="section-title">Explore the analysis</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="nav-list">
    <div class="nav-item"><strong>Study area</strong> — all nine sites on one map</div>
    <div class="nav-item"><strong>KISIP treated</strong> — measured change and drivers per settlement</div>
    <div class="nav-item"><strong>Mukuru readiness</strong> — predicted change and ranking</div>
    <div class="nav-item"><strong>Model comparison</strong> — validation and method choice</div>
</div>
""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown('<div class="section-title">Measured inputs</div>', unsafe_allow_html=True)
    if is_technical():
        tags = [
            "NDVI", "NDBI", "MNDWI", "Contrast", "Entropy",
            "Homogeneity", "Correlation", "road_density", "paved_proportion",
        ]
    else:
        tags = [
            "Vegetation", "Built-up surface", "Water index", "Texture",
            "Road density", "Paved roads",
        ]
    tag_html = "".join(f'<span class="tag">{t}</span>' for t in tags)
    st.markdown(f'<div style="margin-top:0.5rem">{tag_html}</div>', unsafe_allow_html=True)

    if is_technical():
        st.markdown(
            '<p class="section-caption" style="margin-top:1rem">'
            "GEE → spectral indices → GLCM texture → OSMnx roads → "
            "Ridge / RF / XGBoost → SHAP → SCMI</p>",
            unsafe_allow_html=True,
        )

st.markdown(
    '<p class="section-caption">Use the sidebar to switch language and prediction model.</p>',
    unsafe_allow_html=True,
)
