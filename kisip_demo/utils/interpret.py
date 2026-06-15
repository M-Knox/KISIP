"""Plain-language and technical copy helpers."""

from __future__ import annotations

import streamlit as st

from utils.constants import FEATURES, SCMI_PERIODS

FEATURE_LABELS = {
    "NDVI": ("Vegetation cover", "NDVI"),
    "NDBI": ("Built-up surface", "NDBI"),
    "MNDWI": ("Water / moisture", "MNDWI"),
    "Contrast": ("Surface texture contrast", "Contrast"),
    "Entropy": ("Surface texture complexity", "Entropy"),
    "Homogeneity": ("Surface uniformity", "Homogeneity"),
    "Correlation": ("Texture correlation", "Correlation"),
    "road_density": ("Road network density", "road_density"),
    "paved_proportion": ("Share of paved roads", "paved_proportion"),
}

GLOSSARY = {
    "SCMI": (
        "Settlement Change Magnitude Index",
        "How much the physical land surface changed between satellite images "
        f"taken in {SCMI_PERIODS}.",
    ),
    "CVA": (
        "Change Vector Analysis",
        "A method that measures how much and in which direction surface "
        "conditions shifted between the two time periods.",
    ),
    "SHAP": (
        "Feature attribution (SHAP)",
        "Shows which measured factors most influenced the model's prediction "
        "for each zone.",
    ),
    "LOSO-CV": (
        "Leave-one-settlement-out cross-validation",
        "Each model is tested on a settlement it was not trained on, "
        "to check whether results generalise.",
    ),
}

PAGE_EXPLAINERS = {
    "home": (
        "This tool maps physical changes in Nairobi's informal settlements after "
        "KISIP upgrading work, and estimates how much similar change might occur "
        "at Mukuru sites that have not yet been treated.",
        "Remote-sensing ML pipeline: GEE spectral extraction → texture (GLCM) → "
        "road network (OSMnx) → ensemble models → SHAP attribution → SCMI.",
    ),
    "study_area": (
        "All nine study sites on one map. Teal areas are KISIP-treated settlements "
        "where change was measured directly. Amber areas are Mukuru sites where "
        "change was estimated by the model.",
        "Combined GeoJSON of kisip_zones_spatial and mukuru_zones_spatial. "
        "Opacity scales with SCMI magnitude per zone.",
    ),
    "kisip": (
        "For each treated settlement, see where physical change was strongest "
        "and which satellite-derived factors the model linked to that change.",
        "Zone-level observed SCMI (CVA), model predictions, SHAP attribution "
        "(XGBoost), and optional CVA direction layer.",
    ),
    "mukuru": (
        "Predicted change at four Mukuru readiness sites, ranked against the "
        "average change seen in KISIP-treated settlements.",
        "Ensemble SCMI predictions benchmarked to KISIP mean SCMI. "
        "Readiness tier = ratio of predicted SCMI to KISIP baseline.",
    ),
    "models": (
        "Compare how well three models predict settlement change, and whether "
        "CVA or PCA is the better way to measure change.",
        "LOSO-CV metrics for Ridge, Random Forest, and XGBoost. "
        "Global SHAP importance from XGBoost.",
    ),
}


def is_technical() -> bool:
    return st.session_state.get("view_mode") == "Technical"


def feature_label(name: str) -> str:
    simple, technical = FEATURE_LABELS.get(name, (name, name))
    return f"{simple} ({technical})" if is_technical() else simple


def term_label(key: str) -> str:
    simple, technical = GLOSSARY.get(key, (key, key))
    return technical if is_technical() else simple


def scmi_interpretation(value: float) -> str:
    if value < 0.10:
        level = "Minimal change — surface conditions were largely stable"
    elif value < 0.20:
        level = "Moderate change — visible upgrading activity detected"
    else:
        level = "Substantial change — major physical transformation"
    if is_technical():
        return f"{level} · SCMI {value:.4f}"
    return f"{level} between {SCMI_PERIODS}"


def readiness_tier(score: float, baseline: float) -> tuple[str, str]:
    ratio = score / baseline if baseline > 0 else 0
    if ratio >= 0.75:
        return "High Readiness", "high"
    if ratio >= 0.45:
        return "Moderate Readiness", "medium"
    return "Low Readiness", "low"


def tier_dot_class(tier_key: str) -> str:
    return {"high": "dot-green", "medium": "dot-amber", "low": "dot-red"}.get(
        tier_key, "dot-green"
    )


def page_explainer(page_key: str) -> None:
    simple, technical = PAGE_EXPLAINERS[page_key]
    text = technical if is_technical() else simple
    expanded = not is_technical()
    with st.expander("What does this page show?", expanded=expanded):
        st.markdown(text)


def metric_block(label: str, value: str, hint: str | None = None) -> str:
    hint_html = f'<div class="metric-hint">{hint}</div>' if hint else ""
    return f"""
    <div class="metric-block">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {hint_html}
    </div>"""
