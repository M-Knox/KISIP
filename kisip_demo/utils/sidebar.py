"""Global sidebar controls."""

import streamlit as st

from utils.constants import MODEL_OPTIONS


def init_session_state() -> None:
    st.session_state.setdefault("view_mode", "Simple")
    st.session_state.setdefault("selected_model", "Ridge Regression")


def render_sidebar() -> None:
    init_session_state()

    st.sidebar.markdown("### Settings")

    st.session_state["view_mode"] = st.sidebar.radio(
        "Language",
        ["Simple", "Technical"],
        index=0 if st.session_state["view_mode"] == "Simple" else 1,
        help="Simple uses plain English. Technical shows acronyms and method names.",
    )

    st.session_state["selected_model"] = st.sidebar.selectbox(
        "Prediction model",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state["selected_model"]),
        help="Applies to predicted SCMI on pages 2–4. Observed SCMI is unchanged.",
    )

    if st.session_state["view_mode"] == "Technical":
        st.sidebar.caption(
            "SHAP attribution is computed for XGBoost only. "
            "Predictions reflect the selected model above."
        )
