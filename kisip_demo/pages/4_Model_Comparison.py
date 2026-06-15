import sys
from pathlib import Path

_DEMO = Path(__file__).resolve().parent.parent
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loaders import load_model_comparison_data
from utils.interpret import feature_label, is_technical, metric_block, page_explainer, term_label
from utils.sidebar import render_sidebar
from utils.styling import inject_styles, page_header, plotly_layout

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

inject_styles()
render_sidebar()

selected_model = st.session_state["selected_model"]
model_comp, cva_pca, shap_imp, _preds = load_model_comparison_data()

lead = (
    "How reliably three models predict settlement change, and whether measuring "
    "change with CVA or PCA works better."
    if not is_technical()
    else "LOSO-CV metrics for Ridge, Random Forest, and XGBoost. CVA vs PCA SCMI methods."
)

page_header("04 · Model comparison", "Validation and method choice", lead)
page_explainer("models")

best_model = model_comp.loc[model_comp["R²"].idxmax()]
best_rmse = model_comp.loc[model_comp["RMSE"].idxmin()]
cva_best = cva_pca[cva_pca["SCMI_Method"] == "CVA"]["R2"].max()
pca_best = cva_pca[cva_pca["SCMI_Method"] == "PCA"]["R2"].max()

sel_row = model_comp[model_comp["Model"] == selected_model].iloc[0]

r2_label = "Selected model R²" if is_technical() else "Model accuracy (selected)"
kc1, kc2, kc3, kc4 = st.columns(4)
with kc1:
    st.markdown(
        metric_block(r2_label, f"{sel_row['R²']:.3f}", selected_model),
        unsafe_allow_html=True,
    )
with kc2:
    st.markdown(
        metric_block("Best R² overall", f"{best_model['R²']:.3f}", best_model["Model"]),
        unsafe_allow_html=True,
    )
with kc3:
    cva_label = "CVA vs PCA" if is_technical() else "Change method comparison"
    st.markdown(
        metric_block(cva_label, f"{cva_best:.3f}", f"PCA best: {pca_best:.3f}"),
        unsafe_allow_html=True,
    )
with kc4:
    st.markdown(metric_block("RMSE (selected)", f"{sel_row['RMSE']:.4f}"), unsafe_allow_html=True)

st.markdown("---")

col_left, col_right = st.columns(2, gap="large")

MODEL_COLORS = {
    "Ridge Regression": "#4FC3A1",
    "Random Forest": "#F5A623",
    "XGBoost": "#9B59B6",
}

with col_left:
    st.markdown('<div class="section-title">Cross-validation performance</div>', unsafe_allow_html=True)
    if is_technical():
        st.caption(term_label("LOSO-CV"))

    r2_col = "R²" if "R²" in model_comp.columns else "R2"
    mae_col = "MAE"
    rmse_col = "RMSE"

    fig_bar = go.Figure()
    for i, (metric, label) in enumerate(zip([rmse_col, mae_col, r2_col], ["RMSE", "MAE", "R²"])):
        if metric not in model_comp.columns:
            continue
        fig_bar.add_trace(
            go.Bar(
                name=label,
                x=model_comp["Model"],
                y=model_comp[metric],
                text=model_comp[metric].round(4),
                textposition="outside",
                visible=(metric == rmse_col),
            )
        )

    fig_bar.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label="RMSE",
                        method="update",
                        args=[{"visible": [True, False, False]}, {"yaxis.title.text": "RMSE"}],
                    ),
                    dict(
                        label="MAE",
                        method="update",
                        args=[{"visible": [False, True, False]}, {"yaxis.title.text": "MAE"}],
                    ),
                    dict(
                        label="R²",
                        method="update",
                        args=[{"visible": [False, False, True]}, {"yaxis.title.text": "R²"}],
                    ),
                ],
                direction="left",
                x=0.0,
                y=1.12,
                type="buttons",
                bgcolor="rgba(26,31,46,0.8)",
                bordercolor="rgba(255,255,255,0.1)",
                font_color="#E8EAF0",
            )
        ],
        **plotly_layout(height=300, showlegend=False, yaxis_title="RMSE"),
    )
    for trace in fig_bar.data:
        trace.marker.color = [
            MODEL_COLORS.get(m, "#4FC3A1") for m in model_comp["Model"]
        ]
        trace.marker.line = dict(
            width=[3 if m == selected_model else 0 for m in model_comp["Model"]],
            color="#E8EAF0",
        )
    fig_bar.update_xaxes(gridcolor="rgba(0,0,0,0)")
    fig_bar.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-title">Metrics</div>', unsafe_allow_html=True)
    def highlight_selected(row):
        color = "background-color: rgba(79,195,161,0.12)" if row.name == selected_model else ""
        return [color] * len(row)

    styled = (
        model_comp.set_index("Model")
        .style.format({rmse_col: "{:.6f}", mae_col: "{:.6f}", r2_col: "{:.6f}"})
        .apply(highlight_selected, axis=1)
    )
    st.dataframe(styled, use_container_width=True)

with col_right:
    st.markdown('<div class="section-title">CVA vs PCA</div>', unsafe_allow_html=True)
    fig_cvapca = go.Figure()
    r2_col_cva = "R2" if "R2" in cva_pca.columns else "R²"

    for method, color in [("CVA", "#4FC3A1"), ("PCA", "#F5A623")]:
        subset = cva_pca[cva_pca["SCMI_Method"] == method]
        fig_cvapca.add_trace(
            go.Bar(
                name=method,
                x=subset["Model"],
                y=subset[r2_col_cva],
                marker_color=color,
                text=subset[r2_col_cva].round(4),
                textposition="outside",
            )
        )

    fig_cvapca.update_layout(
        barmode="group",
        **plotly_layout(height=260, yaxis_title="R²"),
    )
    fig_cvapca.update_xaxes(gridcolor="rgba(0,0,0,0)")
    fig_cvapca.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    st.plotly_chart(fig_cvapca, use_container_width=True)

    shap_title = (
        "Strongest predictors (XGBoost)"
        if not is_technical()
        else "Global SHAP importance — XGBoost"
    )
    st.markdown(f'<div class="section-title">{shap_title}</div>', unsafe_allow_html=True)

    if not shap_imp.empty:
        val_col = [c for c in shap_imp.columns if c not in ("feature", "settlement", "zone_id")]
        if val_col:
            imp_col = val_col[0]
            if "feature" in shap_imp.columns:
                fi = shap_imp[["feature", imp_col]].sort_values(imp_col, ascending=True)
                x_vals = fi[imp_col]
                y_vals = [feature_label(f) for f in fi["feature"]]
            else:
                fi = shap_imp.set_index(shap_imp.columns[0])[imp_col].sort_values(ascending=True)
                x_vals = fi.values
                y_vals = [feature_label(f) for f in fi.index]

            colors = [
                "#4FC3A1" if i >= len(x_vals) - 3 else "rgba(79,195,161,0.45)"
                for i in range(len(x_vals))
            ]
            fig_imp = go.Figure(
                go.Bar(
                    x=list(x_vals),
                    y=y_vals,
                    orientation="h",
                    marker_color=colors,
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
                )
            )
            fig_imp.update_layout(**plotly_layout(height=280))
            fig_imp.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig_imp, use_container_width=True)

with st.expander("Full CVA vs PCA table", expanded=False):
    st.dataframe(cva_pca.set_index("Model"), use_container_width=True)

st.markdown('<div class="section-title">Findings</div>', unsafe_allow_html=True)

if is_technical():
    findings = [
        f"CVA outperforms PCA as SCMI method — CVA RMSE &lt;0.04 vs PCA &gt;0.48.",
        f"Ridge Regression leads on R² ({best_model['R²']:.4f}) — largely linear feature–SCMI relationship.",
        "Road features near-zero in several zones — limits road_density SHAP contribution.",
        "NDVI and NDBI dominate SHAP — vegetation loss and built-up expansion drive CVA signal.",
    ]
else:
    findings = [
        "Measuring change with CVA works much better than PCA for this dataset.",
        "The simplest model (Ridge) predicts almost as well as the more complex ones.",
        "Road data was sparse before intervention — road-related signals are weak.",
        "Vegetation loss and more built-up surface are the main change patterns detected.",
    ]

fc1, fc2 = st.columns(2)
with fc1:
    for f in findings[:2]:
        st.markdown(f'<p class="section-caption" style="line-height:1.6;margin-bottom:1rem">{f}</p>', unsafe_allow_html=True)
with fc2:
    for f in findings[2:]:
        st.markdown(f'<p class="section-caption" style="line-height:1.6;margin-bottom:1rem">{f}</p>', unsafe_allow_html=True)
