import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from styles import inject_css

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
inject_css()

@st.cache_data
def load_data():
    model_comp = pd.read_csv("data/kisip_model_comparison.csv")
    cva_pca    = pd.read_csv("data/kisip_cva_vs_pca_comparison.csv")
    shap_imp   = pd.read_csv("data/kisip_shap_importance.csv")
    return model_comp, cva_pca, shap_imp

model_comp, cva_pca, shap_imp = load_data()

MODEL_COLORS = {
    "Ridge Regression": "#4FC3A1",
    "Random Forest":    "#F5A623",
    "XGBoost":          "#9B7FD4",
    "Ridge":            "#4FC3A1",
}

# Robust R² column detection — handles both "R²" (with superscript) and "R2"
R2_COL = "R²" if "R²" in model_comp.columns else "R2"
# Normalise column name in cva_pca too
CVA_R2 = "R2" if "R2" in cva_pca.columns else "R²"

best_r2   = model_comp.loc[model_comp[R2_COL].idxmax()]
best_rmse = model_comp.loc[model_comp["RMSE"].idxmin()]
cva_best  = cva_pca[cva_pca["SCMI_Method"]=="CVA"][CVA_R2].max()
pca_best  = cva_pca[cva_pca["SCMI_Method"]=="PCA"][CVA_R2].max()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="pg-label">04 / Model Comparison</div>', unsafe_allow_html=True)
st.markdown('<div class="pg-title">Ensemble Performance & SCMI Method Evaluation</div>', unsafe_allow_html=True)
st.markdown('<div class="pg-sub">Leave-One-Settlement-Out cross-validation metrics for Ridge Regression, Random Forest, and XGBoost. CVA and PCA compared as SCMI computation methods across all three models.</div>', unsafe_allow_html=True)

# ── Stat row ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stat-row">
  <div class="stat-cell">
    <div class="sc-val sc-accent">{best_r2[R2_COL]:.4f}</div>
    <div class="sc-label">Best R² — {best_r2['Model']}</div>
  </div>
  <div class="stat-cell">
    <div class="sc-val">{best_rmse['RMSE']:.4f}</div>
    <div class="sc-label">Best RMSE — {best_rmse['Model']}</div>
  </div>
  <div class="stat-cell">
    <div class="sc-val sc-accent">{cva_best:.4f}</div>
    <div class="sc-label">CVA best R²</div>
  </div>
  <div class="stat-cell">
    <div class="sc-val">{pca_best:.4f}</div>
    <div class="sc-label">PCA best R²</div>
  </div>
  <div class="stat-cell">
    <div class="sc-val">3</div>
    <div class="sc-label">Models evaluated</div>
  </div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<div class="section-rule">LOSO-CV performance</div>', unsafe_allow_html=True)

    metric_opt = st.radio("Metric", ["RMSE", "MAE", R2_COL],
                          horizontal=True, label_visibility="collapsed")

    bar_colors = [MODEL_COLORS.get(m, "#4FC3A1") for m in model_comp["Model"]]
    fig_bar = go.Figure(go.Bar(
        x=model_comp["Model"],
        y=model_comp[metric_opt],
        marker_color=bar_colors,
        text=model_comp[metric_opt].round(4),
        textposition="outside",
        textfont_size=11,
        hovertemplate="%{x}: %{y:.6f}<extra></extra>",
    ))
    fig_bar.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(28,35,51,0.5)",
        font_color="#8B95A8",
        yaxis=dict(gridcolor="#252D3D", title=metric_opt, title_font_size=11),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-rule">Full metric table</div>', unsafe_allow_html=True)
    styled = model_comp.set_index("Model").style \
        .format({R2_COL: "{:.6f}", "MAE": "{:.6f}", "RMSE": "{:.6f}"}) \
        .highlight_max(subset=[R2_COL],       color="#0d2a1e") \
        .highlight_min(subset=["RMSE","MAE"], color="#0d2a1e")
    st.dataframe(styled, use_container_width=True)

with col_right:
    st.markdown('<div class="section-rule">CVA vs PCA — R² by model</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#8B95A8;margin-bottom:0.6rem;">R² is scale-invariant and the appropriate comparison metric. CVA and PCA achieve comparable R² (max CVA {:.2f} vs max PCA {:.2f}), but CVA is preferred for interpretability — its magnitude is directly tied to spectral change units, whereas PCA produces a scale-free latent score.</div>'.format(cva_best, pca_best), unsafe_allow_html=True)

    fig_cp = go.Figure()
    for method, color in [("CVA","#4FC3A1"), ("PCA","#F5A623")]:
        sub = cva_pca[cva_pca["SCMI_Method"] == method]
        fig_cp.add_trace(go.Bar(
            name=method, x=sub["Model"], y=sub[CVA_R2],
            marker_color=color,
            text=sub[CVA_R2].round(4), textposition="outside", textfont_size=10,
            hovertemplate=f"{method} — %{{x}}: %{{y:.4f}}<extra></extra>",
        ))
    fig_cp.update_layout(
        barmode="group", height=260,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(28,35,51,0.5)",
        font_color="#8B95A8",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08),
        yaxis=dict(gridcolor="#252D3D", title="R²", title_font_size=11),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_cp, use_container_width=True)

    st.markdown('<div class="section-rule">Global SHAP feature importance</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#8B95A8;margin-bottom:0.6rem;">Mean absolute SHAP value across all KISIP zones — overall driver ranking for the XGBoost model.</div>', unsafe_allow_html=True)

    if not shap_imp.empty and "mean_abs_shap" in shap_imp.columns:
        fi = shap_imp[["feature","mean_abs_shap"]].sort_values("mean_abs_shap", ascending=True)
        top_imp = fi["mean_abs_shap"].idxmax()
        imp_colors = ["#4FC3A1" if i == top_imp else "#2A5A4A" for i in fi.index]
        fig_imp = go.Figure(go.Bar(
            x=fi["mean_abs_shap"], y=fi["feature"],
            orientation="h", marker_color=imp_colors,
            hovertemplate="%{y}: %{x:.5f}<extra></extra>",
        ))
        fig_imp.update_layout(
            height=290,
            margin=dict(l=0, r=0, t=4, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(28,35,51,0.5)",
            font_color="#8B95A8",
            xaxis=dict(gridcolor="#252D3D", title="Mean |SHAP|", title_font_size=11),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont_size=11),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ── Key findings ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-rule">Key findings</div>', unsafe_allow_html=True)
fc1, fc2 = st.columns(2, gap="large")
with fc1:
    st.markdown(f"""
    <div class="finding">
        <b>CVA is the preferred SCMI method for interpretability.</b> CVA and PCA achieve
        comparable R² across all models (max CVA R²={cva_best:.2f}, max PCA R²={pca_best:.2f} under Ridge),
        but CVA magnitude is directly tied to spectral change units whereas PCA produces
        a scale-free latent score — making CVA results more defensible in a policy context.
    </div>
    <div class="finding">
        <b>Ridge Regression achieves the highest R²</b> ({best_r2[R2_COL]:.4f}) despite being the
        simplest model — suggesting the feature-to-SCMI relationship is largely linear after
        spectral preprocessing via CVA.
    </div>
    """, unsafe_allow_html=True)
with fc2:
    st.markdown("""
    <div class="finding">
        <b>Road features are data-limited.</b> road_density and paved_proportion show near-zero
        values in many pre-intervention zones, suppressing their SHAP contribution — a known
        limitation for the pre-intervention imagery period.
    </div>
    <div class="finding">
        <b>NDVI and NDBI dominate SHAP attribution</b> across all settlements, consistent with
        vegetation clearance and built-surface expansion being the primary physical change
        signal captured by CVA between the two epochs.
    </div>
    """, unsafe_allow_html=True)

with st.expander("Full CVA vs PCA table", expanded=False):
    st.dataframe(cva_pca.set_index("Model"), use_container_width=True)
