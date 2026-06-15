import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.page-eyebrow { font-family:'JetBrains Mono',monospace; font-size:0.65rem; letter-spacing:0.18em; color:#4FC3A1; text-transform:uppercase; margin-bottom:0.4rem; }
.page-title { font-size:1.8rem; font-weight:700; color:#E8EAF0; margin-bottom:0.2rem; }
.page-sub { font-size:0.9rem; color:#8B95A8; margin-bottom:1.5rem; }
.info-card { background:#1A1F2E; border:1px solid #252D3D; border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.8rem; }
.card-label { font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:#8B95A8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.2rem; }
.card-value { font-size:1.6rem; font-weight:700; color:#4FC3A1; }
.card-sub { font-size:0.75rem; color:#8B95A8; margin-top:0.1rem; }
.finding-box { background:#1A1F2E; border-left:3px solid #4FC3A1; border-radius:0 8px 8px 0; padding:0.8rem 1.2rem; margin-bottom:0.6rem; font-size:0.88rem; color:#C5CAD6; line-height:1.5; }
.finding-box b { color:#4FC3A1; }
</style>
""", unsafe_allow_html=True)

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    model_comp = pd.read_csv("data/kisip_model_comparison.csv")
    cva_pca    = pd.read_csv("data/kisip_cva_vs_pca_comparison.csv")
    shap_imp   = pd.read_csv("data/kisip_shap_importance.csv")
    preds      = pd.read_csv("data/kisip_model_predictions.csv")
    return model_comp, cva_pca, shap_imp, preds

model_comp, cva_pca, shap_imp, preds = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-eyebrow">04 / Model Comparison</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Ensemble Performance & SCMI Method Evaluation</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">LOSO-CV metrics across Ridge Regression, Random Forest, and XGBoost. CVA vs PCA comparison for SCMI computation method selection.</div>', unsafe_allow_html=True)

# ── Top KPI row ───────────────────────────────────────────────────────────────
best_model = model_comp.loc[model_comp["R²"].idxmax()]
best_rmse  = model_comp.loc[model_comp["RMSE"].idxmin()]

kc1, kc2, kc3, kc4 = st.columns(4)
with kc1:
    st.markdown(f"""<div class="info-card">
        <div class="card-label">Best R² (model)</div>
        <div class="card-value">{best_model['R²']:.4f}</div>
        <div class="card-sub">{best_model['Model']}</div>
    </div>""", unsafe_allow_html=True)
with kc2:
    st.markdown(f"""<div class="info-card">
        <div class="card-label">Best RMSE</div>
        <div class="card-value">{best_rmse['RMSE']:.4f}</div>
        <div class="card-sub">{best_rmse['Model']}</div>
    </div>""", unsafe_allow_html=True)
with kc3:
    cva_best = cva_pca[cva_pca["SCMI_Method"]=="CVA"]["R2"].max()
    pca_best = cva_pca[cva_pca["SCMI_Method"]=="PCA"]["R2"].max()
    st.markdown(f"""<div class="info-card">
        <div class="card-label">CVA best R²</div>
        <div class="card-value">{cva_best:.4f}</div>
        <div class="card-sub">vs PCA: {pca_best:.4f}</div>
    </div>""", unsafe_allow_html=True)
with kc4:
    n_models = len(model_comp)
    st.markdown(f"""<div class="info-card">
        <div class="card-label">Models evaluated</div>
        <div class="card-value">{n_models}</div>
        <div class="card-sub">Ridge · RF · XGBoost</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

col_left, col_right = st.columns(2, gap="large")

# ── Left column: model perf table + grouped bar ───────────────────────────────
with col_left:
    st.markdown("**LOSO-CV performance — all models**")

    r2_col  = "R²"  if "R²"  in model_comp.columns else "R2"
    mae_col = "MAE" if "MAE" in model_comp.columns else "MAE"
    rmse_col= "RMSE"

    MODEL_COLORS = {
        "Ridge Regression": "#4FC3A1",
        "Random Forest":    "#F5A623",
        "XGBoost":          "#9B59B6",
        "Ridge":            "#4FC3A1",
    }

    fig_bar = go.Figure()
    metrics_to_plot = [rmse_col, mae_col, r2_col]
    metric_labels   = ["RMSE", "MAE", "R²"]

    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        if metric not in model_comp.columns:
            continue
        fig_bar.add_trace(go.Bar(
            name=label,
            x=model_comp["Model"],
            y=model_comp[metric],
            text=model_comp[metric].round(4),
            textposition="outside",
            visible=(i == 0),
        ))

    fig_bar.update_layout(
        updatemenus=[dict(
            buttons=[
                dict(label="RMSE", method="update",
                     args=[{"visible":[True,False,False]}, {"yaxis.title.text":"RMSE"}]),
                dict(label="MAE",  method="update",
                     args=[{"visible":[False,True,False]}, {"yaxis.title.text":"MAE"}]),
                dict(label="R²",   method="update",
                     args=[{"visible":[False,False,True]}, {"yaxis.title.text":"R²"}]),
            ],
            direction="left", x=0.0, y=1.15, type="buttons",
            bgcolor="#1A1F2E", bordercolor="#4FC3A1", font_color="#C5CAD6",
        )],
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,31,46,0.6)",
        font_color="#C5CAD6",
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(gridcolor="#252D3D", title="RMSE"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
        colorway=["#4FC3A1","#F5A623","#9B59B6"],
    )
    # colour bars by model
    for trace in fig_bar.data:
        trace.marker.color = [MODEL_COLORS.get(m, "#4FC3A1") for m in model_comp["Model"]]
    st.plotly_chart(fig_bar, use_container_width=True)

    # Styled table
    st.markdown("**Metric table**")
    styled = model_comp.set_index("Model").style\
        .format({rmse_col:"{:.6f}", mae_col:"{:.6f}", r2_col:"{:.6f}"})\
        .highlight_max(subset=[r2_col], color="#1e3a2f")\
        .highlight_min(subset=[rmse_col, mae_col], color="#1e3a2f")
    st.dataframe(styled, use_container_width=True)

# ── Right column: CVA vs PCA + feature importance ────────────────────────────
with col_right:
    st.markdown("**CVA vs PCA — R² by model**")

    fig_cvapca = go.Figure()
    r2_col_cva = "R2" if "R2" in cva_pca.columns else "R²"

    for method, color in [("CVA","#4FC3A1"), ("PCA","#F5A623")]:
        subset = cva_pca[cva_pca["SCMI_Method"] == method]
        fig_cvapca.add_trace(go.Bar(
            name=method,
            x=subset["Model"],
            y=subset[r2_col_cva],
            marker_color=color,
            text=subset[r2_col_cva].round(4),
            textposition="outside",
        ))

    fig_cvapca.update_layout(
        barmode="group",
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,31,46,0.6)",
        font_color="#C5CAD6",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#C5CAD6"),
        yaxis=dict(gridcolor="#252D3D", title="R²"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_cvapca, use_container_width=True)

    # ── SHAP global feature importance ────────────────────────────────────────
    st.markdown("**Global SHAP feature importance (XGBoost)**")
    if not shap_imp.empty:
        # Detect value column
        val_col = [c for c in shap_imp.columns if c not in ["feature","settlement","zone_id"]]
        if val_col:
            imp_col = val_col[0]
            # If it has a 'feature' column use it; else use index
            if "feature" in shap_imp.columns:
                fi = shap_imp[["feature", imp_col]].sort_values(imp_col, ascending=True)
                x_vals = fi[imp_col]
                y_vals = fi["feature"]
            else:
                fi = shap_imp.set_index(shap_imp.columns[0])[imp_col].sort_values(ascending=True)
                x_vals = fi.values
                y_vals = fi.index
            colors = ["#4FC3A1" if i >= len(x_vals) - 3 else "#2D5A4E"
                      for i in range(len(x_vals))]
            fig_imp = go.Figure(go.Bar(
                x=list(x_vals), y=list(y_vals),
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig_imp.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,31,46,0.6)",
                font_color="#C5CAD6",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(gridcolor="#252D3D", title="Mean |SHAP|"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

# ── CVA vs PCA full table ─────────────────────────────────────────────────────
with st.expander("Full CVA vs PCA comparison table", expanded=False):
    st.dataframe(cva_pca.set_index("Model"), use_container_width=True)

# ── Key findings ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Key findings**")
fc1, fc2 = st.columns(2)
with fc1:
    st.markdown(f"""
    <div class="finding-box">
        <b>CVA outperforms PCA as SCMI method.</b> CVA-based SCMI produces lower RMSE
        (&lt;0.04) across all three models, while PCA inflates RMSE (&gt;0.48) suggesting
        sensitivity to the PCA rotation choice.
    </div>
    <div class="finding-box">
        <b>Ridge Regression leads on R²</b> ({best_model['R²']:.4f}) despite being the
        simplest model — suggesting the feature-SCMI relationship is largely linear after
        spectral preprocessing.
    </div>
    """, unsafe_allow_html=True)
with fc2:
    st.markdown("""
    <div class="finding-box">
        <b>Road features showed near-zero values</b> in several zones, limiting
        road_density and paved_proportion SHAP contribution. This is a known data gap
        for the pre-intervention period.
    </div>
    <div class="finding-box">
        <b>NDVI and NDBI are the dominant SHAP contributors</b> across settlements —
        consistent with vegetation removal and built-surface expansion being the primary
        physical change signal captured by CVA.
    </div>
    """, unsafe_allow_html=True)
