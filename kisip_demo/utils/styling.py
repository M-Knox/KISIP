"""Global minimal dark-theme styles."""

import streamlit as st

from utils.constants import ACCENT, BG, SURFACE, TEXT, TEXT_MUTED


def inject_styles() -> None:
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

.block-container {{
    padding-top: 2rem;
    max-width: 1200px;
}}

/* ── Page header ── */
.page-header {{
    margin-bottom: 2rem;
}}
.page-kicker {{
    font-size: 0.8125rem;
    color: {TEXT_MUTED};
    margin-bottom: 0.35rem;
}}
.page-title {{
    font-size: 1.75rem;
    font-weight: 700;
    color: {TEXT};
    line-height: 1.2;
    margin: 0 0 0.5rem 0;
}}
.page-lead {{
    font-size: 0.9375rem;
    color: {TEXT_MUTED};
    line-height: 1.55;
    max-width: 52rem;
    margin: 0;
}}

/* ── Metrics ── */
.metrics-row {{
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
    margin: 2rem 0;
    padding: 1.5rem 0;
    border-top: 1px solid rgba(255,255,255,0.06);
    border-bottom: 1px solid rgba(255,255,255,0.06);
}}
.metric-block {{
    min-width: 120px;
}}
.metric-label {{
    font-size: 0.8125rem;
    color: {TEXT_MUTED};
    margin-bottom: 0.25rem;
}}
.metric-value {{
    font-size: 1.75rem;
    font-weight: 700;
    color: {TEXT};
    line-height: 1.1;
}}
.metric-hint {{
    font-size: 0.8125rem;
    color: {TEXT_MUTED};
    margin-top: 0.35rem;
    line-height: 1.4;
}}

/* ── Status dot (replaces tier badge box) ── */
.status-line {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9375rem;
    color: {TEXT};
    margin-bottom: 1.25rem;
}}
.status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}}
.dot-green {{ background: {ACCENT}; }}
.dot-amber {{ background: #F5A623; }}
.dot-red {{ background: #E74C3C; }}

/* ── Section labels ── */
.section-title {{
    font-size: 0.9375rem;
    font-weight: 600;
    color: {TEXT};
    margin: 1.5rem 0 0.75rem 0;
}}
.section-caption {{
    font-size: 0.8125rem;
    color: {TEXT_MUTED};
    margin-top: 0.35rem;
}}

/* ── List rows (ranking, legend) ── */
.list-row {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.625rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 0.875rem;
    color: {TEXT};
}}
.list-row:last-child {{ border-bottom: none; }}
.list-rank {{
    font-size: 0.8125rem;
    color: {TEXT_MUTED};
    width: 1.75rem;
}}
.list-value {{
    margin-left: auto;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}}

/* ── Nav links on home ── */
.nav-list {{
    margin-top: 1rem;
}}
.nav-item {{
    padding: 0.75rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 0.875rem;
    color: {TEXT_MUTED};
}}
.nav-item strong {{
    color: {TEXT};
    font-weight: 600;
}}

/* ── Tags (features) ── */
.tag {{
    display: inline-block;
    font-size: 0.75rem;
    color: {TEXT_MUTED};
    padding: 0.2rem 0.5rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
    background: {SURFACE};
    border-radius: 4px;
}}

/* ── Plain summary box ── */
.summary-box {{
    background: {SURFACE};
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.5rem;
    font-size: 0.9375rem;
    line-height: 1.6;
    color: {TEXT_MUTED};
}}
.summary-box em {{
    color: {TEXT};
    font-style: normal;
    font-weight: 500;
}}

/* ── Hide Streamlit chrome noise ── */
div[data-testid="stExpander"] details {{
    border: none;
    background: transparent;
}}
div[data-testid="stExpander"] summary {{
    font-size: 0.875rem;
    color: {TEXT_MUTED};
}}

hr {{
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 2rem 0;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def page_header(kicker: str, title: str, lead: str) -> None:
    st.markdown(
        f"""
<div class="page-header">
    <div class="page-kicker">{kicker}</div>
    <h1 class="page-title">{title}</h1>
    <p class="page-lead">{lead}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def plotly_layout(**overrides) -> dict:
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,31,46,0.4)",
        font=dict(color=TEXT, size=12),
        margin=dict(l=8, r=8, t=32, b=8),
    )
    base.update(overrides)
    return base
