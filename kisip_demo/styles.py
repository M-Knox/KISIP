"""
Shared CSS injected on every page.
Single source of truth — import and call inject_css() at the top of each page.
"""
import streamlit as st

TEAL   = "#4FC3A1"
AMBER  = "#F5A623"
MUTED  = "#8B95A8"
TEXT   = "#E2E5EC"
SURFACE = "#161B27"
CARD   = "#1C2333"
BORDER = "#252D3D"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #E2E5EC;
}

/* ── Typography scale ── */
.pg-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    color: #8B95A8;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.pg-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #E2E5EC;
    line-height: 1.2;
    margin-bottom: 0.25rem;
}
.pg-sub {
    font-size: 0.9rem;
    color: #8B95A8;
    line-height: 1.6;
    margin-bottom: 1.75rem;
    max-width: 680px;
}

/* ── Stat block — borderless, size does the work ── */
.stat-block { padding: 0.5rem 0 1rem 0; }
.stat-block .s-val {
    font-size: 2rem;
    font-weight: 700;
    color: #E2E5EC;
    line-height: 1;
    letter-spacing: -0.02em;
}
.stat-block .s-label {
    font-size: 0.75rem;
    color: #8B95A8;
    margin-top: 0.2rem;
}

/* ── Divider row of stats ── */
.stat-row {
    display: flex;
    gap: 0;
    border-top: 1px solid #252D3D;
    border-bottom: 1px solid #252D3D;
    margin-bottom: 1.75rem;
    flex-wrap: wrap;
}
.stat-cell {
    flex: 1;
    min-width: 100px;
    padding: 1rem 1.5rem;
    border-right: 1px solid #252D3D;
}
.stat-cell:last-child { border-right: none; }
.stat-cell .sc-val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #E2E5EC;
    line-height: 1;
    letter-spacing: -0.02em;
}
.stat-cell .sc-accent { color: #4FC3A1; }
.stat-cell .sc-label {
    font-size: 0.72rem;
    color: #8B95A8;
    margin-top: 0.25rem;
}

/* ── Subtle card — light surface, no heavy border ── */
.s-card {
    background: #1C2333;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.s-card .sc-key {
    font-size: 0.7rem;
    color: #8B95A8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.15rem;
    font-family: 'JetBrains Mono', monospace;
}
.s-card .sc-val {
    font-size: 1.4rem;
    font-weight: 600;
    color: #E2E5EC;
}
.s-card .sc-sub {
    font-size: 0.75rem;
    color: #8B95A8;
    margin-top: 0.1rem;
}

/* ── Tier indicator — dot + text only, no box ── */
.tier-line {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.25rem;
}
.tier-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.tier-text {
    font-size: 0.95rem;
    font-weight: 600;
    color: #E2E5EC;
}
.tier-sub {
    font-size: 0.8rem;
    color: #8B95A8;
    margin-left: 1.6rem;
    margin-top: -0.9rem;
    margin-bottom: 0.75rem;
}

/* ── Inline data tags ── */
.dtag {
    display: inline-block;
    background: rgba(79,195,161,0.08);
    color: #4FC3A1;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 7px;
    border-radius: 3px;
    margin: 2px;
}

/* ── Section rule ── */
.section-rule {
    font-size: 0.72rem;
    font-weight: 600;
    color: #8B95A8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #252D3D;
    padding-bottom: 0.4rem;
    margin-bottom: 0.75rem;
    margin-top: 1.25rem;
}

/* ── Finding strip ── */
.finding {
    border-left: 2px solid #4FC3A1;
    padding: 0.5rem 0.9rem;
    margin-bottom: 0.75rem;
    font-size: 0.875rem;
    color: #C5CAD6;
    line-height: 1.55;
}
.finding b { color: #E2E5EC; font-weight: 600; }

/* ── Legend row ── */
.leg-row { display:flex; align-items:center; gap:0.5rem; font-size:0.82rem; color:#8B95A8; margin-bottom:0.4rem; }
.leg-dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }

/* ── Rank list ── */
.rank-item {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid #252D3D;
    font-size: 0.875rem;
}
.rank-item:last-child { border-bottom: none; }
.rank-n {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #8B95A8;
    min-width: 20px;
}
.rank-name { color: #E2E5EC; flex: 1; }
.rank-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #4FC3A1;
}

/* ── Nav list on home ── */
.nav-item {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid #252D3D;
}
.nav-item:last-child { border-bottom: none; }
.nav-n {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #4FC3A1;
    padding-top: 0.15rem;
    min-width: 24px;
}
.nav-title { font-size: 0.9rem; font-weight: 500; color: #E2E5EC; }
.nav-desc  { font-size: 0.8rem; color: #8B95A8; margin-top: 0.1rem; }

/* Account for Streamlit multipage header bar */
.block-container { padding-top: 3.5rem !important; }
"""

def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)
