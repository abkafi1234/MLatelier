import streamlit as st


def setup_page():
    """Initializes page config and injects the global CSS payload."""
    st.set_page_config(
        page_title="MLatelier | Optimizer",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
    <style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ── Theme-adaptive tokens ──────────────────────────────────────────────────
   All colour decisions flow through these variables.
   In dark mode  : Streamlit sets --background-color to ~#0E1117
                   and --secondary-background-color to ~#262730
   In light mode : --background-color → ~#FFFFFF
                   --secondary-background-color → ~#F0F2F6
   Accent colours are intentionally fixed across both themes.
────────────────────────────────────────────────────────────────────────── */
:root {
    --pm-bg:     var(--background-color);
    --pm-bg-card: var(--secondary-background-color);
    --pm-text:   var(--text-color);
    --pm-border: rgba(128, 128, 128, 0.18);
    --pm-muted:  rgba(128, 128, 128, 0.75);
    --pm-blue:   #5B9AFF;
    --pm-green:  #3FB950;
    --pm-purple: #8A63FF;
    --pm-amber:  #D29630;
    --pm-red:    #E05252;
}

/* ── Global Typography ── */
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--pm-bg-card) !important;
    border-right: 1px solid var(--pm-border) !important;
}

/* ── Hero & Status ── */
.hero-title { font-size: 2.5rem; font-weight: 800; letter-spacing: -0.02em; color: var(--pm-text); }
.hero-accent { color: var(--pm-blue); }
.hero-sub { color: var(--pm-muted); font-size: 1.05rem; margin-top: 4px; font-weight: 500; }
.status-pill { border: 1px solid #238636; background: #23863622; color: #3fb950; display: inline-flex; align-items: center; padding: 4px 12px; border-radius: 30px; font-size: 0.8rem; font-weight: 600; float: right; }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: #3fb950; margin-right: 6px; box-shadow: 0 0 8px #3fb950; }

/* ── Metric Cards ── */
.metric-card { background: var(--pm-bg-card); border: 1px solid var(--pm-border); border-radius: 12px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.12); display: flex; flex-direction: column; position: relative; overflow: hidden; }
.metric-card::before { content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 2px; }
.metric-card.cyan::before   { background: var(--pm-blue); }
.metric-card.teal::before   { background: var(--pm-green); }
.metric-card.indigo::before { background: var(--pm-purple); }
.metric-card.amber::before  { background: var(--pm-amber); }

.metric-label { font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: var(--pm-muted); letter-spacing: 0.05em; margin-bottom: 8px; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 2.2rem; font-weight: 700; line-height: 1.1; }
.metric-value.cyan   { color: var(--pm-blue); }
.metric-value.teal   { color: var(--pm-green); }
.metric-value.indigo { color: var(--pm-purple); }
.metric-value.amber  { color: var(--pm-amber); }
.metric-delta { font-size: 0.8rem; font-weight: 500; color: var(--pm-muted); margin-top: 6px; }

/* ── Tabs & Buttons ── */
[data-testid="stTabs"] [data-baseweb="tab"] { color: var(--pm-muted); font-weight: 600; font-size: 0.95rem; }
[data-testid="stTabs"] [aria-selected="true"] { color: var(--pm-blue) !important; border-bottom-color: var(--pm-blue) !important; }
/* Green gradient only for primary (Run) buttons */
button[data-testid="baseButton-primary"] { background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important; color: #fff !important; font-weight: 600; border: none !important; padding: 12px 0; border-radius: 8px; box-shadow: 0 4px 14px rgba(35,134,54,0.3); }
button[data-testid="baseButton-primary"]:hover { opacity: 0.9; transform: translateY(-1px); }
/* Stop / danger buttons (secondary type) */
button[data-testid="baseButton-secondary"] { border: 1px solid rgba(224,82,82,0.45) !important; color: var(--pm-red) !important; border-radius: 8px; font-weight: 600; }
button[data-testid="baseButton-secondary"]:hover { background: rgba(224,82,82,0.1) !important; }
hr { border-color: var(--pm-border); }

/* ── Expanders ── */
[data-testid="stSidebar"] [data-testid="stExpander"] { background-color: transparent !important; border: 1px solid var(--pm-border) !important; border-radius: 8px !important; margin-bottom: 12px !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary { padding: 12px 15px !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover p,
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover svg { color: var(--pm-blue) !important; fill: var(--pm-blue) !important; }
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] { border-top: 1px solid var(--pm-border) !important; }

/* ── Multiselect selected tags ── */
[data-testid="stSidebar"] span[data-baseweb="tag"] { background-color: #E04343 !important; border: none !important; border-radius: 4px !important; }
[data-testid="stSidebar"] span[data-baseweb="tag"] span { color: #FFFFFF !important; font-weight: 600 !important; }
[data-testid="stSidebar"] span[data-baseweb="tag"] svg { fill: #FFFFFF !important; }

/* ── Animations ── */
@keyframes slideUpFade {
    from { opacity: 0; transform: translateY(15px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Error Box ── */
.error-box {
    background: rgba(200, 30, 30, 0.08);
    border: 1px solid rgba(200, 50, 50, 0.35);
    border-left: 4px solid #FF4D4D;
    padding: 16px;
    border-radius: 12px;
    margin-top: 10px;
    animation: slideUpFade 0.3s ease-out;
}
.error-title { margin: 0; font-size: 1rem; font-weight: 600; color: #FF6B6B; }
.error-text  { margin-top: 8px; font-size: 0.92rem; line-height: 1.5; color: var(--pm-text); }
</style>
    """,
        unsafe_allow_html=True,
    )


def initialize_session_state():
    """No-op: per-engine state is now initialised directly in app.py."""
    pass


def show_ai_dependency_error():
    st.markdown(
        '<div style="'
        "background: linear-gradient(135deg, #1A1A24, #121214);"
        "border: 1px solid #2A2A2D;"
        "border-left: 3px solid #8A63FF;"
        "padding: 15px 20px;"
        "border-radius: 12px 12px 12px 0px;"
        "margin-bottom: 12px;"
        "margin-right: 15%;"
        "box-shadow: 0 4px 15px rgba(0,0,0,0.2);"
        '">'
        '<p style="'
        "color: #8A63FF;"
        "font-size: 0.75rem;"
        "font-weight: 700;"
        "text-transform: uppercase;"
        "margin-bottom: 5px;"
        '">✨ MLatelier Assistant</p>'
        '<p style="'
        "color: #D1D1D6;"
        "font-size: 0.95rem;"
        "line-height: 1.6;"
        "margin: 0 0 10px 0;"
        '">The <code style="'
        "background: rgba(138,99,255,0.15);"
        "color: #8A63FF;"
        "padding: 2px 6px;"
        "border-radius: 4px;"
        "font-size: 0.85rem;"
        '">google-genai</code> package is not installed. Please install it to enable AI chat.</p>'
        '<div style="'
        "background: rgba(138,99,255,0.08);"
        "border: 1px solid rgba(138,99,255,0.25);"
        "padding: 6px 12px;"
        "border-radius: 8px;"
        "display: inline-block;"
        "font-family: monospace;"
        "font-size: 0.85rem;"
        "color: #8A63FF;"
        '">pip install google-genai</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    
    return
