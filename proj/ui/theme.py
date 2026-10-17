"""Візуальна система інтерфейсу: CSS, палітра, шаблон графіків і дрібні компоненти."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from .help import tip

INK = "#0f172a"
INK_2 = "#475569"
INK_3 = "#94a3b8"
GRID = "#e9edf3"
SURFACE = "#ffffff"
ACCENT = "#4f46e5"

# Кольори закріплені за методом (сутністю), а не за позицією в рейтингу.
METHOD_COLORS = {
    "proposed": "#2a78d6",
    "adaptive": "#0d9488",
    "weighted": "#a855f7",
    "cleanlab": "#eb6834",
    "loss": "#1baf7a",
    "margin": "#eda100",
    "aum": "#b45309",
    "dynamics": "#e87ba4",
    "semantic": "#008300",
    "oracle": "#6250d6",
    "random": "#a3a3a0",
    "noisy": "#64748b",
    "C": "#1baf7a", "D": "#eda100", "S": "#008300",
}
CLEAN_COLOR = "#94a3b8"
CORRUPT_COLOR = "#e34948"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}
code, pre, .mono { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }

.stApp { background: radial-gradient(1200px 600px at 100% -10%, #eef2ff 0%, rgba(238,242,255,0) 60%),
                     radial-gradient(900px 500px at -10% 10%, #ecfeff 0%, rgba(236,254,255,0) 55%), #f6f7fb; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1320px; }
header[data-testid="stHeader"] { background: transparent; }

/* Бічна панель */
section[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
section[data-testid="stSidebar"] * { color: #cbd5e1; }
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] { border-radius: 10px; margin: 2px 0; }
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover { background: #1e293b; }
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] {
  background: linear-gradient(90deg, rgba(79,70,229,.35), rgba(79,70,229,.08)); }
section[data-testid="stSidebar"] a[aria-current="page"] span { color: #ffffff !important; font-weight: 600; }
[data-testid="stSidebarNavSeparator"] { border-color: #1e293b; }
[data-testid="stNavSectionHeader"] { color: #64748b !important; text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; }
.brand { display:flex; gap:.7rem; align-items:center; padding: .3rem 0 1.1rem; }
.brand-logo { width: 40px; height: 40px; border-radius: 12px; display:grid; place-items:center;
  background: linear-gradient(135deg,#6366f1,#06b6d4); color:#fff !important; font-weight:800; font-size:1.05rem;
  box-shadow: 0 6px 18px rgba(99,102,241,.45); }
.brand-name { font-weight: 800; font-size: 1.12rem; color: #f8fafc !important; letter-spacing: -.01em; }
.brand-sub { font-size: .72rem; color: #94a3b8 !important; line-height: 1.25; }
.side-note { font-size: .74rem; color:#64748b !important; border-top:1px solid #1e293b; padding-top:.8rem;
  position: fixed; bottom: 1.2rem; width: 230px; }
[data-testid="stSidebarContent"] { display: flex; flex-direction: column; }
[data-testid="stSidebarHeader"] { order: -2; height: 1.2rem; min-height: 1.2rem; }
[data-testid="stSidebarUserContent"] { order: -1; padding-bottom: 0; }

/* Герой */
.hero { position: relative; overflow: hidden; border-radius: 22px; padding: 2.1rem 2.3rem;
  background: linear-gradient(120deg, #1e1b4b 0%, #312e81 45%, #0e7490 100%); color: #fff;
  box-shadow: 0 20px 45px -20px rgba(49,46,129,.55); margin-bottom: 1.4rem; }
.hero:after { content:""; position:absolute; right:-80px; top:-80px; width:320px; height:320px; border-radius:50%;
  background: radial-gradient(circle, rgba(255,255,255,.18), rgba(255,255,255,0) 70%); }
.hero h1 { color:#fff !important; font-size: 2.05rem; line-height:1.15; font-weight: 800; letter-spacing:-.02em; margin: .35rem 0 .6rem; padding:0; }
.hero p { color: #e0e7ff; font-size: 1.02rem; max-width: 860px; margin: 0; }
.eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; font-weight: 700; color: #a5f3fc; }
.chips { display:flex; flex-wrap: wrap; gap:.45rem; margin-top: 1.1rem; }
.chip { font-size: .78rem; padding: .28rem .7rem; border-radius: 999px; background: rgba(255,255,255,.12);
  border: 1px solid rgba(255,255,255,.22); color: #f1f5f9; }

/* Заголовок сторінки */
.page-head { margin-bottom: 1.2rem; }
.page-head h1 { font-size: 1.75rem; font-weight: 800; letter-spacing: -.02em; color: #0f172a; margin: .15rem 0 .25rem; padding: 0; }
.page-head p { color: #475569; margin: 0; max-width: 900px; }
.page-head .eyebrow { color: #4f46e5; }

/* Картки */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div.card-marker),
div[class*="st-key-card"] { background: #ffffff; border: 1px solid #e6e9f0 !important; border-radius: 16px !important;
  box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -16px rgba(15,23,42,.18); padding: 1.1rem 1.2rem 1rem; }
.card-title { display:flex; align-items:center; gap:.55rem; font-weight: 700; color:#0f172a; font-size: 1rem; margin-bottom: .2rem; }
.card-title .num { width:26px; height:26px; border-radius: 8px; display:grid; place-items:center; font-size:.78rem;
  background:#eef2ff; color:#4f46e5; font-weight:800; }
.card-sub { color:#64748b; font-size:.84rem; margin-bottom:.6rem; }

.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: .9rem; margin: .3rem 0 1.2rem; }
.kpi { background:#fff; border:1px solid #e6e9f0; border-radius: 16px; padding: 1rem 1.1rem;
  box-shadow: 0 8px 24px -18px rgba(15,23,42,.25); position: relative; overflow: hidden; }
.kpi:before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background: var(--kc, #4f46e5); }
.kpi .lbl { color:#64748b; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing:.05em; }
.kpi .val { color:#0f172a; font-size: 1.75rem; font-weight: 800; letter-spacing: -.02em; margin-top: .15rem; }
.kpi .sub { color:#64748b; font-size: .8rem; margin-top: .1rem; }
.kpi .pos { color:#047857; font-weight:600; } .kpi .neg { color:#b91c1c; font-weight:600; }

.formula { background: #f8fafc; border:1px dashed #cbd5e1; border-radius: 12px; padding: .7rem .9rem; margin:.4rem 0;
  font-family: 'JetBrains Mono', monospace; font-size: .86rem; color:#0f172a; }
.feature { background:#fff; border:1px solid #e6e9f0; border-radius:16px; padding:1.1rem 1.2rem; height:100%;
  box-shadow: 0 8px 24px -18px rgba(15,23,42,.25); }
.feature .ico { width:38px; height:38px; border-radius: 11px; display:grid; place-items:center; color:#fff; font-weight:800; margin-bottom:.6rem; }
.feature h4 { margin:0 0 .3rem; font-size: 1rem; color:#0f172a; padding:0; }
.feature p { margin:0; color:#475569; font-size:.88rem; }

.steps { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap:.7rem; }
.step { background:#fff; border:1px solid #e6e9f0; border-radius:14px; padding:.85rem .9rem; position:relative; }
.step .n { font-size:.7rem; font-weight:800; color:#4f46e5; letter-spacing:.08em; }
.step .t { font-weight:700; color:#0f172a; font-size:.92rem; margin:.15rem 0 .2rem; }
.step .d { color:#64748b; font-size:.8rem; line-height:1.35; }

.note { border-radius: 12px; padding: .75rem 1rem; font-size: .88rem; margin: .4rem 0 .8rem; border:1px solid; }
.note.info { background:#eff6ff; border-color:#bfdbfe; color:#1e3a8a; }
.note.warn { background:#fffbeb; border-color:#fde68a; color:#78350f; }
.note.ok { background:#ecfdf5; border-color:#a7f3d0; color:#064e3b; }

.section-title { font-size: 1.08rem; font-weight: 750; color:#0f172a; margin: 1.1rem 0 .15rem; }
.section-sub { color:#64748b; font-size:.86rem; margin-bottom:.5rem; }

/* Вкладки */
.stTabs [data-baseweb="tab-list"] { gap: .25rem; background:#eef1f6; padding: .3rem; border-radius: 12px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { border-radius: 9px; padding: .45rem .9rem; height:auto; background: transparent; }
.stTabs [aria-selected="true"] { background: #fff !important; box-shadow: 0 1px 3px rgba(15,23,42,.12); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

/* Кнопки */
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: linear-gradient(135deg,#4f46e5,#0891b2); border: 0; font-weight: 650; box-shadow: 0 8px 20px -10px rgba(79,70,229,.7); }
.stButton > button, .stDownloadButton > button { border-radius: 11px; }
div[data-testid="stProgress"] > div > div > div > div { background: linear-gradient(90deg,#6366f1,#06b6d4); }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def sidebar_brand() -> None:
    with st.sidebar:
        st.markdown(
            """<div class="brand"><div class="brand-logo">CT</div>
            <div><div class="brand-name">CleanText</div>
            <div class="brand-sub">Виявлення помилкових міток<br>україномовних текстів</div></div></div>""",
            unsafe_allow_html=True)


def sidebar_footer(device: str) -> None:
    with st.sidebar:
        st.markdown(f"""<div class="side-note">Обчислювальний пристрій: <b>{device.upper()}</b><br>
        Результати зберігаються в теці <span class="mono">runs/</span></div>""", unsafe_allow_html=True)


def page_head(eyebrow: str, title: str, text: str = "") -> None:
    st.markdown(f"""<div class="page-head"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1>
    <p>{text}</p></div>""", unsafe_allow_html=True)


def card_title(num: str, title: str, sub: str = "") -> None:
    st.markdown(f"""<div class="card-title"><span class="num">{num}</span>{title}</div>
    {f'<div class="card-sub">{sub}</div>' if sub else ''}""", unsafe_allow_html=True)


def kpis(items: list[dict]) -> None:
    """items: {label, value, sub, color, delta(float|None)}"""
    cells = []
    for it in items:
        sub = it.get("sub", "")
        d = it.get("delta")
        if d is not None and d == d:
            cls = "pos" if d >= 0 else "neg"
            sub = f'<span class="{cls}">{"+" if d >= 0 else "−"}{abs(d):.3f}</span> {sub}'
        cells.append(f"""<div class="kpi" style="--kc:{it.get('color', ACCENT)}"><div class="lbl">{it['label']}</div>
        <div class="val">{it['value']}</div><div class="sub">{sub}</div></div>""")
    st.markdown(f'<div class="kpis">{"".join(cells)}</div>', unsafe_allow_html=True)


def note(text: str, kind: str = "info") -> None:
    st.markdown(f'<div class="note {kind}">{text}</div>', unsafe_allow_html=True)


def section(title: str, sub: str = "", hint: str | None = None) -> None:
    """Заголовок розділу; hint — ключ підказки «?» (ui/locales)."""
    html_ = f'<div class="section-title">{title}</div>' + (f'<div class="section-sub">{sub}</div>' if sub else "")
    text = tip(hint) if hint else None
    if text is None:
        st.markdown(html_, unsafe_allow_html=True)
        return
    c1, c2 = st.columns([30, 1], vertical_alignment="bottom")
    c1.markdown(html_, unsafe_allow_html=True)
    c2.markdown(":material/help:", help=text)


def esc(s) -> str:
    return html.escape(str(s))


# ---------------------------------------------------------------- Plotly
def _template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        colorway=SERIES,
        margin=dict(l=10, r=10, t=48, b=10),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", y=0.98),
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor="#cbd5e1", ticks="", tickfont=dict(color=INK_2),
                   title=dict(font=dict(color=INK_2, size=12))),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", tickfont=dict(color=INK_2),
                   title=dict(font=dict(color=INK_2, size=12))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=INK_2, size=12),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0f172a", font=dict(color="#f8fafc", family="Inter"), bordercolor="#0f172a"),
        bargap=0.28,
    )
    return t


pio.templates["cleantext"] = _template()
pio.templates.default = "plotly_white+cleantext"

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
               "toImageButtonOptions": {"format": "png", "scale": 3, "filename": "cleantext_chart"}}


def show(fig, key=None, height=None) -> None:
    fig.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)")
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, theme=None, config=PLOT_CONFIG, key=key)
