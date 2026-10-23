import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import html

import pandas as pd
import streamlit as st

from ticketclf.predict import TicketClassifier

st.set_page_config(
    page_title="TicketIQ | Support triage",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PRIORITY = {
    "High": {
        "main": "#ff5d73",
        "soft": "rgba(255, 93, 115, .14)",
        "label": "Urgent",
        "route": "Escalate to the security and fraud queue now.",
        "sla": "Reply target: under 15 minutes",
    },
    "Medium": {
        "main": "#ffb454",
        "soft": "rgba(255, 180, 84, .14)",
        "label": "Standard",
        "route": "Send to the general support queue.",
        "sla": "Reply target: within 1 business day",
    },
    "Low": {
        "main": "#3ddc97",
        "soft": "rgba(61, 220, 151, .14)",
        "label": "Low",
        "route": "Answer with a help-centre article or saved reply.",
        "sla": "Reply target: within 2 business days",
    },
}
EXAMPLES = {
    "Stolen card": "My card was stolen and I see payments I did not make",
    "Top up": "How do I top up my account?",
    "Card delivery": "I still haven't received my card, where is it?",
    "Double charge": "Why was I charged twice for the same purchase?",
}
MAX_BATCH_ROWS = 500

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;700&display=swap');

:root {
    color-scheme: dark;
    --ink: #060f14;
    --ink-2: #0b1a21;
    --glass: rgba(255, 255, 255, .045);
    --glass-2: rgba(255, 255, 255, .075);
    --line: rgba(160, 220, 230, .14);
    --line-strong: rgba(160, 220, 230, .28);
    --text: #e9f2f4;
    --muted: #8da6ad;
    --brand-a: #5eead4;
    --brand-b: #818cf8;
    --display: 'Bricolage Grotesque', 'DM Sans', system-ui, sans-serif;
    --body: 'DM Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}

html, body, .stApp, [data-testid="stAppViewContainer"] { font-family: var(--body); }
.stApp {
    color: var(--text);
    background:
        radial-gradient(900px 520px at 8% -8%, rgba(94, 234, 212, .13), transparent 60%),
        radial-gradient(800px 560px at 100% 0%, rgba(129, 140, 248, .15), transparent 60%),
        radial-gradient(700px 500px at 50% 115%, rgba(94, 234, 212, .07), transparent 60%),
        var(--ink);
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] {
    max-width: 100% !important;
    width: 100% !important;
    padding: 1.4rem clamp(1rem, 2.6vw, 3.2rem) 2.4rem !important;
}
@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] { padding: 1rem 1rem 2rem !important; }
}

.hero {
    position: relative; overflow: hidden;
    padding: 2.6rem 2.8rem 2.4rem; border-radius: 22px;
    display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, .75fr); gap: 2.4rem; align-items: center;
    background: linear-gradient(135deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
    border: 1px solid var(--line);
    backdrop-filter: blur(14px);
    margin-bottom: 1.4rem;
    animation: rise .7s cubic-bezier(.2,.8,.2,1) both;
}
.hero::before {
    content: ""; position: absolute; inset: -40% -10% auto auto;
    width: 520px; height: 520px; border-radius: 50%;
    background: conic-gradient(from 120deg, rgba(94,234,212,.35), rgba(129,140,248,.35), rgba(94,234,212,.0), rgba(94,234,212,.35));
    filter: blur(70px); opacity: .55;
    animation: drift 18s linear infinite;
}
.hero::after {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background-image: linear-gradient(var(--line) 1px, transparent 1px), linear-gradient(90deg, var(--line) 1px, transparent 1px);
    background-size: 46px 46px; opacity: .18;
    mask-image: radial-gradient(ellipse at 85% 20%, #000 0%, transparent 65%);
    -webkit-mask-image: radial-gradient(ellipse at 85% 20%, #000 0%, transparent 65%);
}
.hero > * { position: relative; z-index: 1; }
@media (max-width: 1000px) { .hero { grid-template-columns: 1fr; padding: 1.8rem 1.4rem; } }
.flow { display: grid; gap: .7rem; }
.flow-step {
    display: flex; align-items: center; gap: .9rem; padding: .85rem 1rem;
    border: 1px solid var(--line); border-radius: 14px; background: rgba(6, 15, 20, .45);
}
.flow-num {
    width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center; flex: 0 0 auto;
    font-family: var(--display); font-weight: 800; color: #04201c;
    background: linear-gradient(135deg, var(--brand-a), var(--brand-b));
}
.flow-title { font-weight: 700; color: #f4fafb; font-size: .95rem; }
.flow-copy { color: var(--muted); font-size: .84rem; margin-top: .05rem; }
.brand-row { display: flex; align-items: center; gap: .65rem; margin-bottom: 1.1rem; }
.brand-mark {
    width: 38px; height: 38px; border-radius: 11px; display: grid; place-items: center;
    font-size: 1.15rem; color: #04201c;
    background: linear-gradient(135deg, var(--brand-a), var(--brand-b));
    box-shadow: 0 8px 24px rgba(94, 234, 212, .28);
}
.brand-name { font-family: var(--display); font-weight: 800; font-size: 1.15rem; letter-spacing: -.01em; }
.brand-tag { color: var(--muted); font-size: .86rem; }
.hero-title {
    font-family: var(--display); font-weight: 800;
    font-size: clamp(2.2rem, 4.4vw, 3.9rem); line-height: 1.02; letter-spacing: -.035em;
    margin: 0 0 .9rem; max-width: 16ch; color: #f4fafb;
}
.hero-copy { max-width: 62ch; color: #b6c9ce; font-size: 1.04rem; line-height: 1.7; }
.hero-pills { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.5rem; }
.hero-pill {
    border: 1px solid var(--line-strong); background: rgba(255,255,255,.04);
    padding: .38rem .8rem; border-radius: 999px; font-size: .8rem; color: #d3e3e6;
}

.stat {
    background: var(--glass); border: 1px solid var(--line); border-radius: 16px;
    padding: 1rem 1.15rem; min-height: 118px;
    display: flex; flex-direction: column; justify-content: center;
}
.stat-label { color: var(--muted); font-size: .8rem; font-weight: 500; }
.stat-value { font-family: var(--display); font-size: 1.75rem; font-weight: 800; letter-spacing: -.02em; margin-top: .15rem; color: #f4fafb; }
.stat-note { color: var(--muted); font-size: .76rem; margin-top: .1rem; }

.panel-head { margin: .2rem 0 .9rem; }
.panel-title { font-family: var(--display); font-size: 1.35rem; font-weight: 700; letter-spacing: -.02em; color: #f4fafb; }
.panel-copy { color: var(--muted); font-size: .92rem; line-height: 1.55; margin-top: .2rem; }
.mini-title { color: var(--muted); font-size: .84rem; font-weight: 500; margin: 1.1rem 0 .5rem; }

.result {
    position: relative; overflow: hidden;
    background: linear-gradient(160deg, var(--glass-2), var(--glass));
    border: 1px solid var(--line-strong); border-radius: 22px; padding: 1.5rem 1.6rem;
    animation: rise .5s cubic-bezier(.2,.8,.2,1) both;
}
.result::before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: var(--accent);
    box-shadow: 0 0 28px 2px var(--accent);
}
.result-top { display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; }
.result-label { font-size: .84rem; color: var(--muted); font-weight: 500; }
.chip {
    display: inline-flex; align-items: center; gap: .45rem;
    border-radius: 999px; padding: .38rem .8rem; font-size: .82rem; font-weight: 700;
    color: var(--accent); background: var(--accent-soft);
    border: 1px solid color-mix(in srgb, var(--accent) 40%, transparent);
}
.chip .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); animation: pulse 1.8s ease-in-out infinite; }
.result-body { display: flex; align-items: center; gap: 1.5rem; margin-top: 1rem; flex-wrap: wrap; }
.result-intent {
    flex: 1 1 220px; min-width: 0;
    font-family: var(--display); font-size: clamp(1.7rem, 2.6vw, 2.35rem); line-height: 1.1;
    font-weight: 800; letter-spacing: -.03em; color: #f6fbfc;
    text-transform: capitalize; overflow-wrap: anywhere;
}
.ring { width: 132px; height: 132px; flex: 0 0 auto; }
.ring-bg { fill: none; stroke: rgba(255,255,255,.09); stroke-width: 10; }
.ring-fg {
    fill: none; stroke: var(--accent); stroke-width: 10; stroke-linecap: round;
    stroke-dasharray: 364.42; stroke-dashoffset: var(--off);
    filter: drop-shadow(0 0 6px var(--accent));
    animation: draw 1.1s cubic-bezier(.2,.8,.2,1) both;
}
.ring-num { fill: #f6fbfc; font-family: var(--display); font-weight: 800; font-size: 28px; text-anchor: middle; }
.ring-cap { fill: var(--muted); font-size: 10px; text-anchor: middle; }
.note { color: var(--muted); font-size: .8rem; margin-top: 1rem; }

.route {
    margin-top: 1rem; border-radius: 16px; padding: 1rem 1.2rem;
    border: 1px solid var(--line); background: var(--glass);
    display: flex; gap: .9rem; align-items: flex-start;
}
.route-ico {
    width: 36px; height: 36px; border-radius: 10px; display: grid; place-items: center; flex: 0 0 auto;
    font-size: 1.05rem; background: var(--tone-soft); color: var(--tone);
}
.route-title { font-weight: 700; color: #f4fafb; }
.route-copy { color: #b6c9ce; font-size: .9rem; line-height: 1.55; margin-top: .1rem; }

.bars { display: grid; gap: .7rem; }
.bar-row { display: grid; grid-template-columns: minmax(120px, 38%) 1fr 48px; gap: .8rem; align-items: center; }
.bar-name { font-size: .88rem; color: #d3e3e6; text-transform: capitalize; overflow-wrap: anywhere; }
.bar-track { height: 8px; border-radius: 999px; background: rgba(255,255,255,.07); overflow: hidden; }
.bar-fill {
    height: 100%; border-radius: 999px; transform-origin: left;
    background: linear-gradient(90deg, var(--brand-a), var(--brand-b));
    animation: grow .8s cubic-bezier(.2,.8,.2,1) both;
}
.bar-row.first .bar-name { color: #f6fbfc; font-weight: 700; }
.bar-val { text-align: right; color: var(--muted); font-size: .82rem; font-variant-numeric: tabular-nums; }

.empty {
    min-height: 430px; border: 1px dashed var(--line-strong); border-radius: 22px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    text-align: center; padding: 2rem; background: var(--glass);
}
.empty-ico {
    width: 66px; height: 66px; border-radius: 20px; display: grid; place-items: center; font-size: 1.9rem;
    color: #04201c; background: linear-gradient(135deg, var(--brand-a), var(--brand-b));
    box-shadow: 0 12px 34px rgba(129, 140, 248, .3); margin-bottom: 1rem;
}
.empty-title { font-family: var(--display); font-size: 1.25rem; font-weight: 700; color: #f4fafb; }
.empty-copy { max-width: 34ch; color: var(--muted); line-height: 1.6; margin-top: .4rem; }

/* ---------- Streamlit widgets ---------- */
div[data-baseweb="tab-list"] {
    gap: .3rem; background: var(--glass); border: 1px solid var(--line);
    border-radius: 14px; padding: .3rem; width: fit-content; margin-bottom: 1rem;
}
button[role="tab"] {
    border-radius: 10px !important; padding: .5rem 1.1rem !important; height: auto !important;
    color: var(--muted) !important; font-weight: 500 !important;
}
button[role="tab"][aria-selected="true"] { background: var(--glass-2) !important; color: #f4fafb !important; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none !important; }

div.stButton > button, div.stDownloadButton > button {
    border-radius: 12px; border: 1px solid var(--line-strong); min-height: 2.7rem;
    font-weight: 500; color: var(--text); background: var(--glass);
    transition: border-color .15s ease, background .15s ease, transform .15s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    border-color: var(--brand-a); color: #fff; background: var(--glass-2);
}
div.stButton > button:active { transform: scale(.985); }
div.stButton > button:focus-visible, div.stDownloadButton > button:focus-visible,
button[role="tab"]:focus-visible { outline: 2px solid var(--brand-a) !important; outline-offset: 2px; }
div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {
    border: 0; color: #04201c; font-weight: 700;
    background: linear-gradient(135deg, var(--brand-a), var(--brand-b));
    box-shadow: 0 10px 28px rgba(94, 234, 212, .22);
}
div.stButton > button[kind="primary"]:hover { filter: brightness(1.08); color: #04201c; }
div.stButton > button:disabled { opacity: .45; }

div.stTextArea textarea {
    border-radius: 14px; border: 1px solid var(--line-strong); background: rgba(6, 15, 20, .7);
    color: var(--text); font-size: 1rem; line-height: 1.65; padding: 1rem 1.1rem;
}
div.stTextArea textarea:focus { border-color: var(--brand-a); box-shadow: 0 0 0 3px rgba(94, 234, 212, .18); }
div[data-baseweb="select"] > div, [data-testid="stFileUploaderDropzone"] {
    background: rgba(6, 15, 20, .7); border-color: var(--line-strong); border-radius: 14px;
}
label, .stMarkdown, [data-testid="stCaptionContainer"] { color: var(--text); }
[data-testid="stCaptionContainer"] { color: var(--muted); }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--line); }
.footer { text-align: center; color: var(--muted); font-size: .8rem; padding: 2rem 0 .2rem; }

@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes drift { to { transform: rotate(360deg); } }
@keyframes draw { from { stroke-dashoffset: 364.42; } }
@keyframes grow { from { transform: scaleX(0); } }
@keyframes pulse { 50% { opacity: .35; } }
@media (prefers-reduced-motion: reduce) {
    .hero, .hero::before, .result, .ring-fg, .bar-fill, .chip .dot { animation: none !important; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def h(markup: str) -> str:
    return "".join(line.strip() for line in markup.splitlines())


def pretty(intent: str) -> str:
    return intent.rstrip("?").replace("_", " ").strip()


def palette_for(priority: str) -> dict:
    return PRIORITY.get(priority, PRIORITY["Medium"])


def use_example(example: str) -> None:
    st.session_state.ticket_text = example
    st.session_state.ticket_result = None


def log_result(text: str, res: dict) -> None:
    st.session_state.history.insert(
        0,
        {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Message": text,
            "Intent": pretty(str(res.get("intent", "Unknown"))),
            "Priority": str(res.get("priority", "Medium")),
            "Confidence": round(float(res.get("confidence", 0.0)), 4),
            "Needs review": bool(res.get("needs_human_review")),
        },
    )


@st.cache_resource(show_spinner=False)
def load_classifier() -> TicketClassifier:
    return TicketClassifier()


def stat_tile(label: str, value: str, note: str = "") -> str:
    return h(
        f"""
        <div class="stat">
          <div class="stat-label">{html.escape(label)}</div>
          <div class="stat-value">{html.escape(value)}</div>
          <div class="stat-note">{html.escape(note)}</div>
        </div>
        """
    )


def result_card(res: dict) -> str:
    priority = str(res.get("priority", "Medium"))
    p = palette_for(priority)
    confidence = max(0.0, min(1.0, float(res.get("confidence", 0.0))))
    intent = html.escape(pretty(str(res.get("intent", "Unknown"))))
    offset = 364.42 * (1 - confidence)
    return h(
        f"""
        <div class="result" style="--accent:{p['main']};--accent-soft:{p['soft']};">
          <div class="result-top">
            <div class="result-label">Predicted intent</div>
            <span class="chip"><span class="dot"></span>{p['label']} priority</span>
          </div>
          <div class="result-body">
            <div class="result-intent">{intent}</div>
            <svg class="ring" viewBox="0 0 140 140" role="img" aria-label="Model confidence {confidence:.0%}">
              <circle class="ring-bg" cx="70" cy="70" r="58"></circle>
              <circle class="ring-fg" cx="70" cy="70" r="58" transform="rotate(-90 70 70)" style="--off:{offset:.2f};"></circle>
              <text class="ring-num" x="70" y="75">{confidence:.0%}</text>
              <text class="ring-cap" x="70" y="93">confidence</text>
            </svg>
          </div>
          <div class="note">Confidence is a model score, not a guarantee of correctness.</div>
        </div>
        """
    )


def routing_card(res: dict) -> str:
    p = palette_for(str(res.get("priority", "Medium")))
    if res.get("needs_human_review"):
        tone, soft, icon = "#ffb454", "rgba(255,180,84,.14)", "👤"
        title = "Send to a human agent"
        copy = "Confidence is low, so don't auto-route this ticket. An agent should read it first."
    else:
        tone, soft, icon = "#3ddc97", "rgba(61,220,151,.14)", "✓"
        title = "Safe to route automatically"
        copy = f"{p['route']} {p['sla']}."
    return h(
        f"""
        <div class="route" style="--tone:{tone};--tone-soft:{soft};">
          <div class="route-ico">{icon}</div>
          <div>
            <div class="route-title">{title}</div>
            <div class="route-copy">{copy}</div>
          </div>
        </div>
        """
    )


def alt_bars(top_k: list) -> str:
    df = pd.DataFrame(top_k)
    if df.empty or not {"intent", "score"}.issubset(df.columns):
        return ""
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).sort_values("score", ascending=False).head(5)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        score = max(0.0, min(1.0, float(r["score"])))
        rows.append(
            f"""
            <div class="bar-row{' first' if i == 0 else ''}">
              <div class="bar-name">{html.escape(pretty(str(r['intent'])))}</div>
              <div class="bar-track"><div class="bar-fill" style="width:{score * 100:.1f}%;animation-delay:{i * 70}ms;"></div></div>
              <div class="bar-val">{score:.0%}</div>
            </div>
            """
        )
    return h(f'<div class="bars">{"".join(rows)}</div>')


st.session_state.setdefault("ticket_text", "")
st.session_state.setdefault("ticket_result", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("batch_df", None)

st.markdown(
    h(
        """
        <div class="hero">
          <div>
          <div class="brand-row">
            <div class="brand-mark">🎫</div>
            <div>
              <div class="brand-name">TicketIQ</div>
              <div class="brand-tag">Support triage for banking teams</div>
            </div>
          </div>
          <div class="hero-title">Read every ticket. Route it right.</div>
          <div class="hero-copy">
            Paste a customer message and get its intent, a priority level and a routing decision in seconds.
            When the model isn't sure, the ticket goes to a person instead of the wrong queue.
          </div>
          <div class="hero-pills">
            <span class="hero-pill">✦ DistilBERT on Banking77</span>
            <span class="hero-pill">77 intents</span>
            <span class="hero-pill">Priority-aware routing</span>
            <span class="hero-pill">Human-review fallback</span>
          </div>
          </div>
          <div class="flow">
            <div class="flow-step"><div class="flow-num">1</div><div><div class="flow-title">Read the message</div><div class="flow-copy">Paste one ticket or upload a CSV.</div></div></div>
            <div class="flow-step"><div class="flow-num">2</div><div><div class="flow-title">Classify intent and priority</div><div class="flow-copy">DistilBERT scores 77 banking intents.</div></div></div>
            <div class="flow-step"><div class="flow-num">3</div><div><div class="flow-title">Route or hand to a person</div><div class="flow-copy">Low confidence goes to a human agent.</div></div></div>
          </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

try:
    with st.spinner("Loading TicketIQ model..."):
        clf = load_classifier()
except FileNotFoundError:
    st.error("Model not found. Run Step 3 (DistilBERT training) in notebooks/pipeline.ipynb first.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the classifier: {exc}")
    st.stop()

hist = pd.DataFrame(st.session_state.history)
total = len(hist)
if total:
    auto_pct = f"{(~hist['Needs review']).mean():.0%}"
    avg_conf = f"{hist['Confidence'].mean():.0%}"
    urgent = int((hist["Priority"] == "High").sum())
else:
    auto_pct, avg_conf, urgent = "-", "-", 0

s1, s2, s3, s4 = st.columns(4, gap="medium")
s1.markdown(stat_tile("Tickets analyzed", str(total), "This session"), unsafe_allow_html=True)
s2.markdown(stat_tile("Auto-routable", auto_pct, "Share not sent to review"), unsafe_allow_html=True)
s3.markdown(stat_tile("Average confidence", avg_conf, "Across all predictions"), unsafe_allow_html=True)
s4.markdown(stat_tile("Urgent tickets", str(urgent), "Needs fast escalation"), unsafe_allow_html=True)

st.write("")
tab_single, tab_batch, tab_log = st.tabs(["Analyze a ticket", "Batch from CSV", "Session log"])


with tab_single:
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            h(
                """
                <div class="panel-head">
                  <div class="panel-title">Customer message</div>
                  <div class="panel-copy">Paste a ticket or start from a sample.</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        sample_cols = st.columns(2, gap="small")
        for i, (label, example) in enumerate(EXAMPLES.items()):
            with sample_cols[i % 2]:
                st.button(
                    label,
                    key=f"sample_{label}",
                    use_container_width=True,
                    on_click=use_example,
                    args=(example,),
                )

        ticket_text = st.text_area(
            "Customer message",
            key="ticket_text",
            height=210,
            placeholder="Example: My card was stolen and I see payments I did not make...",
            help="Enter at least 3 characters. Avoid entering real personal or financial information.",
            label_visibility="collapsed",
        )
        st.caption(f"{len(ticket_text.strip())} characters. Use fictional or anonymized examples only.")

        run_col, clear_col = st.columns([3, 1], gap="small")
        with run_col:
            classify_clicked = st.button(
                "Analyze ticket",
                type="primary",
                use_container_width=True,
                disabled=len(ticket_text.strip()) < 3,
            )
        with clear_col:
            st.button("Clear", use_container_width=True, on_click=use_example, args=("",))

        if classify_clicked:
            try:
                with st.spinner("Analyzing intent and priority..."):
                    res = clf.predict([ticket_text.strip()])[0]
                st.session_state.ticket_result = res
                log_result(ticket_text.strip(), res)
                st.rerun()
            except Exception as exc:
                st.session_state.ticket_result = None
                st.error(f"Prediction failed: {exc}")

    with right:
        st.markdown(
            h(
                """
                <div class="panel-head">
                  <div class="panel-title">Prediction</div>
                  <div class="panel-copy">Intent, priority, confidence and the suggested next step.</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        res = st.session_state.ticket_result
        if res:
            st.markdown(result_card(res), unsafe_allow_html=True)
            st.markdown(routing_card(res), unsafe_allow_html=True)
            bars = alt_bars(res.get("top_k", []))
            if bars:
                st.markdown('<div class="mini-title">Other intents the model considered</div>', unsafe_allow_html=True)
                st.markdown(bars, unsafe_allow_html=True)
        else:
            st.markdown(
                h(
                    """
                    <div class="empty">
                      <div class="empty-ico">✦</div>
                      <div class="empty-title">Results appear here</div>
                      <div class="empty-copy">
                        Pick a sample or paste a message, then select Analyze ticket
                        to see intent, priority, confidence and the next step.
                      </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

with tab_batch:
    st.markdown(
        h(
            f"""
            <div class="panel-head">
              <div class="panel-title">Classify many tickets at once</div>
              <div class="panel-copy">Upload a CSV with one message per row (up to {MAX_BATCH_ROWS} rows). You can download the results.</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception as exc:
            raw = None
            st.error(f"Could not read this CSV: {exc}")
        if raw is not None:
            if raw.empty:
                st.warning("This CSV has no rows.")
            else:
                text_cols = [c for c in raw.columns if raw[c].dtype == object] or list(raw.columns)
                col = st.selectbox("Column that holds the message", text_cols)
                if st.button("Classify all rows", type="primary"):
                    texts = raw[col].dropna().astype(str).str.strip()
                    texts = texts[texts.str.len() >= 3].head(MAX_BATCH_ROWS)
                    if texts.empty:
                        st.warning("No usable messages found in that column.")
                    else:
                        try:
                            with st.spinner(f"Classifying {len(texts)} tickets..."):
                                preds = clf.predict(texts.tolist())
                            st.session_state.batch_df = pd.DataFrame(
                                {
                                    "message": texts.values,
                                    "intent": [pretty(str(p.get("intent", ""))) for p in preds],
                                    "priority": [p.get("priority", "") for p in preds],
                                    "confidence": [round(float(p.get("confidence", 0.0)), 4) for p in preds],
                                    "needs_human_review": [bool(p.get("needs_human_review")) for p in preds],
                                }
                            )
                        except Exception as exc:
                            st.session_state.batch_df = None
                            st.error(f"Batch prediction failed: {exc}")

    bdf = st.session_state.batch_df
    if bdf is not None and not bdf.empty:
        n = len(bdf)
        review_n = int(bdf["needs_human_review"].sum())
        b1, b2, b3 = st.columns(3, gap="medium")
        b1.markdown(stat_tile("Rows classified", str(n)), unsafe_allow_html=True)
        b2.markdown(stat_tile("Need human review", str(review_n), f"{review_n / n:.0%} of rows"), unsafe_allow_html=True)
        b3.markdown(stat_tile("Urgent", str(int((bdf["priority"] == "High").sum()))), unsafe_allow_html=True)
        st.write("")
        st.dataframe(
            bdf,
            use_container_width=True,
            hide_index=True,
            column_config={
                "confidence": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0, format="%.0f%%"),
            },
        )
        st.download_button(
            "Download results (CSV)",
            bdf.to_csv(index=False).encode("utf-8"),
            file_name="ticketiq_results.csv",
            mime="text/csv",
            type="primary",
        )

with tab_log:
    st.markdown(
        h(
            """
            <div class="panel-head">
              <div class="panel-title">Session log</div>
              <div class="panel-copy">Every ticket you analyze in this session. It clears when you close the page.</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    if total:
        st.dataframe(
            hist,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.0f%%"),
            },
        )
        st.button("Clear log", on_click=lambda: st.session_state.update(history=[]))
    else:
        st.info("Nothing here yet. Analyze a ticket and it will show up in this log.")

st.markdown(
    '<div class="footer">TicketIQ: AI-assisted support triage. Use anonymized sample messages for testing.</div>',
    unsafe_allow_html=True,
)