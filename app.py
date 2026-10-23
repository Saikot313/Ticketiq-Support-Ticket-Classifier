import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import html
import pandas as pd
import streamlit as st

from ticketclf.predict import TicketClassifier

st.set_page_config(
    page_title="TicketIQ | AI Support Intelligence",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COLORS = {
    "High": {"main": "#ef4444", "soft": "rgba(239, 68, 68, .12)", "label": "URGENT"},
    "Medium": {"main": "#f59e0b", "soft": "rgba(245, 158, 11, .14)", "label": "STANDARD"},
    "Low": {"main": "#22c55e", "soft": "rgba(34, 197, 94, .12)", "label": "LOW"},
}
EXAMPLES = {
    "🚨 Stolen card": "My card was stolen and I see payments I did not make",
    "💳 Top up": "How do I top up my account?",
    "📦 Card delivery": "I still haven't received my card, where is it?",
    "🧾 Double charge": "Why was I charged twice for the same purchase?",
}

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp {
        background: #0d1117;
        color: #e6edf3;
    }
    [data-testid="stHeader"] {
        background: rgba(13, 17, 23, .94);
        border-bottom: 1px solid #30363d;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 100% !important;
        padding: 1.45rem 3rem 2.8rem !important;
    }
    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem 1rem 2rem !important; }
    }
    .hero {
        position: relative; overflow: hidden;
        padding: 2rem 2.2rem; border-radius: 16px;
        background: linear-gradient(120deg, #161b22 0%, #1f2937 58%, #0d3b45 100%);
        color: #e6edf3; margin-bottom: 1.35rem;
        border: 1px solid #30363d;
        box-shadow: 0 12px 34px rgba(0, 0, 0, .22);
    }
    .hero:after {
        content: ""; position: absolute; width: 300px; height: 300px;
        right: -80px; top: -140px; border-radius: 50%;
        background: rgba(88,166,255,.06); border: 1px solid rgba(139,148,158,.15);
    }
    .eyebrow {
        font-size: .76rem; letter-spacing: .14em; text-transform: uppercase;
        font-weight: 800; color: #58a6ff; margin-bottom: .65rem;
    }
    .hero-title { font-size: clamp(2rem, 3vw, 3.1rem); line-height: 1.08;
        font-weight: 850; letter-spacing: -.04em; margin: 0 0 .7rem; color:#f0f6fc; }
    .hero-copy { max-width: 780px; color: #c9d1d9; font-size: 1.02rem; line-height: 1.65; }
    .hero-pills { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.25rem; }
    .hero-pill { border:1px solid #484f58; background:rgba(110,118,129,.12);
        padding:.42rem .75rem; border-radius:999px; font-size:.8rem; color:#e6edf3; }
    .section-kicker { color:#8b949e; font-size:.76rem; font-weight:800;
        text-transform:uppercase; letter-spacing:.12em; margin: .2rem 0 .45rem; }
    .panel {
        background: #161b22; border:1px solid #30363d;
        border-radius:16px; padding:1.3rem 1.4rem;
        box-shadow:0 6px 20px rgba(0,0,0,.12); min-height:100%;
    }
    .panel-title { font-size:1.25rem; font-weight:800; letter-spacing:-.02em;
        color:#f0f6fc; margin-bottom:.25rem; }
    .panel-copy { color:#8b949e; font-size:.92rem; margin-bottom:1.05rem; line-height:1.5; }
    .metric {
        background:#161b22; border:1px solid #30363d; border-radius:12px;
        padding:1rem 1.1rem; box-shadow:0 4px 14px rgba(0,0,0,.1);
    }
    .metric-label { color:#8b949e; font-size:.76rem; font-weight:750;
        letter-spacing:.07em; text-transform:uppercase; }
    .metric-value { color:#f0f6fc; font-size:1.12rem; font-weight:850; margin-top:.25rem; }
    .result-card { background:#161b22; border:1px solid #30363d; border-radius:16px;
        padding:1.4rem; box-shadow:0 6px 20px rgba(0,0,0,.12); margin-top:.6rem; }
    .result-label { font-size:.75rem; color:#8b949e; font-weight:800;
        letter-spacing:.1em; text-transform:uppercase; margin-bottom:.45rem; }
    .result-intent { font-size:1.55rem; line-height:1.2; color:#f0f6fc;
        font-weight:850; text-transform:capitalize; overflow-wrap:anywhere; }
    .priority-chip { display:inline-block; border-radius:999px; padding:.42rem .72rem;
        font-size:.76rem; font-weight:850; letter-spacing:.06em; }
    .confidence-track { width:100%; height:9px; border-radius:999px;
        background:#30363d; overflow:hidden; margin:.65rem 0 .35rem; }
    .confidence-fill { height:100%; border-radius:999px; }
    div.stButton > button {
        border-radius:8px; border:1px solid #30363d; min-height:2.65rem;
        font-weight:700; color:#e6edf3; background:#21262d; transition:all .15s ease;
    }
    div.stButton > button:hover { border-color:#58a6ff; color:#58a6ff; background:#1c2128; }
    div.stButton > button[kind="primary"] {
        background:#238636; border:1px solid rgba(240,246,252,.1); color:#fff;
        box-shadow:none;
    }
    div.stButton > button[kind="primary"]:hover { background:#2ea043; color:#fff; }
    div.stTextArea textarea {
        border-radius:10px; border:1px solid #30363d; background:#0d1117;
        color:#e6edf3; font-size:1rem; line-height:1.6; padding:1rem;
    }
    div.stTextArea textarea:focus { border-color:#58a6ff; box-shadow:0 0 0 2px rgba(56,139,253,.2); }
    label, .stMarkdown, [data-testid="stCaptionContainer"] { color:#e6edf3; }
    [data-testid="stAlert"] { border-radius:10px; }
    .footer { text-align:center; color:#8b949e; font-size:.8rem; padding:1.5rem 0 .2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI Support Intelligence · Banking</div>
      <div class="hero-title">TicketIQ</div>
      <div class="hero-copy">
        Turn customer messages into actionable support signals. Classify ticket intent,
        understand priority, and route uncertain cases to a human agent.
      </div>
      <div class="hero-pills">
        <span class="hero-pill">✦ DistilBERT · Banking77</span>
        <span class="hero-pill">◈ Intent classification</span>
        <span class="hero-pill">↗ Priority-aware routing</span>
        <span class="hero-pill">✓ Human-review fallback</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    with st.spinner("Loading TicketIQ model..."):
        clf = TicketClassifier()
except FileNotFoundError:
    st.error("Model not found. Run Step 3 (DistilBERT training) in notebooks/pipeline.ipynb first.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the classifier: {exc}")
    st.stop()

if "ticket_text" not in st.session_state:
    st.session_state.ticket_text = ""
if "ticket_result" not in st.session_state:
    st.session_state.ticket_result = None


def pretty(intent: str) -> str:
    return intent.rstrip("?").replace("_", " ").strip()


def use_example(example: str):
    st.session_state.ticket_text = example
    st.session_state.ticket_result = None


# Full-width quick stats row
m1, m2, m3, m4 = st.columns(4, gap="medium")
with m1:
    st.markdown('<div class="metric"><div class="metric-label">Model</div><div class="metric-value">DistilBERT</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric"><div class="metric-label">Domain</div><div class="metric-value">Banking support</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric"><div class="metric-label">Output</div><div class="metric-value">Intent + priority</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric"><div class="metric-label">Safety</div><div class="metric-value">Human-review fallback</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.12, 0.88], gap="large")

with left:
    st.markdown(
        """
        <div class="panel">
          <div class="section-kicker">Step 01 · Input</div>
          <div class="panel-title">Classify a support ticket</div>
          <div class="panel-copy">Paste a customer message or start with one of the sample tickets below.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-kicker">TRY A SAMPLE</div>', unsafe_allow_html=True)
    sample_cols = st.columns(2, gap="small")
    for col, (label, example) in zip(sample_cols * 2, EXAMPLES.items()):
        with col:
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
        height=220,
        placeholder="Example: My card was stolen and I see payments I did not make...",
        help="Enter at least 3 characters. Avoid entering real personal or financial information.",
        label_visibility="visible",
    )
    st.caption(f"{len(ticket_text.strip())} characters · Use fictional or anonymized examples")

    run_col, clear_col = st.columns([3, 1], gap="small")
    with run_col:
        classify_clicked = st.button(
            "✦  Analyze ticket",
            type="primary",
            use_container_width=True,
            disabled=len(ticket_text.strip()) < 3,
        )
    with clear_col:
        st.button(
            "Clear",
            use_container_width=True,
            on_click=use_example,
            args=("",),
        )

    if classify_clicked:
        try:
            with st.spinner("Analyzing intent and routing priority..."):
                st.session_state.ticket_result = clf.predict([ticket_text.strip()])[0]
        except Exception as exc:
            st.session_state.ticket_result = None
            st.error(f"Prediction failed: {exc}")

with right:
    st.markdown(
        """
        <div class="panel">
          <div class="section-kicker">Step 02 · Intelligence</div>
          <div class="panel-title">Prediction results</div>
          <div class="panel-copy">The model's best match, confidence, and suggested handling path appear here.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    res = st.session_state.ticket_result
    if res:
        priority = res.get("priority", "Medium")
        palette = COLORS.get(priority, COLORS["Medium"])
        confidence = max(0.0, min(1.0, float(res.get("confidence", 0.0))))
        intent = html.escape(pretty(str(res.get("intent", "Unknown"))))
        priority_text = html.escape(str(priority))
        st.markdown(
            f"""
            <div class="result-card">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;">
                <div class="result-label">Predicted intent</div>
                <span class="priority-chip" style="color:{palette['main']};background:{palette['soft']};">
                  {palette['label']} PRIORITY
                </span>
              </div>
              <div class="result-intent">{intent}</div>
              <div style="margin-top:1.35rem;display:flex;justify-content:space-between;align-items:center;gap:1rem;">
                <span style="font-weight:750;color:#475569;">Priority level</span>
                <span style="font-weight:850;color:{palette['main']};">{priority_text}</span>
              </div>
              <div style="margin-top:1.1rem;display:flex;justify-content:space-between;align-items:center;">
                <span style="font-weight:750;color:#475569;">Model confidence</span>
                <span style="font-weight:850;color:#172033;">{confidence:.0%}</span>
              </div>
              <div class="confidence-track">
                <div class="confidence-fill" style="width:{confidence * 100:.1f}%;background:{palette['main']};"></div>
              </div>
              <div style="font-size:.82rem;color:#64748b;">Confidence is a model score, not a guarantee of correctness.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if res.get("needs_human_review"):
            st.warning("Human review recommended: confidence is low, so do not auto-route this ticket yet.")
        else:
            st.success("Suggested path: eligible for automated routing under your configured business rules.")

        st.markdown('<div class="section-kicker" style="margin-top:1.25rem;">ALTERNATIVE INTENTS</div>', unsafe_allow_html=True)
        top = pd.DataFrame(res.get("top_k", []))
        if not top.empty and {"intent", "score"}.issubset(top.columns):
            top["intent"] = top["intent"].map(lambda x: pretty(str(x)))
            top["score"] = pd.to_numeric(top["score"], errors="coerce")
            top = top.dropna(subset=["score"]).sort_values("score", ascending=True).tail(5)
            st.bar_chart(top.set_index("intent")["score"], use_container_width=True)
        elif not top.empty:
            st.dataframe(top, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            """
            <div class="result-card" style="min-height:300px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;">
              <div style="width:70px;height:70px;border-radius:22px;background:#eef2ff;color:#4f46e5;display:flex;align-items:center;justify-content:center;font-size:2rem;margin-bottom:1rem;">✦</div>
              <div style="font-size:1.15rem;font-weight:800;color:#172033;">Your results will appear here</div>
              <div style="max-width:320px;color:#64748b;line-height:1.6;margin-top:.5rem;">
                Enter a sample ticket, then select <b>Analyze ticket</b> to view intent, priority, confidence, and alternative predictions.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer">TicketIQ · AI-assisted support triage · Use anonymized sample messages for testing</div>',
    unsafe_allow_html=True,
)
