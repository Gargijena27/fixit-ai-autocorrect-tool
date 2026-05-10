import streamlit as st
import streamlit.components.v1 as components
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import difflib
import re
import time
import json

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="FixIt AI · Grammar Corrector",
    page_icon="⚡",
    layout="centered"
)

# ---------------- THEME ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    background: #080c14;
    color: #dce6f5;
    min-height: 100vh;
}

/* Animated mesh background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(99,102,241,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(20,184,166,0.14) 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(139,92,246,0.08) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* Scanline overlay for depth */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.04) 2px,
        rgba(0,0,0,0.04) 4px
    );
    pointer-events: none;
    z-index: 0;
}

[data-testid="stMain"], [data-testid="block-container"] {
    position: relative;
    z-index: 1;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10, 14, 26, 0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
}

/* ── Header ── */
.fixit-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}

.fixit-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6366f1;
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 100px;
    padding: 4px 14px;
    margin-bottom: 16px;
    background: rgba(99,102,241,0.07);
}

.fixit-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #a5b4fc 0%, #38bdf8 45%, #2dd4bf 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
}

.fixit-sub {
    font-size: 15px;
    color: #94a3b8;
    font-weight: 400;
    letter-spacing: 0.3px;
}

/* ── Card container ── */
.card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 20px;
    padding: 28px;
    backdrop-filter: blur(16px);
    box-shadow: 0 4px 40px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
    margin-bottom: 20px;
}

/* ── Labels ── */
.field-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.field-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.3), transparent);
}

/* ── Textarea ── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextArea"] > div > div > textarea {
    background: #0d1424 !important;
    color: #f1f5f9 !important;
    border-radius: 14px !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    padding: 16px !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    resize: vertical !important;
    caret-color: #818cf8 !important;
}

[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextArea"] > div > div > textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15), 0 0 24px rgba(99,102,241,0.1) !important;
    outline: none !important;
}

[data-testid="stTextArea"] textarea::placeholder { color: #3d5068 !important; }

/* Streamlit wraps textarea in extra divs — kill their bg too */
[data-testid="stTextArea"] > div,
[data-testid="stTextArea"] > div > div {
    background: transparent !important;
    border: none !important;
}

/* ── Buttons — universal dark override ── */
.stButton > button {
    width: 100% !important;
    padding: 13px 20px !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.4px !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
    background: #1e293b !important;
    color: #c7d2fe !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
}

.stButton > button:hover {
    background: #273549 !important;
    color: #e0e7ff !important;
    border-color: rgba(99,102,241,0.7) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.25) !important;
}

/* Correct Text button — make it pop with indigo gradient */
.stButton > button p {
    color: inherit !important;
}

/* ── Output section ── */
.output-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}

.output-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2dd4bf;
    display: flex;
    align-items: center;
    gap: 8px;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #2dd4bf;
    box-shadow: 0 0 8px #2dd4bf;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

.output-box {
    background: rgba(8, 12, 20, 0.9);
    border: 1px solid rgba(20,184,166,0.2);
    border-radius: 14px;
    padding: 20px;
    font-size: 15px;
    line-height: 1.8;
    color: #e2e8f0;
    box-shadow: 0 0 30px rgba(20,184,166,0.06), inset 0 1px 0 rgba(255,255,255,0.03);
    min-height: 80px;
    position: relative;
    overflow: hidden;
}

.output-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(45,212,191,0.4), transparent);
}

/* Diff highlighting */
.diff-ins {
    background: rgba(45, 212, 191, 0.15);
    color: #5eead4;
    border-radius: 4px;
    padding: 1px 5px;
    text-decoration: none;
    font-weight: 500;
    border-bottom: 1px solid rgba(45,212,191,0.4);
}

.diff-del {
    background: rgba(239,68,68,0.1);
    color: #fca5a5;
    text-decoration: line-through;
    border-radius: 4px;
    padding: 1px 4px;
    font-size: 13px;
    opacity: 0.7;
}

/* ── Metrics strip ── */
.metrics-strip {
    display: flex;
    gap: 12px;
    margin-top: 16px;
}

.metric-chip {
    flex: 1;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
}

.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #a5b4fc;
}

.metric-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #64748b;
    margin-top: 3px;
}

/* ── Spinner override ── */
[data-testid="stSpinner"] > div {
    border-color: #6366f1 !important;
    border-right-color: transparent !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-color: #6366f1 !important;
    background: rgba(99,102,241,0.08) !important;
}

/* ── Sidebar content ── */
.sidebar-section {
    margin-bottom: 24px;
}
.sidebar-title {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 12px;
}
.sidebar-tip {
    font-size: 13px;
    color: #64748b;
    line-height: 1.6;
    padding: 10px 14px;
    background: rgba(99,102,241,0.06);
    border-radius: 10px;
    border-left: 2px solid rgba(99,102,241,0.3);
    margin-bottom: 8px;
}

/* ── History items ── */
.history-item {
    font-size: 13px;
    color: #94a3b8;
    padding: 10px 12px;
    background: rgba(15,23,42,0.5);
    border-radius: 8px;
    border: 1px solid rgba(99,102,241,0.1);
    margin-bottom: 8px;
    transition: all 0.2s;
    overflow: hidden;
}

/* ── Mode selector ── */
div[data-testid="stSelectbox"] > div > div {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}


/* ── st.code copy block ── */
[data-testid="stCode"] { border-radius: 12px !important; overflow: hidden !important; }
[data-testid="stCode"] > div { background: #0d1a2e !important; border: 1px solid rgba(45,212,191,0.3) !important; border-radius: 12px !important; }
[data-testid="stCode"] pre { background: #0d1a2e !important; }
[data-testid="stCode"] pre code { background: #0d1a2e !important; color: #e2e8f0 !important; font-size: 15px !important; line-height: 1.75 !important; font-family: 'DM Sans', sans-serif !important; white-space: pre-wrap !important; word-break: break-word !important; }
[data-testid="stCode"] pre code span { color: #e2e8f0 !important; background: #0d1a2e !important; }
[data-testid="stCode"] button { background: rgba(45,212,191,0.15) !important; color: #2dd4bf !important; border: 1px solid rgba(45,212,191,0.35) !important; border-radius: 6px !important; opacity: 1 !important; }
[data-testid="stCode"] button:hover { background: rgba(45,212,191,0.3) !important; }
[data-testid="stCode"] button svg { fill: #2dd4bf !important; }


/* Done-copied confirmation button */
button[data-testid="baseButton-secondary"][kind="secondary"]:has(+ *),
div[data-testid="stButton"]:has(button[key="copied_confirm"]) button {
    background: rgba(45,212,191,0.08) !important;
    color: #2dd4bf !important;
    border: 1px dashed rgba(45,212,191,0.35) !important;
    font-size: 13px !important;
}
/* ── Hide streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ---------------- LOAD MODEL ----------------
MODEL_NAME = "vennify/t5-base-grammar-correction"

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

# ── Accuracy Improvements ──
# 1. Pre-processing: normalize common typos and whitespace
def preprocess(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    # Capitalize first letter of sentences
    text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    return text

# 2. Post-processing: fix punctuation spacing
def postprocess(text: str) -> str:
    text = re.sub(r'\s([?.!,;:])', r'\1', text)
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    return text.strip()

# 3. Core correction with beam search + length penalty tuning
def correct_text(text: str, tokenizer, model, device, mode: str = "Standard") -> str:
    if not text.strip():
        return text

    # Mode-based parameters
    params = {
        "Standard":  {"num_beams": 5,  "length_penalty": 1.0, "repetition_penalty": 1.2},
        "Formal":    {"num_beams": 8,  "length_penalty": 1.2, "repetition_penalty": 1.3},
        "Casual":    {"num_beams": 4,  "length_penalty": 0.9, "repetition_penalty": 1.1},
        "Technical": {"num_beams": 8,  "length_penalty": 1.1, "repetition_penalty": 1.4},
    }
    p = params.get(mode, params["Standard"])

    inputs = tokenizer(
        "grammar: " + text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=p["num_beams"],
            length_penalty=p["length_penalty"],
            repetition_penalty=p["repetition_penalty"],
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return postprocess(result)

# 4. Smart paragraph splitting — respect sentence boundaries
def split_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]

def correct_paragraph(text: str, tokenizer, model, device, mode: str) -> str:
    text = preprocess(text)
    sentences = split_sentences(text)
    corrected = [correct_text(s, tokenizer, model, device, mode) for s in sentences]
    return " ".join(corrected)

# 5. Diff highlighting
def build_diff_html(original: str, corrected: str) -> str:
    orig_words = original.split()
    corr_words = corrected.split()
    sm = difflib.SequenceMatcher(None, orig_words, corr_words)
    html_parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            html_parts.append(" ".join(corr_words[j1:j2]))
        elif tag == "replace":
            html_parts.append(f'<span class="diff-del">{" ".join(orig_words[i1:i2])}</span> <span class="diff-ins">{" ".join(corr_words[j1:j2])}</span>')
        elif tag == "insert":
            html_parts.append(f'<span class="diff-ins">{" ".join(corr_words[j1:j2])}</span>')
        elif tag == "delete":
            html_parts.append(f'<span class="diff-del">{" ".join(orig_words[i1:i2])}</span>')
    return " ".join(html_parts)

def count_changes(original: str, corrected: str) -> int:
    orig_words = original.split()
    corr_words = corrected.split()
    sm = difflib.SequenceMatcher(None, orig_words, corr_words)
    return sum(1 for tag, *_ in sm.get_opcodes() if tag != "equal")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-title">⚙ Correction Mode</p>', unsafe_allow_html=True)
    mode = st.selectbox(
        "",
        ["Standard", "Formal", "Casual", "Technical"],
        help="Choose a tone/context for smarter correction"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-title">💡 Tips</p>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tip">Use <b>Formal</b> mode for essays, emails, and professional writing.</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tip">Use <b>Technical</b> mode for code comments, docs, or reports.</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tip">Longer texts are split by sentence for better accuracy.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-title">&#128336; Recent Corrections</p>', unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-5:]):
            orig_preview = item["original"][:40] + "..." if len(item["original"]) > 40 else item["original"]
            changes = item.get("changes", "?")
            mode = item.get("mode", "Std")[:3]
            st.markdown(f"""
            <div class="history-item">
                <div style="color:#94a3b8;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{orig_preview}</div>
                <div style="display:flex;gap:8px;margin-top:4px;">
                    <span style="font-family:'DM Mono',monospace;font-size:10px;color:#6366f1;">{changes} changes</span>
                    <span style="font-family:'DM Mono',monospace;font-size:10px;color:#475569;">{mode}</span>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MAIN UI ----------------
st.markdown("""
<div class="fixit-header">
    <div class="fixit-badge">⚡ Powered by T5 Transformer</div>
    <div class="fixit-title">FixIt AI</div>
    <div class="fixit-sub">Grammar & spelling correction that actually understands context</div>
</div>
""", unsafe_allow_html=True)

# Input card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="field-label">✦ Your Text</div>', unsafe_allow_html=True)

user_input = st.text_area(
    "",
    height=160,
    placeholder='Paste or type your text here... e.g. "i dont likes her perfume. she go to market yesterday"',
    key="input_text",
    label_visibility="collapsed"
)

word_count = len(user_input.split()) if user_input.strip() else 0
char_count = len(user_input)
st.markdown(f'<div style="text-align:right;font-family:\'DM Mono\',monospace;font-size:11px;color:#64748b;margin-top:6px;">{word_count} words · {char_count} chars</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Buttons row
col1, col2 = st.columns([3, 1.5])

with col1:
    correct_btn = st.button("⚡ Correct Text", use_container_width=True)
with col2:
    clear_btn = st.button("✕ Clear", use_container_width=True)

# Actions
if clear_btn:
    st.session_state.last_result = None
    st.rerun()


if correct_btn:
    if not user_input.strip():
        st.warning("Drop some text in the box above first.")
    else:
        with st.spinner("Running correction pipeline…"):
            tokenizer, model, device = load_model()
            t0 = time.time()
            corrected = correct_paragraph(user_input, tokenizer, model, device, mode)
            elapsed = round(time.time() - t0, 2)

        # Save to history
        changes = count_changes(user_input, corrected)
        st.session_state.last_result = {
            "original": user_input,
            "corrected": corrected,
            "changes": changes,
            "elapsed": elapsed,
            "mode": mode,
        }
        st.session_state.history.append(st.session_state.last_result)
        st.rerun()

# ── Output ──
if st.session_state.last_result:
    r = st.session_state.last_result

    st.markdown("""
    <div class="output-header">
        <div class="output-label">
            <span class="status-dot"></span>
            Corrected Output
        </div>
    </div>
    """, unsafe_allow_html=True)

    diff_html = build_diff_html(r["original"], r["corrected"])
    st.markdown(f'<div class="output-box">{diff_html}</div>', unsafe_allow_html=True)

    # Metrics strip
    accuracy = max(0, min(100, 100 - (r["changes"] / max(len(r["original"].split()), 1)) * 100))
    st.markdown(f"""
    <div class="metrics-strip">
        <div class="metric-chip">
            <div class="metric-val">{r["changes"]}</div>
            <div class="metric-lbl">Changes Made</div>
        </div>
        <div class="metric-chip">
            <div class="metric-val">{r["elapsed"]}s</div>
            <div class="metric-lbl">Time Taken</div>
        </div>
        <div class="metric-chip">
            <div class="metric-val">{round(accuracy)}%</div>
            <div class="metric-lbl">Words Kept</div>
        </div>
        <div class="metric-chip">
            <div class="metric-val">{r["mode"][:3]}</div>
            <div class="metric-lbl">Mode</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Native Streamlit copy — st.code() has a built-in copy button that always works
    st.markdown('<div class="field-label" style="margin-top:18px;">⎘ Copy Fixed Text</div>', unsafe_allow_html=True)
    corrected_json = json.dumps(r["corrected"])
    copy_component = f"""
    <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:transparent; font-family: DM Sans, sans-serif; }}
    #wrap {{ position:relative; }}
    #textbox {{
        background:#0d1a2e;
        border:1px solid rgba(45,212,191,0.3);
        border-radius:12px;
        color:#e2e8f0;
        font-size:15px;
        line-height:1.75;
        padding:18px 52px 18px 18px;
        width:100%;
        word-break:break-word;
        white-space:pre-wrap;
    }}
    #copybtn {{
        position:absolute;
        top:10px; right:10px;
        width:34px; height:34px;
        background:rgba(45,212,191,0.12);
        border:1px solid rgba(45,212,191,0.35);
        border-radius:7px;
        cursor:pointer;
        display:flex; align-items:center; justify-content:center;
        transition:all 0.2s;
    }}
    #copybtn:hover {{ background:rgba(45,212,191,0.25); }}
    #copybtn svg {{ width:16px; height:16px; fill:none; stroke:#2dd4bf; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }}
    #toast {{
        display:none;
        position:fixed;
        bottom:24px; left:50%; transform:translateX(-50%);
        background:linear-gradient(135deg,#059669,#047857);
        color:#fff;
        padding:12px 28px;
        border-radius:50px;
        font-size:14px;
        font-weight:600;
        letter-spacing:0.3px;
        box-shadow:0 8px 32px rgba(5,150,105,0.45);
        z-index:9999;
        animation: fadeup 0.3s ease;
    }}
    @keyframes fadeup {{
        from {{ opacity:0; transform:translateX(-50%) translateY(10px); }}
        to   {{ opacity:1; transform:translateX(-50%) translateY(0); }}
    }}
    </style>
    <div id="wrap">
        <div id="textbox">{r["corrected"]}</div>
        <button id="copybtn" title="Copy fixed text">
            <svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        </button>
    </div>
    <div id="toast">✓ Fixed text copied!</div>
    <script>
    document.getElementById('copybtn').addEventListener('click', function() {{
        var text = {corrected_json};
        var btn = this;
        // Try modern API first, fallback for HTTP
        if (navigator.clipboard && window.isSecureContext) {{
            navigator.clipboard.writeText(text).then(showToast);
        }} else {{
            var ta = document.createElement('textarea');
            ta.value = text;
            ta.style.cssText = 'position:fixed;left:-9999px;top:-9999px';
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
            showToast();
        }}
    }});
    function showToast() {{
        var t = document.getElementById('toast');
        t.style.display = 'block';
        setTimeout(function() {{ t.style.display = 'none'; }}, 2500);
    }}
    </script>
    """
    components.html(copy_component, height=max(180, min(500, 120 + len(r["corrected"]) // 3)))