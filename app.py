import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="TextShield — Protect Originality", page_icon="🛡️", layout="wide")

# ── TEXTSHIELD CLONE CSS — DARK PREMIUM LIKE https://textshield-4zh.pages.dev/ ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #07080B !important;
    color: #E5E7EB !important;
}
[data-testid="stHeader"] { background: rgba(7,8,11,0.8) !important; backdrop-filter: blur(20px); }
[data-testid="stToolbar"] { display: none; }
section.main > div { padding-top: 0 !important; }

/* Hide default streamlit */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

/* Glass card */
.glass {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    backdrop-filter: blur(20px);
    padding: 24px;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 56px;
    font-weight: 700;
    line-height: 0.95;
    letter-spacing: -2.5px;
    background: linear-gradient(100deg, #FFFFFF 20%, #9CA3AF 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.gradient-text {
    background: linear-gradient(90deg, #60A5FA, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtle { color: #9CA3AF; font-family: Inter, sans-serif; }
.pill {
    display: inline-flex; padding: 4px 12px; border-radius: 999px;
    background: rgba(96,165,250,0.12); border: 1px solid rgba(96,165,250,0.25);
    color: #93C5FD; font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
}

/* Upload area */
.upload-box {
    border: 1.5px dashed rgba(255,255,255,0.15);
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    padding: 28px;
    transition: all 0.2s;
}
.upload-box:hover { border-color: rgba(96,165,250,0.5); background: rgba(96,165,250,0.06); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.04); border-radius: 999px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 999px; font-size: 13px; color: #9CA3AF; }
.stTabs [aria-selected="true"] { background: white !important; color: black !important; }

/* Textarea */
textarea {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* Primary button */
.stButton>button[kind="primary"] {
    background: linear-gradient(90deg, #3B82F6, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    border-radius: 999px !important;
    height: 48px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.02em;
    box-shadow: 0 0 20px rgba(59,130,246,0.4);
}
.stButton>button[kind="primary"]:hover { box-shadow: 0 0 30px rgba(59,130,246,0.6); transform: translateY(-1px); }

/* Score */
.score-card {
    background: radial-gradient(120% 120% at 0% 0%, rgba(59,130,246,0.15), transparent 60%), rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 28px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()

def extract_text(file):
    if file.name.endswith('.pdf'):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "".join([p.get_text() for p in doc])
    return file.read().decode('utf-8', errors='ignore')

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s.strip())>20]

# ── HEADER ──
st.markdown("""
<div style="max-width:1200px; margin:0 auto; padding:18px 24px; display:flex; justify-content:space-between; align-items:center; position:sticky; top:0; z-index:10; background:rgba(7,8,11,0.7); backdrop-filter:blur(20px); border-bottom:1px solid rgba(255,255,255,0.06);">
  <div style="display:flex; align-items:center; gap:12px;">
    <div style="width:32px; height:32px; border-radius:9px; background:linear-gradient(135deg,#3B82F6,#8B5CF6); display:flex; align-items:center; justify-content:center; box-shadow:0 0 15px rgba(59,130,246,0.5);">🛡️</div>
    <span style="font-family:'Fraunces',serif; font-weight:700; font-size:18px; letter-spacing:-0.5px; color:white;">TextShield</span>
    <span style="font-size:11px; padding:3px 8px; border-radius:999px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12); color:#9CA3AF; margin-left:8px;">BETA</span>
  </div>
  <div style="display:flex; gap:24px; align-items:center; font-family:Inter,sans-serif; font-size:13px; color:#9CA3AF;">
    <span>How it works</span><span>Pricing</span>
    <span style="background:white; color:black; padding:8px 16px; border-radius:999px; font-weight:600;">Launch App</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── HERO ──
st.markdown("""
<div style="max-width:1200px; margin:0 auto; padding:64px 24px 32px; text-align:center;">
  <div class="pill" style="margin-bottom:24px;">● Beyond percentage — explainable detection</div>
  <div class="hero-title">Protect originality.<br><span class="gradient-text">Detect everything.</span></div>
  <p class="subtle" style="max-width:600px; margin:24px auto 0; font-size:17px; line-height:1.6;">We don't just give you a %. TextShield shows <span style="color:white;">which concepts</span> were copied, <span style="color:white;">where</span> (S3 → O1), and <span style="color:white;">why</span> it was flagged — exact vs paraphrased with semantic similarity.</p>
  <div style="margin-top:32px; display:flex; gap:12px; justify-content:center;">
    <span style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); padding:10px 18px; border-radius:999px; font-size:13px;">✦ 48.3% similarity engine</span>
    <span style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); padding:10px 18px; border-radius:999px; font-size:13px;">✦ Explainable matches</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD ──
st.markdown('<div style="max-width:1120px; margin:0 auto; padding:0 24px;">', unsafe_allow_html=True)
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown('<div style="font-family:Inter,sans-serif; font-size:12px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#9CA3AF; margin-bottom:12px;">Original Document (Source)</div>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["Paste text", "Upload file"])
    src_paste = ""
    src_file = None
    with t1:
        src_paste = st.text_area("src", placeholder="Paste your source document here… Supports research papers, articles, any text.", height=160, label_visibility="collapsed", key="src_p")
    with t2:
        src_file = st.file_uploader("src file", type=['pdf','txt'], label_visibility="collapsed", key="sf")

with c2:
    st.markdown('<div style="font-family:Inter,sans-serif; font-size:12px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#9CA3AF; margin-bottom:12px;">Suspected Document</div>', unsafe_allow_html=True)
    t3, t4 = st.tabs(["Paste text", "Upload file"])
    susp_paste = ""
    susp_file = None
    with t3:
        susp_paste = st.text_area("susp", placeholder="Paste suspected content to check against source…", height=160, label_visibility="collapsed", key="susp_p")
    with t4:
        susp_file = st.file_uploader("susp file", type=['pdf','txt'], label_visibility="collapsed", key="sf2")

def get_text(paste, file):
    if paste and paste.strip(): return paste
    if file: return extract_text(file)
    return None

src_doc = get_text(src_paste, src_file)
susp_doc = get_text(susp_paste, susp_file)

analyze = st.button("✦ Analyze Documents — Explainable Detection", type="primary", use_container_width=True)

if analyze:
    if not src_doc or not susp_doc:
        st.warning("Add both documents — paste or upload.")
    else:
        with st.spinner("Scanning with MiniLM-L6-v2 • Computing cosine similarity • Mapping concepts…"):
            src_sents = split_sentences(src_doc)
            susp_sents = split_sentences(susp_doc)
            src_emb = model.encode(src_sents)
            susp_emb = model.encode(susp_sents)
            sim = cosine_similarity(susp_emb, src_emb)
            matches = []
            for i,row in enumerate(sim):
                j = row.argmax(); score = float(row[j])
                if score>0.75:
                    matches.append({"i":i,"j":j,"score":score,"type":"Exact" if score>0.92 else "Paraphrased","s":susp_sents[i],"o":src_sents[j]})
            total = len(susp_sents)
            pct = round(len(matches)/total*100,1) if total else 0
            exact = len([m for m in matches if m['type']=='Exact'])
            para = len(matches)-exact

            # SCORES
            st.markdown(f"""
            <div style="margin-top:32px; display:grid; grid-template-columns:1.2fr 0.8fr 0.8fr; gap:16px;">
              <div class="score-card">
                <div style="font-size:11px; letter-spacing:0.1em; text-transform:uppercase; color:#9CA3AF; font-weight:600;">Overall Plagiarism</div>
                <div style="display:flex; align-items:baseline; gap:12px; margin-top:12px;"><div style="font-family:'Fraunces',serif; font-size:48px; font-weight:700; color:white;">{pct}%</div><div style="color:#9CA3AF; font-size:13px;">{len(matches)}/{total} sentences flagged</div></div>
                <div style="height:4px; background:rgba(255,255,255,0.08); border-radius:999px; margin-top:20px;"><div style="height:4px; width:{pct}%; background:linear-gradient(90deg,#3B82F6,#8B5CF6); border-radius:999px;"></div></div>
              </div>
              <div class="glass"><div style="font-family:'Fraunces',serif; font-size:36px; font-weight:700; color:white;">{len(matches)}</div><div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:#9CA3AF; margin-top:6px;">Explainable Matches</div><div style="margin-top:8px; font-size:12px; color:#6B7280;">{exact} Exact • {para} Paraphrased</div></div>
              <div class="glass"><div style="font-family:'Fraunces',serif; font-size:36px; font-weight:700; color:white;">{'Suspicious' if pct>30 else 'Clean'}</div><div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:#9CA3AF; margin-top:6px;">Document Health</div><div style="margin-top:8px; font-size:12px; color:#6B7280;">{'Needs review' if pct>30 else 'Original'}</div></div>
            </div>
            """, unsafe_allow_html=True)

            # MATCHES
            for m in matches[:10]:
                col = "#60A5FA" if m['type']=="Exact" else "#A78BFA"
                st.markdown(f"""
                <div style="margin-top:16px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-left:3px solid {col}; border-radius:16px; padding:20px 24px;">
                  <div style="display:flex; justify-content:space-between; align-items:center;"><span style="font-weight:600; color:white; font-size:13px;">{m['type']} • {m['score']:.2f} similarity</span><span style="font-size:11px; border:1px solid rgba(255,255,255,0.15); border-radius:999px; padding:3px 10px; color:#9CA3AF;">S{m['i']+1} → O{m['j']+1}</span></div>
                  <div style="margin-top:12px; background:rgba(0,0,0,0.3); border-left:2px solid rgba(255,255,255,0.1); padding:10px 14px; border-radius:0 8px 8px 0; font-size:13px; color:#D1D5DB;">Flagged because semantic similarity {m['score']:.2f} with { 'verbatim reproduction' if m['type']=='Exact' else 'paraphrased overlap' }.</div>
                  <div style="margin-top:12px; font-size:12px; color:#9CA3AF;"><b style="color:#6B7280;">Suspected:</b> <span style="font-style:italic; color:#E5E7EB;">"{m['s']}"</span></div>
                  <div style="margin-top:6px; font-size:12px; color:#9CA3AF;"><b style="color:#6B7280;">Original:</b> <span style="font-style:italic; color:#E5E7EB;">"{m['o']}"</span></div>
                </div>
                """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
