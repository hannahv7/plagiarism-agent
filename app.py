import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="TextShield — AI Semantic Engine", page_icon="🛡️", layout="wide")

# ── LIGHT THEME - EXACT LIKE SCREENSHOT #F9F4EF ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,400&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #FAF6F1 !important;
    color: #111827 !important;
    color-scheme: light only !important;
}
[data-testid="stHeader"] { background: rgba(250,246,241,0.8) !important; backdrop-filter: blur(20px); border-bottom: 1px solid #E8E2DA; }
[data-testid="stToolbar"], footer, #MainMenu { display: none !important; }
section.main > div { padding-top: 0 !important; }

/* Light glass */
.mockup-card {
    background: #FFFFFF;
    border: 1px solid #E8E2DA;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.04);
    padding: 24px;
}
.doc-card {
    background: #F9F4EF;
    border: 1px solid #E8E2DA;
    border-radius: 14px;
    padding: 14px 16px;
}
.stats-bar {
    background: #111827;
    border-radius: 16px;
    padding: 20px 24px;
}
.pill-light {
    background: #F0EBE3;
    border: 1px solid #E8E2DA;
    border-radius: 999px;
    padding: 6px 14px;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #57534E;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.hero-really {
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-weight: 400;
    color: #8B7E74;
    position: relative;
}
.hero-really::after {
    content: '';
    position: absolute;
    bottom: -4px;
    left: 0;
    width: 100%;
    height: 1px;
    background: #D6D0C8;
}

/* Tabs light */
.stTabs [data-baseweb="tab-list"] { background: #F0EBE3; border-radius: 999px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 999px; font-size: 13px; color: #78716C; border: none; }
.stTabs [aria-selected="true"] { background: white !important; color: #111827 !important; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }

/* Textarea light */
textarea {
    background: #FFFFFF !important;
    border: 1px solid #E8E2DA !important;
    border-radius: 12px !important;
    color: #111827 !important;
}
textarea:focus { border-color: #111827 !important; box-shadow: 0 0 0 2px rgba(17,24,39,0.08) !important; }

/* Buttons */
.stButton>button {
    border-radius: 14px !important;
    height: 52px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
.stButton>button[kind="primary"] {
    background: #111827 !important;
    color: white !important;
    border: none !important;
}
.stButton>button[kind="secondary"] {
    background: white !important;
    color: #111827 !important;
    border: 1px solid #E8E2DA !important;
}
</style>
<meta name="color-scheme" content="light only">
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
<div style="max-width:1280px; margin:0 auto; padding:16px 32px; display:flex; justify-content:space-between; align-items:center;">
  <div style="display:flex; align-items:center; gap:14px;">
    <div style="width:36px; height:36px; border-radius:10px; background:#111827; display:flex; align-items:center; justify-content:center; color:white; font-size:18px;">🛡️</div>
    <div style="display:flex; align-items:center; gap:10px;">
      <span style="font-family:'Inter',sans-serif; font-weight:700; font-size:20px; letter-spacing:-0.5px; color:#111827;">Text<span style="font-family:'Fraunces',serif; font-style:italic; font-weight:400;">Shield</span></span>
      <span style="font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:0.08em; background:#F0EBE3; border:1px solid #E8E2DA; padding:3px 10px; border-radius:999px; color:#78716C;">AI SEMANTIC ENGINE</span>
    </div>
  </div>
  <div style="display:flex; gap:32px; align-items:center; font-family:'Inter',sans-serif; font-size:14px; color:#78716C;">
    <span style="color:#111827; border-bottom:2px solid #111827; padding-bottom:4px; font-weight:600;">Product</span>
    <span>How It Works</span><span>Features</span><span>Dashboard Overview</span>
  </div>
  <div style="background:#111827; color:white; padding:10px 20px; border-radius:999px; font-family:'Inter',sans-serif; font-size:14px; font-weight:600; display:flex; align-items:center; gap:8px;">Get Started ↗</div>
</div>
<div style="height:1px; background:#E8E2DA; width:100%;"></div>
""", unsafe_allow_html=True)

# ── HERO ──
st.markdown("""
<div style="max-width:1280px; margin:0 auto; padding:72px 32px 48px; display:grid; grid-template-columns: 1.1fr 0.9fr; gap:48px; align-items:start;">
  <div>
    <div class="pill-light">✦ AI DOCUMENT ANALYSIS</div>
    <h1 style="font-family:'Inter',sans-serif; font-size:64px; font-weight:700; line-height:0.95; letter-spacing:-2.8px; color:#111827; margin:28px 0 0;">Understand how similar<br>your documents <span style="font-family:'Fraunces',serif; font-style:italic; font-weight:400; color:#8B7E74; border-bottom:1px solid #D6D0C8;">really are.</span></h1>
    <p style="font-family:'Inter',sans-serif; font-size:18px; line-height:1.6; color:#78716C; max-width:520px; margin:24px 0 0;">Compare documents using semantic similarity, detect meaningful matching sections, and identify potential plagiarism risk beyond simple keyword matching.</p>
    <div style="display:flex; gap:12px; margin-top:36px;">
      <div style="background:#111827; color:white; padding:16px 28px; border-radius:14px; font-weight:600; font-size:15px; display:flex; align-items:center; gap:8px; cursor:pointer;">Analyze Documents →</div>
      <div style="background:white; border:1px solid #E8E2DA; padding:16px 28px; border-radius:14px; font-weight:600; font-size:15px; color:#111827;">See How It Works</div>
    </div>
    <div style="display:grid; grid-template-columns: repeat(3,1fr); gap:24px; margin-top:48px; padding-top:24px; border-top:1px solid #E8E2DA;">
      <div style="display:flex; align-items:center; gap:8px; font-size:13px; color:#78716C;"><span style="width:18px; height:18px; border-radius:50%; border:1.5px solid #111827; display:flex; align-items:center; justify-content:center; font-size:10px;">✓</span> Vector Embeddings</div>
      <div style="display:flex; align-items:center; gap:8px; font-size:13px; color:#78716C;"><span style="width:18px; height:18px; border-radius:50%; border:1.5px solid #111827; display:flex; align-items:center; justify-content:center; font-size:10px;">✓</span> Section Matching</div>
      <div style="display:flex; align-items:center; gap:8px; font-size:13px; color:#78716C;"><span style="width:18px; height:18px; border-radius:50%; border:1.5px solid #111827; display:flex; align-items:center; justify-content:center; font-size:10px;">✓</span> PDF / DOCX / TXT</div>
    </div>
  </div>

  <div class="mockup-card">
    <div style="display:flex; justify-content:space-between; align-items:center; padding-bottom:16px; border-bottom:1px solid #F0EBE3;">
      <div style="display:flex; align-items:center; gap:12px;">
        <div style="display:flex; gap:6px;"><div style="width:12px; height:12px; border-radius:50%; background:#FF8FA3;"></div><div style="width:12px; height:12px; border-radius:50%; background:#FFD93D;"></div><div style="width:12px; height:12px; border-radius:50%; background:#6BCFB8;"></div></div>
        <span style="font-family:'JetBrains Mono',monospace; font-size:12px; color:#78716C;">analysis_report_v2.json</span>
      </div>
      <div style="background:#111827; color:white; font-size:10px; font-weight:700; letter-spacing:0.06em; padding:4px 10px; border-radius:999px;">LIVE MOCKUP</div>
    </div>

    <div style="margin-top:20px; display:flex; flex-direction:column; gap:12px;">
      <div class="doc-card" style="display:flex; justify-content:space-between; align-items:center;">
        <div style="display:flex; gap:10px; align-items:center;">
          <div style="width:28px; height:28px; background:white; border:1px solid #E8E2DA; border-radius:8px; display:flex; align-items:center; justify-content:center;">📄</div>
          <div><div style="font-size:13px; font-weight:600; color:#111827;">Document_A_Research.docx</div><div style="font-size:11px; color:#78716C;">2,450 words • Reference Document</div></div>
        </div>
        <div style="font-size:11px; font-weight:600; color:#57534E;">Doc A</div>
      </div>
      <div class="doc-card" style="display:flex; justify-content:space-between; align-items:center;">
        <div style="display:flex; gap:10px; align-items:center;">
          <div style="width:28px; height:28px; background:white; border:1px solid #E8E2DA; border-radius:8px; display:flex; align-items:center; justify-content:center;">📄</div>
          <div><div style="font-size:13px; font-weight:600; color:#111827;">Student_Submission_B.pdf</div><div style="font-size:11px; color:#78716C;">2,180 words • Target Comparison</div></div>
        </div>
        <div style="font-size:11px; font-weight:600; color:#57534E;">Doc B</div>
      </div>
    </div>

    <div class="stats-bar" style="margin-top:20px; display:grid; grid-template-columns: repeat(3,1fr); text-align:center;">
      <div><div style="font-size:24px; font-weight:700; color:white;">82%</div><div style="font-size:10px; letter-spacing:0.08em; text-transform:uppercase; color:#9CA3AF; margin-top:4px;">SIMILARITY</div></div>
      <div style="border-left:1px solid rgba(255,255,255,0.12); border-right:1px solid rgba(255,255,255,0.12);"><div style="display:flex; justify-content:center;"><div style="background:#4A1F2A; color:#FF8FA3; font-size:11px; font-weight:700; padding:3px 10px; border-radius:999px; display:flex; align-items:center; gap:4px;">🛡️ High</div></div><div style="font-size:10px; letter-spacing:0.08em; text-transform:uppercase; color:#9CA3AF; margin-top:6px;">RISK LEVEL</div></div>
      <div><div style="font-size:24px; font-weight:700; color:white;">12</div><div style="font-size:10px; letter-spacing:0.08em; text-transform:uppercase; color:#9CA3AF; margin-top:4px;">MATCHES</div></div>
    </div>

    <div style="margin-top:16px; border:1px solid #E8E2DA; border-radius:12px; padding:14px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div style="font-size:13px; font-weight:600; color:#111827;">Section #3 Semantic Match</div>
        <div style="background:#FFF0F3; border:1px solid #FFD0D8; color:#C2185B; font-size:11px; font-weight:700; padding:3px 10px; border-radius:999px;">86.2% Overlap</div>
      </div>
      <div style="margin-top:12px; background:#FAF6F1; border:1px solid #F0EBE3; border-radius:8px; padding:12px; font-family:'JetBrains Mono',monospace; font-size:11px; line-height:1.6; color:#57534E;">"Deep learning algorithms enable automated diagnostic scans to detect early stage anomalous structures..."</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── ANALYZE SECTION (Light) ──
st.markdown('<div id="analyze" style="max-width:1120px; margin:0 auto; padding:48px 32px;"><div style="height:1px; background:#E8E2DA; margin-bottom:48px;"></div>', unsafe_allow_html=True)
st.markdown('<div style="font-family:Fraunces,serif; font-size:28px; font-weight:700; color:#111827; margin-bottom:8px;">Analyze Documents</div><div style="color:#78716C; margin-bottom:24px;">Paste or upload — same engine powering the mockup above</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown('<div style="font-size:11px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#111827; margin-bottom:10px;">Original Document (Source)</div>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["📝 Paste text", "📁 Upload file"])
    src_paste = ""; src_file = None
    with t1: src_paste = st.text_area("src", placeholder="Paste source document here…", height=160, label_visibility="collapsed", key="src_l")
    with t2: src_file = st.file_uploader("Source file", type=['pdf','txt'], label_visibility="collapsed", key="src_f_l")
with c2:
    st.markdown('<div style="font-size:11px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#111827; margin-bottom:10px;">Suspected Document</div>', unsafe_allow_html=True)
    t3, t4 = st.tabs(["📝 Paste text", "📁 Upload file"])
    susp_paste = ""; susp_file = None
    with t3: susp_paste = st.text_area("susp", placeholder="Paste suspected document here…", height=160, label_visibility="collapsed", key="susp_l")
    with t4: susp_file = st.file_uploader("Suspected file", type=['pdf','txt'], label_visibility="collapsed", key="susp_f_l")

def get_text(paste, file):
    if paste and paste.strip(): return paste
    if file: return extract_text(file)
    return None

src_doc = get_text(src_paste, src_file)
susp_doc = get_text(susp_paste, susp_file)

if st.button("Analyze Documents →", type="primary", use_container_width=True):
    if not src_doc or not susp_doc:
        st.warning("Please provide both documents — paste or upload.")
    else:
        with st.spinner("Analyzing with semantic embeddings…"):
            src_sents = split_sentences(src_doc)
            susp_sents = split_sentences(susp_doc)
            if len(src_sents)<1 or len(susp_sents)<1:
                st.error("Documents too short.")
            else:
                sim = cosine_similarity(model.encode(susp_sents), model.encode(src_sents))
                matches=[]
                for i,row in enumerate(sim):
                    j=row.argmax(); score=float(row[j])
                    if score>0.75: matches.append({"i":i,"j":j,"score":score,"type":"Exact" if score>0.92 else "Paraphrased","s":susp_sents[i],"o":src_sents[j]})
                pct = round(len(matches)/len(susp_sents)*100,1) if susp_sents else 0
                exact=len([m for m in matches if m['type']=='Exact']); para=len(matches)-exact
                
                st.markdown(f"""
                <div style="margin-top:24px; display:grid; grid-template-columns:repeat(3,1fr); gap:16px;">
                  <div style="background:white; border:1px solid #E8E2DA; border-radius:14px; padding:20px;"><div style="font-size:32px; font-weight:700; color:#111827;">{pct}%</div><div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:#78716C; margin-top:6px;">Overall Similarity</div></div>
                  <div style="background:white; border:1px solid #E8E2DA; border-radius:14px; padding:20px;"><div style="font-size:32px; font-weight:700; color:#111827;">{len(matches)}</div><div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:#78716C; margin-top:6px;">{exact} Exact • {para} Paraphrased</div></div>
                  <div style="background:#111827; border-radius:14px; padding:20px;"><div style="font-size:32px; font-weight:700; color:white;">{'High' if pct>30 else 'Low'}</div><div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:#9CA3AF; margin-top:6px;">Risk Level</div></div>
                </div>
                """, unsafe_allow_html=True)
                for m in matches[:10]:
                    st.markdown(f"""
                    <div style="margin-top:14px; background:white; border:1px solid #E8E2DA; border-left:3px solid #111827; border-radius:14px; padding:18px 20px;">
                      <div style="display:flex; justify-content:space-between;"><span style="font-size:13px; font-weight:600; color:#111827;">{m['type']} • {m['score']:.3f}</span><span style="font-size:11px; border:1px solid #E8E2DA; border-radius:999px; padding:2px 10px; color:#78716C;">S{m['i']+1} → O{m['j']+1}</span></div>
                      <div style="margin-top:10px; background:#FAF6F1; border:1px solid #F0EBE3; border-radius:8px; padding:10px 12px; font-size:12px; color:#57534E;">"{m['s']}"</div>
                      <div style="margin-top:8px; font-size:12px; color:#78716C;">Matched to: "{m['o']}"</div>
                    </div>
                    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; padding:40px; color:#A8A29E; font-size:12px; border-top:1px solid #E8E2DA; margin-top:48px;">
  TextShield • Light Theme #FAF6F1 • Built for Forge • Vector Embeddings + Section Matching + PDF/DOCX/TXT
</div>
""", unsafe_allow_html=True)
