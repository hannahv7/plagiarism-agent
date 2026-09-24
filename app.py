import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="TextShield", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,400&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [data-testid="stAppViewContainer"] { background:#FAF6F1!important; color:#111827!important; }
[data-testid="stHeader"] { background:rgba(250,246,241,0.9)!important; backdrop-filter: blur(20px); border-bottom:1px solid #E8E2DA; }
[data-testid="stToolbar"], footer, #MainMenu { display:none!important; }
section.main > div { padding-top:0!important; }
.stTabs [data-baseweb="tab-list"] { background:#F0EBE3; border-radius:999px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] { border-radius:999px; font-size:13px; color:#78716C; }
.stTabs [aria-selected="true"] { background:white!important; color:#111827!important; }
textarea { background:white!important; border:1px solid #E8E2DA!important; border-radius:12px!important; color:#111827!important; }
.stButton>button { border-radius:14px!important; height:52px!important; font-weight:600!important; }
.stButton>button[kind="primary"] { background:#111827!important; color:white!important; border:none!important; }
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

# HEADER
st.html("""
<div style="max-width:1280px; margin:0 auto; padding:16px 32px; display:flex; justify-content:space-between; align-items:center;">
  <div style="display:flex; align-items:center; gap:14px;">
    <div style="width:36px; height:36px; border-radius:10px; background:#111827; display:flex; align-items:center; justify-content:center;">🛡️</div>
    <div style="display:flex; align-items:center; gap:10px;">
      <span style="font-family:Inter,sans-serif; font-weight:700; font-size:20px; color:#111827;">Text<span style="font-family:Fraunces,serif; font-style:italic; font-weight:400;">Shield</span></span>
      <span style="font-family:JetBrains Mono,monospace; font-size:10px; background:#F0EBE3; border:1px solid #E8E2DA; padding:3px 10px; border-radius:999px; color:#78716C;">AI SEMANTIC ENGINE</span>
    </div>
  </div>
  <div style="display:flex; gap:32px; align-items:center; font-family:Inter,sans-serif; font-size:14px; color:#78716C;">
    <span style="color:#111827; border-bottom:2px solid #111827; padding-bottom:4px; font-weight:600;">Product</span><span>How It Works</span><span>Features</span><span>Dashboard Overview</span>
  </div>
  <div style="background:#111827; color:white; padding:10px 20px; border-radius:999px; font-size:14px; font-weight:600;">Get Started ↗</div>
</div>
<div style="height:1px; background:#E8E2DA;"></div>
""")

# HERO - FIXED RENDER
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.html("""
    <div style="padding:64px 0 0 32px;">
      <div style="display:inline-flex; background:#F0EBE3; border:1px solid #E8E2DA; border-radius:999px; padding:6px 14px; font-size:12px; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; color:#57534E;">✦ AI DOCUMENT ANALYSIS</div>
      <h1 style="font-family:Inter,sans-serif; font-size:62px; font-weight:800; line-height:0.92; letter-spacing:-2.8px; color:#111827; margin:28px 0 0;">Understand how<br>similar your<br>documents <span style="font-family:Fraunces,serif; font-style:italic; font-weight:400; color:#8B7E74; border-bottom:1px solid #D6D0C8;">really are.</span></h1>
      <p style="font-family:Inter,sans-serif; font-size:17px; line-height:1.6; color:#78716C; max-width:480px; margin:24px 0 0;">Compare documents using semantic similarity, detect meaningful matching sections, and identify potential plagiarism risk beyond simple keyword matching.</p>
      <div style="display:flex; gap:12px; margin-top:36px;">
        <div style="background:#111827; color:white; padding:15px 26px; border-radius:14px; font-weight:600; font-size:14px;">Analyze Documents →</div>
        <div style="background:white; border:1px solid #E8E2DA; padding:15px 26px; border-radius:14px; font-weight:600; font-size:14px; color:#111827;">See How It Works</div>
      </div>
      <div style="display:flex; gap:28px; margin-top:40px; padding-top:20px; border-top:1px solid #E8E2DA; font-size:13px; color:#78716C;">
        <span>✓ Vector Embeddings</span><span>✓ Section Matching</span><span>✓ PDF / DOCX / TXT</span>
      </div>
    </div>
    """)

with right:
    st.html("""
    <div style="background:white; border:1px solid #E8E2DA; border-radius:20px; box-shadow:0 20px 40px rgba(0,0,0,0.06); padding:22px; margin:32px 32px 0 0;">
      <div style="display:flex; justify-content:space-between; align-items:center; padding-bottom:14px; border-bottom:1px solid #F0EBE3;">
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="display:flex; gap:6px;"><div style="width:11px; height:11px; border-radius:50%; background:#FF8FA3;"></div><div style="width:11px; height:11px; border-radius:50%; background:#FFD93D;"></div><div style="width:11px; height:11px; border-radius:50%; background:#6BCFB8;"></div></div>
          <span style="font-family:JetBrains Mono,monospace; font-size:11px; color:#78716C;">analysis_report_v2.json</span>
        </div>
        <div style="background:#111827; color:white; font-size:10px; font-weight:700; padding:4px 10px; border-radius:999px;">LIVE MOCKUP</div>
      </div>

      <div style="margin-top:18px; display:flex; flex-direction:column; gap:10px;">
        <div style="background:#F9F4EF; border:1px solid #E8E2DA; border-radius:14px; padding:12px 14px; display:flex; justify-content:space-between; align-items:center;">
          <div style="display:flex; gap:10px; align-items:center;">
            <div style="width:28px; height:28px; background:white; border:1px solid #E8E2DA; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:14px;">📄</div>
            <div><div style="font-size:12px; font-weight:700; color:#111827;">Document_A_Research.docx</div><div style="font-size:10px; color:#78716C;">2,450 words • Reference Document</div></div>
          </div>
          <div style="font-size:10px; font-weight:700; color:#57534E;">Doc A</div>
        </div>

        <div style="background:#F9F4EF; border:1px solid #E8E2DA; border-radius:14px; padding:12px 14px; display:flex; justify-content:space-between; align-items:center;">
          <div style="display:flex; gap:10px; align-items:center;">
            <div style="width:28px; height:28px; background:white; border:1px solid #E8E2DA; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:14px;">📄</div>
            <div><div style="font-size:12px; font-weight:700; color:#111827;">Student_Submission_B.pdf</div><div style="font-size:10px; color:#78716C;">2,180 words • Target Comparison</div></div>
          </div>
          <div style="font-size:10px; font-weight:700; color:#57534E;">Doc B</div>
        </div>
      </div>

      <div style="margin-top:16px; background:#111827; border-radius:14px; padding:16px 18px; display:grid; grid-template-columns:repeat(3,1fr); text-align:center;">
        <div><div style="font-size:22px; font-weight:800; color:white;">82%</div><div style="font-size:9px; letter-spacing:0.08em; text-transform:uppercase; color:#9CA3AF; margin-top:3px;">SIMILARITY</div></div>
        <div style="border-left:1px solid rgba(255,255,255,0.12); border-right:1px solid rgba(255,255,255,0.12);"><div style="display:flex; justify-content:center;"><div style="background:#4A1F2A; color:#FF8FA3; font-size:10px; font-weight:700; padding:2px 8px; border-radius:999px;">🛡️ High</div></div><div style="font-size:9px; letter-spacing:0.08em; text-transform:uppercase; color:#9CA3AF; margin-top:5px;">RISK LEVEL</div></div>
        <div><div style="font-size:22px; font-weight:800; color:white;">12</div><div style="font-size:9px; letter-spacing:0.08em; text-transform:uppercase; color:#9CA3AF; margin-top:3px;">MATCHES</div></div>
      </div>

      <div style="margin-top:14px; border:1px solid #E8E2DA; border-radius:12px; padding:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div style="font-size:12px; font-weight:700; color:#111827;">Section #3 Semantic Match</div>
          <div style="background:#FFF0F3; border:1px solid #FFD0D8; color:#C2185B; font-size:10px; font-weight:700; padding:2px 8px; border-radius:999px;">86.2% Overlap</div>
        </div>
        <div style="margin-top:10px; background:#FAF6F1; border:1px solid #F0EBE3; border-radius:8px; padding:10px; font-family:JetBrains Mono,monospace; font-size:10px; line-height:1.5; color:#57534E;">"Deep learning algorithms enable automated diagnostic scans to detect early stage anomalous structures..."</div>
      </div>
    </div>
    """)

# ANALYZE
st.html('<div style="max-width:1120px; margin:48px auto 0; padding:0 32px;"><div style="height:1px; background:#E8E2DA; margin-bottom:36px;"></div><div style="font-family:Fraunces,serif; font-size:28px; font-weight:700; color:#111827;">Analyze Documents</div><div style="color:#78716C; font-size:14px; margin-top:6px;">Paste or upload — same engine powering the mockup</div></div>')

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown('<div style="font-size:11px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#111827; margin:20px 0 10px;">Original Document</div>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["Paste text", "Upload file"])
    src_paste=""; src_file=None
    with t1: src_paste = st.text_area("src", placeholder="Paste source here…", height=150, label_visibility="collapsed", key="src_fix")
    with t2: src_file = st.file_uploader("src", type=['pdf','txt'], label_visibility="collapsed", key="src_f_fix")
with c2:
    st.markdown('<div style="font-size:11px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#111827; margin:20px 0 10px;">Suspected Document</div>', unsafe_allow_html=True)
    t3, t4 = st.tabs(["Paste text", "Upload file"])
    susp_paste=""; susp_file=None
    with t3: susp_paste = st.text_area("susp", placeholder="Paste suspected here…", height=150, label_visibility="collapsed", key="susp_fix")
    with t4: susp_file = st.file_uploader("susp", type=['pdf','txt'], label_visibility="collapsed", key="susp_f_fix")

def get_text(paste, file):
    if paste and paste.strip(): return paste
    if file: return extract_text(file)
    return None

src_doc = get_text(src_paste, src_file)
susp_doc = get_text(susp_paste, susp_file)

if st.button("Analyze Documents →", type="primary", use_container_width=True):
    if not src_doc or not susp_doc:
        st.warning("Add both documents")
    else:
        with st.spinner("Analyzing…"):
            src_sents = split_sentences(src_doc); susp_sents = split_sentences(susp_doc)
            sim = cosine_similarity(model.encode(susp_sents), model.encode(src_sents))
            matches=[]
            for i,row in enumerate(sim):
                j=row.argmax(); score=float(row[j])
                if score>0.75: matches.append({"i":i,"j":j,"score":score,"type":"Exact" if score>0.92 else "Paraphrased","s":susp_sents[i],"o":src_sents[j]})
            pct = round(len(matches)/len(susp_sents)*100,1) if susp_sents else 0
            st.markdown(f"""
            <div style="margin-top:24px; display:grid; grid-template-columns:repeat(3,1fr); gap:14px;">
              <div style="background:white; border:1px solid #E8E2DA; border-radius:14px; padding:18px;"><div style="font-size:28px; font-weight:800;">{pct}%</div><div style="font-size:10px; text-transform:uppercase; color:#78716C;">Similarity</div></div>
              <div style="background:white; border:1px solid #E8E2DA; border-radius:14px; padding:18px;"><div style="font-size:28px; font-weight:800;">{len(matches)}</div><div style="font-size:10px; text-transform:uppercase; color:#78716C;">Matches</div></div>
              <div style="background:#111827; border-radius:14px; padding:18px;"><div style="font-size:28px; font-weight:800; color:white;">{'High' if pct>30 else 'Low'}</div><div style="font-size:10px; text-transform:uppercase; color:#9CA3AF;">Risk</div></div>
            </div>
            """, unsafe_allow_html=True)
            for m in matches[:8]:
                st.markdown(f"""
                <div style="margin-top:12px; background:white; border:1px solid #E8E2DA; border-left:3px solid #111827; border-radius:14px; padding:16px;">
                  <div style="display:flex; justify-content:space-between; font-size:12px;"><b>{m['type']} • {m['score']:.3f}</b><span style="border:1px solid #E8E2DA; border-radius:999px; padding:2px 8px;">S{m['i']+1}→O{m['j']+1}</span></div>
                  <div style="margin-top:8px; background:#FAF6F1; border:1px solid #F0EBE3; border-radius:8px; padding:8px; font-size:11px;">"{m['s']}"</div>
                  <div style="margin-top:6px; font-size:11px; color:#78716C;">→ "{m['o']}"</div>
                </div>
                """, unsafe_allow_html=True)
