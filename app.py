import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="TextShield — Protect Originality", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    background: #07080B!important; color: #E5E7EB!important;
}
[data-testid="stHeader"] { background: rgba(7,8,11,0.8)!important; backdrop-filter: blur(20px); }
[data-testid="stToolbar"] { display: none; }
footer { visibility: hidden; } #MainMenu { visibility: hidden; }
.glass { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; backdrop-filter: blur(20px); padding: 24px; }
.hero-title { font-family: 'Fraunces', serif; font-size: 56px; font-weight: 700; line-height: 0.95; letter-spacing: -2.5px; background: linear-gradient(100deg, #FFFFFF 20%, #9CA3AF 80%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.gradient-text { background: linear-gradient(90deg, #60A5FA, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.subtle { color: #9CA3AF; font-family: Inter, sans-serif; }
.pill { display: inline-flex; padding: 4px 12px; border-radius: 999px; background: rgba(96,165,250,0.12); border: 1px solid rgba(96,165,250,0.25); color: #93C5FD; font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.04); border-radius: 999px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 999px; font-size: 13px; color: #9CA3AF; }
.stTabs [aria-selected="true"] { background: white!important; color: black!important; }
textarea { background: rgba(0,0,0,0.3)!important; border: 1px solid rgba(255,255,255,0.1)!important; border-radius: 12px!important; color: white!important; }
.stButton>button[kind="primary"] { background: linear-gradient(90deg, #3B82F6, #8B5CF6)!important; color: white!important; border: none!important; border-radius: 999px!important; height: 48px!important; font-weight: 600!important; box-shadow: 0 0 20px rgba(59,130,246,0.4); }
.score-card { background: radial-gradient(120% 120% at 0% 0%, rgba(59,130,246,0.15), transparent 60%), rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 20px; padding: 28px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(): return SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()

def extract_text(file):
    if file.name.endswith('.pdf'):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "".join([p.get_text() for p in doc])
    return file.read().decode('utf-8', errors='ignore')

def split_sentences(text): return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s.strip())>20]

st.markdown("""
<div style="max-width:1200px; margin:0 auto; padding:18px 24px; display:flex; justify-content:space-between; align-items:center; background:rgba(7,8,11,0.7); backdrop-filter:blur(20px); border-bottom:1px solid rgba(255,255,255,0.06);">
  <div style="display:flex; align-items:center; gap:12px;">
    <div style="width:32px; height:32px; border-radius:9px; background:linear-gradient(135deg,#3B82F6,#8B5CF6); display:flex; align-items:center; justify-content:center;">🛡️</div>
    <span style="font-family:'Fraunces',serif; font-weight:700; font-size:18px; color:white;">TextShield</span>
    <span style="font-size:11px; padding:3px 8px; border-radius:999px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12); color:#9CA3AF; margin-left:8px;">BETA</span>
  </div>
  <div style="display:flex; gap:24px; align-items:center; font-size:13px; color:#9CA3AF;"><span>How it works</span><span>Pricing</span><span style="background:white; color:black; padding:8px 16px; border-radius:999px; font-weight:600;">Launch App</span></div>
</div>
<div style="max-width:1200px; margin:0 auto; padding:64px 24px 32px; text-align:center;">
  <div class="pill" style="margin-bottom:24px;">● Beyond percentage — explainable detection</div>
  <div class="hero-title">Protect originality.<br><span class="gradient-text">Detect everything.</span></div>
  <p class="subtle" style="max-width:600px; margin:24px auto 0; font-size:17px; line-height:1.6;">We don't just give you a %. TextShield shows which concepts were copied, where, and why.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="max-width:1120px; margin:0 auto; padding:0 24px;">', unsafe_allow_html=True)
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown('<div style="font-size:12px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#9CA3AF; margin-bottom:12px;">Original Document</div>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["Paste text", "Upload file"])
    src_paste = ""; src_file = None
    with t1: src_paste = st.text_area("src", placeholder="Paste source here…", height=160, label_visibility="collapsed", key="src_p")
    with t2: src_file = st.file_uploader("src file", type=['pdf','txt'], label_visibility="collapsed", key="sf")
with c2:
    st.markdown('<div style="font-size:12px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#9CA3AF; margin-bottom:12px;">Suspected Document</div>', unsafe_allow_html=True)
    t3, t4 = st.tabs(["Paste text", "Upload file"])
    susp_paste = ""; susp_file = None
    with t3: susp_paste = st.text_area("susp", placeholder="Paste suspected here…", height=160, label_visibility="collapsed", key="susp_p")
    with t4: susp_file = st.file_uploader("susp file", type=['pdf','txt'], label_visibility="collapsed", key="sf2")

def get_text(paste, file):
    if paste and paste.strip(): return paste
    if file: return extract_text(file)
    return None

src_doc = get_text(src_paste, src_file)
susp_doc = get_text(susp_paste, susp_file)

if st.button("✦ Analyze Documents", type="primary", use_container_width=True):
    if not src_doc or not susp_doc: st.warning("Add both documents")
    else:
        with st.spinner("Analyzing…"):
            src_sents = split_sentences(src_doc); susp_sents = split_sentences(susp_doc)
            sim = cosine_similarity(model.encode(susp_sents), model.encode(src_sents))
            matches = []
            for i,row in enumerate(sim):
                j = row.argmax(); score = float(row[j])
                if score>0.75: matches.append({"i":i,"j":j,"score":score,"type":"Exact" if score>0.92 else "Paraphrased","s":susp_sents[i],"o":src_sents[j]})
            pct = round(len(matches)/len(susp_sents)*100,1) if susp_sents else 0
            st.markdown(f'<div class="score-card"><div style="font-size:48px; font-weight:700; color:white;">{pct}%</div><div>{len(matches)} matches</div></div>', unsafe_allow_html=True)
            for m in matches[:10]:
                st.markdown(f'<div style="margin-top:16px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-left:3px solid #60A5FA; border-radius:16px; padding:20px;"><b>{m["type"]} • {m["score"]:.2f}</b><br><i>{m["s"]}</i><br><span style="color:#9CA3AF">{m["o"]}</span></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
