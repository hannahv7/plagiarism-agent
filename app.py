import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ── Page config - FORCE LIGHT ──
st.set_page_config(
    page_title="TextShield - Explainable Plagiarism Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Force Light Theme CSS ──
st.markdown("""
<style>
    /* Force light everywhere */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FCFAF8 !important;
        color: #111827 !important;
        color-scheme: light !important;
    }
    [data-testid="stAppViewContainer"] { background: #FCFAF8 !important; }
    
    /* Cards */
    .card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    
    /* Hide Streamlit dark menu */
    [data-testid="stToolbar"] { visibility: hidden; }
    
    /* Text areas - light */
    textarea {
        background: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #E5E7EB !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: #111827 !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        height: 48px !important;
        font-weight: 500 !important;
        border: none !important;
        width: 100%;
    }
    
    /* Metric */
    [data-testid="stMetricValue"] { color: #111827 !important; }
</style>
<meta name="color-scheme" content="light only">
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div style="max-width:1120px; margin:0 auto; padding: 24px 0 8px; display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #E5E7EB;">
    <span style="font-family:'Fraunces', serif; font-size:14px; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; color:#111827;">🛡️ TextShield</span>
    <span style="font-family:Inter, sans-serif; font-size:12px; color:#9CA3AF;">Light Mode • Explainable Detection</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="max-width:1120px; margin:0 auto; padding: 40px 0 32px;">
    <h1 style="font-family:'Fraunces', serif; font-size:36px; font-weight:700; letter-spacing:-1px; color:#111827; margin:0 0 12px;">TextShield</h1>
    <p style="font-family:Inter, sans-serif; font-size:16px; color:#6B7280; margin:0;">Beyond percentage — we show which concepts were copied, where, and why</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ──
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text(file):
    if file.name.endswith('.pdf'):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in doc])
        return text
    else:
        return file.read().decode('utf-8', errors='ignore')

def split_sentences(text):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 20]

# ── Input Row - Paste OR Upload ──
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown("""
    <div style="font-family:'Fraunces', serif; font-size:11px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#111827; margin-bottom:12px;">Original Document (Source)</div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📝 Paste text", "📁 Upload file"])
    source_text_input = ""
    source_file = None
    with tab1:
        source_text_input = st.text_area("Paste source", placeholder="Paste or type your document text here…", height=150, key="src_paste", label_visibility="collapsed")
        if source_text_input:
            st.caption(f"{len(source_text_input.split())} words · {len(source_text_input)} chars")
    with tab2:
        source_file = st.file_uploader("Upload source", type=['pdf','txt'], key="src_file", label_visibility="collapsed")

with col2:
    st.markdown("""
    <div style="font-family:'Fraunces', serif; font-size:11px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#111827; margin-bottom:12px;">Suspected Document</div>
    """, unsafe_allow_html=True)
    
    tab3, tab4 = st.tabs(["📝 Paste text", "📁 Upload file"])
    susp_text_input = ""
    susp_file = None
    with tab3:
        susp_text_input = st.text_area("Paste suspected", placeholder="Paste or type your document text here…", height=150, key="susp_paste", label_visibility="collapsed")
        if susp_text_input:
            st.caption(f"{len(susp_text_input.split())} words · {len(susp_text_input)} chars")
    with tab4:
        susp_file = st.file_uploader("Upload suspected", type=['pdf','txt'], key="susp_file", label_visibility="collapsed")

# ── Determine actual texts (Paste has priority) ──
def get_doc_text(text_input, file_obj):
    if text_input and text_input.strip():
        return text_input
    elif file_obj:
        return extract_text(file_obj)
    return None

source_doc = get_doc_text(source_text_input, source_file)
suspected_doc = get_doc_text(susp_text_input, susp_file)

# ── Analyze ──
if st.button("Analyze Documents", use_container_width=True, type="primary"):
    if not source_doc or not suspected_doc:
        st.warning("Please provide both documents — either paste text or upload files.")
    else:
        with st.spinner("Analyzing documents… Extracting concepts & computing similarity…"):
            src_sents = split_sentences(source_doc)
            susp_sents = split_sentences(suspected_doc)
            
            if len(src_sents) < 1 or len(susp_sents) < 1:
                st.error("Documents too short. Please add more text.")
            else:
                src_emb = model.encode(src_sents)
                susp_emb = model.encode(susp_sents)
                sim_matrix = cosine_similarity(susp_emb, src_emb)
                
                matches = []
                for i, row in enumerate(sim_matrix):
                    j = row.argmax()
                    score = row[j]
                    if score > 0.75:
                        m_type = "Exact" if score > 0.92 else "Paraphrased"
                        # Simple concept extraction: keywords
                        concepts = list(set(re.findall(r'\b\w{5,}\b', susp_sents[i].lower())) )[:3]
                        matches.append({
                            "id": i,
                            "type": m_type,
                            "similarity": round(float(score), 2),
                            "suspectedSentence": i+1,
                            "originalSentence": j+1,
                            "concepts": concepts,
                            "suspectedText": susp_sents[i],
                            "originalText": src_sents[j],
                        })
                
                total = len(susp_sents)
                score_pct = round((len(matches)/total*100) if total else 0, 1)
                
                # ── Results ──
                st.markdown(f"""
                <div style="margin-top:32px; display:grid; grid-template-columns:repeat(3,1fr); gap:16px;">
                    <div class="card"><div style="font-family:'Fraunces', serif; font-size:38px; font-weight:700; color:#111827;">{score_pct}%</div><div style="font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#6B7280; margin-top:8px;">Overall Plagiarism</div><div style="height:3px; background:#F3F4F6; margin-top:16px;"><div style="height:3px; background:#111827; width:{score_pct}%;"></div></div></div>
                    <div class="card"><div style="font-family:'Fraunces', serif; font-size:38px; font-weight:700; color:#111827;">{len(matches)}</div><div style="font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#6B7280; margin-top:8px;">Explainable Matches</div><div style="font-size:12px; color:#9CA3AF; margin-top:4px;">{len([m for m in matches if m['type']=='Exact'])} Exact • {len([m for m in matches if m['type']=='Paraphrased'])} Paraphrased</div></div>
                    <div class="card"><div style="font-family:'Fraunces', serif; font-size:38px; font-weight:700; color:#111827;">{'Suspicious' if score_pct>30 else 'Clean'}</div><div style="font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#6B7280; margin-top:8px;">Document Health</div><div style="font-size:12px; color:#9CA3AF; margin-top:4px;">{'Needs review' if score_pct>30 else 'Original'}</div></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Matches
                st.markdown("<h2 style='font-family:Fraunces, serif; font-size:22px; margin:32px 0 16px; color:#111827;'>Explainable Matches — WHY was it flagged?</h2>", unsafe_allow_html=True)
                
                for m in matches[:10]:
                    st.markdown(f"""
                    <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-left:3px solid #111827; border-radius:12px; padding:20px 24px; margin-bottom:16px;">
                        <div style="display:flex; justify-content:space-between;"><span style="font-size:13px; font-weight:600; color:#111827;">{m['type']} · {m['similarity']} similarity</span><span style="font-size:11px; border:1px solid #E5E7EB; border-radius:999px; padding:2px 10px;">S{m['suspectedSentence']} → O{m['originalSentence']}</span></div>
                        <div style="margin-top:12px; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#6B7280;">Concepts Copied</div>
                        <div style="margin-top:8px; display:flex; gap:6px;">{''.join([f'<span style="font-size:12px; border:1px solid #111827; border-radius:999px; padding:2px 10px;">{c}</span>' for c in m['concepts']])}</div>
                        <div style="margin-top:12px; background:#FAFAF9; border-left:2px solid #E5E7EB; padding:10px 14px; font-size:13px; color:#374151; line-height:1.6;">Flagged because semantic similarity {m['similarity']} detected with {"verbatim reproduction" if m['type']=="Exact" else "paraphrased overlap"}.</div>
                        <div style="margin-top:12px; font-size:12px; color:#6B7280; font-style:italic;"><b style="color:#9CA3AF; font-style:normal;">Suspected:</b> "{m['suspectedText']}"</div>
                        <div style="font-size:12px; color:#6B7280; font-style:italic; margin-top:4px;"><b style="color:#9CA3AF; font-style:normal;">Original:</b> "{m['originalText']}"</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(matches)==0:
                    st.success("✅ No significant plagiarism detected — document appears original!")

# ── Instructions for light mode ──
st.markdown("""
<div style="margin-top:48px; padding-top:24px; border-top:1px solid #E5E7EB; font-size:12px; color:#9CA3AF; text-align:center;">
    💡 This app is forced to Light Mode (#FCFAF8) via config + CSS. No dark inversion.
</div>
""", unsafe_allow_html=True)
