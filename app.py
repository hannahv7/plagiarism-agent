
import streamlit as st
import fitz
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
import json, re

try: nltk.data.find('tokenizers/punkt')
except: nltk.download('punkt')
try: nltk.data.find('tokenizers/punkt_tab')
except: nltk.download('punkt_tab')

@st.cache_resource
def load_model(): return SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()

def extract_text(file):
    if file.name.endswith('.pdf'):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([p.get_text() for p in doc])
    else: return file.read().decode('utf-8', errors='ignore')

def split_sentences(t): return [s.strip() for s in nltk.sent_tokenize(t) if len(s.strip())>20]

def get_common(s1,s2,top=4):
    try:
        v = TfidfVectorizer(stop_words='english', max_features=30)
        tf = v.fit_transform([s1,s2]); names=v.get_feature_names_out()
        s1s=tf[0].toarray()[0]; s2s=tf[1].toarray()[0]
        common=[]
        for i,w in enumerate(names):
            if s1s[i]>0 and s2s[i]>0: common.append((w,s1s[i]+s2s[i]))
        return [w for w,_ in sorted(common,key=lambda x:x[1],reverse=True)[:top]]
    except:
        w1=set(re.findall(r'\w+',s1.lower()))-{'the','is','are','was','a','an','and','or','to','of','in','for','with','on','at'}
        w2=set(re.findall(r'\w+',s2.lower()))-{'the','is','are','was','a','an','and','or','to','of','in','for','with','on','at'}
        return list(w1.intersection(w2))[:top]

st.set_page_config(page_title="Explainable Plagiarism Detection", layout="wide")
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
header,#MainMenu,footer{visibility:hidden;}
.block-container{max-width:1120px; padding-top:20px;}
.topbar{display:flex; justify-content:space-between; border-bottom:1px solid #e5e7eb; padding:16px 0; margin-bottom:24px;}
.logo{font-family:'Fraunces'; letter-spacing:2.5px; font-size:12px; font-weight:700; color:#111827;}
.title{font-family:'Fraunces'; font-size:36px; font-weight:800; letter-spacing:-1px; color:#111827; line-height:1.1;}
.sub{color:#6b7280; font-size:14px; margin-top:6px;}
.card{background:white; border:1px solid #e5e7eb; border-radius:12px; padding:20px; box-shadow:0 1px 2px rgba(0,0,0,0.04);}
.metric-num{font-family:'Fraunces'; font-size:32px; font-weight:800; color:#111827;}
.metric-label{font-size:11px; font-weight:600; letter-spacing:0.8px; text-transform:uppercase; color:#6b7280; margin-top:6px;}
textarea{border-radius:12px !important; border:1px solid #e5e7eb !important;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="topbar"><div class="logo">EXPLAINABLE</div></div>', unsafe_allow_html=True)
st.markdown('<div class="title">Explainable Plagiarism Detection</div><div class="sub">Beyond percentage — we show which concepts were copied, where, and why</div>', unsafe_allow_html=True)

st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
st.markdown('**Option 1: Upload Documents**')
c1,c2 = st.columns(2, gap="large")
with c1:
    f1 = st.file_uploader("Original Document (Source)", type=['pdf','txt'], key="o")
with c2:
    f2 = st.file_uploader("Suspected Document", type=['pdf','txt'], key="s")

st.markdown('<div style="text-align:center; color:#9ca3af; margin:16px 0;">— or —</div>', unsafe_allow_html=True)
st.markdown('**Option 2: Paste Text Directly**')
p1,p2 = st.columns(2, gap="large")
with p1:
    t1_paste = st.text_area("Paste Original Text", height=200, placeholder="Paste original source text here... (up to 20,000 chars)", label_visibility="collapsed")
    st.caption("Paste Original Text")
with p2:
    t2_paste = st.text_area("Paste Suspected Text", height=200, placeholder="Paste student / suspected text here...", label_visibility="collapsed")
    st.caption("Paste Suspected Text")

# Determine texts
orig_text = None
susp_text = None
if t1_paste and len(t1_paste.strip())>30:
    orig_text = t1_paste
elif f1:
    orig_text = extract_text(f1)

if t2_paste and len(t2_paste.strip())>30:
    susp_text = t2_paste
elif f2:
    susp_text = extract_text(f2)

if orig_text and susp_text:
    if st.button("Analyze Documents", use_container_width=True, type="primary"):
        with st.spinner("Analyzing with Explainable AI..."):
            s1 = split_sentences(orig_text)
            s2 = split_sentences(susp_text)
            if not s1 or not s2: st.error("Need longer text"); st.stop()
            e1 = model.encode(s1, show_progress_bar=False)
            e2 = model.encode(s2, show_progress_bar=False)
            sim = cosine_similarity(e2,e1)
            matches=[]; plag=0; status=["Original"]*len(s2)
            for i,row in enumerate(sim):
                j=int(row.argmax()); sc=float(row[j])
                stt="Original"
                if sc>0.95: stt="Exact Copy"; plag+=1; status[i]="Exact"
                elif sc>0.82: stt="Paraphrased"; plag+=1; status[i]="Paraphrased"
                if stt!="Original":
                    cc=get_common(s1[j],s2[i])
                    expl=f"Flagged because core concepts {cc} overlap with similarity {sc:.2f}"
                    matches.append({"si":i,"oi":j,"sus":s2[i],"orig":s1[j],"sim":sc,"status":stt,"concepts":cc,"expl":expl})
            overall=plag/len(s2)*100 if s2 else 0
            st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
            m1,m2,m3=st.columns(3)
            with m1: st.markdown(f'<div class="card"><div class="metric-num">{overall:.1f}%</div><div class="metric-label">Overall Plagiarism</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="card"><div class="metric-num">{len(matches)}</div><div class="metric-label">Explainable Matches</div></div>', unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="card"><div class="metric-num" style="font-size:22px;">{"Suspicious" if overall>=30 else "Original"}</div><div class="metric-label">Document Health</div></div>', unsafe_allow_html=True)
            gc,hc = st.columns([1,1.2], gap="large")
            with gc:
                fig=go.Figure(go.Indicator(mode="gauge+number", value=overall, number={'suffix':'%','font':{'family':'Fraunces','size':22}}, gauge={'axis':{'range':[0,100]},'bar':{'color':'#111827'},'bgcolor':'white','steps':[{'range':[0,30],'color':'#f3f4f6'},{'range':[30,60],'color':'#e5e7eb'},{'range':[60,100],'color':'#d1d5db'}]}))
                fig.update_layout(height=240, margin=dict(l=10,r=10,t=20,b=10))
                st.plotly_chart(fig, use_container_width=True)
            with hc:
                st.markdown("**Plagiarism Heatmap**")
                cols="".join([f"<div style='width:36px; height:36px; border-radius:8px; display:inline-flex; align-items:center; justify-content:center; margin:4px; background:{'#111827' if s=='Exact' else '#9ca3af' if s=='Paraphrased' else '#f3f4f6'}; color:{'white' if s!='Original' else '#6b7280'}; font-size:11px;'>{i+1}</div>" for i,s in enumerate(status)])
                st.markdown(f"<div style='display:flex; flex-wrap:wrap;'>{cols}</div>", unsafe_allow_html=True)
            for m in matches:
                st.markdown(f"**{m['status']} {m['sim']:.2f}** - {m['expl']}")
                st.caption(f"S: {m['sus'][:200]}")
                st.caption(f"O: {m['orig'][:200]}")
                st.divider()
else:
    st.info("Upload PDFs or paste text in both columns to analyze")
