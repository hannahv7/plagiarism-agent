import streamlit as st
import fitz
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
import json
import re

# --- NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text(file):
    if file.name.endswith('.pdf'):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    else:
        return file.read().decode('utf-8', errors='ignore')

def split_sentences(text):
    return [s.strip() for s in nltk.sent_tokenize(text) if len(s.strip()) > 20]

def get_common_concepts(s1, s2, top_n=4):
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=30)
        tfidf = vectorizer.fit_transform([s1, s2])
        feature_names = vectorizer.get_feature_names_out()
        scores1 = tfidf[0].toarray()[0]
        scores2 = tfidf[1].toarray()[0]
        common = []
        for i, word in enumerate(feature_names):
            if scores1[i] > 0 and scores2[i] > 0:
                common.append((word, scores1[i] + scores2[i]))
        common_sorted = sorted(common, key=lambda x: x[1], reverse=True)
        return [w for w, _ in common_sorted[:top_n]]
    except:
        w1 = set(re.findall(r'\w+', s1.lower())) - set(['the','is','are','was','were','a','an','and','or','to','of','in','for','with','on','at'])
        w2 = set(re.findall(r'\w+', s2.lower())) - set(['the','is','are','was','were','a','an','and','or','to','of','in','for','with','on','at'])
        return list(w1.intersection(w2))[:top_n]

# --- UI CONFIG ---
st.set_page_config(page_title="Explainable Plagiarism Detector", layout="wide", page_icon="🧠")

# Clean CSS
st.markdown("""
<style>
    .main { background: #fafafa; }
    .stMetric { background: white; padding: 16px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .header-badge { background: #eef2ff; color: #4f46e5; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; display: inline-block; margin-bottom: 8px; }
    .concept-chip { background: #4f46e5; color: white; padding: 4px 10px; border-radius: 20px; font-size: 12px; margin: 3px; display: inline-block; font-weight: 500; }
    .concept-chip-exact { background: #dc2626; }
    .heatmap-box { width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-size: 11px; font-weight: 700; cursor: pointer; transition: transform 0.1s; }
    .heatmap-box:hover { transform: scale(1.1); }
    .match-card { border-left: 5px solid; background: white; padding: 18px; margin: 16px 0; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-badge">🔬 Domain: Education | Uniqueness: Explainable AI</div>', unsafe_allow_html=True)
st.title("Explainable Plagiarism Detection")
st.markdown("**Beyond percentage — we show *which concepts* were copied, *where*, and *why* it was flagged.** Unlike basic checkers, our AI explains its reasoning like a professor would.")
st.divider()

# Sidebar - Clean info
with st.sidebar:
    st.header("📋 How it Works")
    st.markdown("""
    **7 Core Requirements Met:**
    1. ✅ PDF/TXT processing (PyMuPDF + NLTK)
    2. ✅ Semantic embeddings (MiniLM-L6-v2)
    3. ✅ Cosine similarity
    4. ✅ Paraphrase detection (>0.82)
    5. ✅ Matching section identification
    6. ✅ Similarity percentage (Gauge)
    7. ✅ Downloadable report (JSON)
    
    **🧠 Uniqueness 3: Explainable AI**
    - Common Concept Chips
    - Plagiarism Heatmap
    - Natural Language Explanation
    
    **Innovation:** Judges can see *why* AI flagged, not just *what*.
    """)
    st.divider()
    st.info("💡 **Demo Tip:** Upload original.txt and suspected.txt (paraphrased version) to see heatmap + concept overlap.")

# Upload Area
col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("#### 📄 Original Document")
    file1 = st.file_uploader("Upload original", type=['pdf','txt'], key="orig", label_visibility="collapsed")
    if file1:
        st.success(f"Loaded: {file1.name}")
with col2:
    st.markdown("#### 🔍 Suspected Document")
    file2 = st.file_uploader("Upload suspected", type=['pdf','txt'], key="susp", label_visibility="collapsed")
    if file2:
        st.success(f"Loaded: {file2.name}")

if file1 and file2:
    if st.button("✨ Analyze with Explainable AI", type="primary", use_container_width=True):
        with st.spinner("Analyzing semantics + extracting concepts + building explainable report..."):
            text1 = extract_text(file1)
            text2 = extract_text(file2)
            sents1 = split_sentences(text1)
            sents2 = split_sentences(text2)

            if not sents1 or not sents2:
                st.error("Not enough sentences found. Try longer documents.")
                st.stop()

            emb1 = model.encode(sents1, show_progress_bar=False)
            emb2 = model.encode(sents2, show_progress_bar=False)
            sim_matrix = cosine_similarity(emb2, emb1)

            matches = []
            plag_count = 0
            sentence_status = ["Original"] * len(sents2)

            for i, row in enumerate(sim_matrix):
                max_idx = int(row.argmax())
                max_score = float(row[max_idx])
                status = "Original"
                if max_score > 0.95:
                    status = "Exact Copy"
                    plag_count += 1
                    sentence_status[i] = "Exact"
                elif max_score > 0.82:
                    status = "Paraphrased"
                    plag_count += 1
                    sentence_status[i] = "Paraphrased"

                if status != "Original":
                    common_concepts = get_common_concepts(sents1[max_idx], sents2[i])
                    explanation = f"Flagged because core concepts {common_concepts} overlap with high semantic similarity ({max_score:.2f}). The model detected preserved meaning despite wording changes."
                    matches.append({
                        "suspected_idx": i,
                        "original_idx": max_idx,
                        "suspected_sentence": sents2[i],
                        "original_sentence": sents1[max_idx],
                        "similarity": max_score,
                        "status": status,
                        "common_concepts": common_concepts,
                        "explanation": explanation
                    })

            overall = (plag_count / len(sents2) * 100) if sents2 else 0

            # Results Header
            st.divider()
            st.subheader("📊 Analysis Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Overall Plagiarism", f"{overall:.1f}%", delta=f"{len(matches)} matches" if matches else "Clean")
            c2.metric("Explainable Matches", len(matches))
            c3.metric("Document Health", "Original" if overall < 30 else "Suspicious" if overall < 60 else "High Risk", delta="Needs review" if overall>30 else "Good")

            # Two columns: Gauge + Heatmap
            g_col, h_col = st.columns([1, 1.5], gap="large")
            with g_col:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=overall,
                    title={'text': "Plagiarism Score", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#dc2626" if overall>60 else "#f59e0b" if overall>30 else "#10b981", 'thickness': 0.3},
                        'bgcolor': "white",
                        'steps': [
                            {'range': [0,30], 'color': "#d1fae5"},
                            {'range': [30,60], 'color': "#fef3c7"},
                            {'range': [60,100], 'color': "#fee2e2"}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': 80}
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="white", font={'family': "Inter"})
                st.plotly_chart(fig, use_container_width=True)
            
            with h_col:
                st.markdown("##### 🔥 Plagiarism Heatmap")
                st.caption("Visual map of where copying occurs in suspected document. Green=Original, Orange=Paraphrased, Red=Exact.")
                heatmap_html = "<div style='display:flex; gap:6px; flex-wrap:wrap; margin:12px 0; padding:16px; background:white; border-radius:12px; box-shadow:0 1px 3px rgba(0,0,0,0.08)'>"
                for idx, status in enumerate(sentence_status):
                    color = "#10b981" if status == "Original" else "#f59e0b" if status == "Paraphrased" else "#ef4444"
                    heatmap_html += f"<div class='heatmap-box' title='Sentence {idx+1}: {status}' style='background:{color};'>{idx+1}</div>"
                heatmap_html += "</div>"
                st.markdown(heatmap_html, unsafe_allow_html=True)
                st.markdown("<div style='display:flex; gap:12px; font-size:12px; margin-top:8px'><span>🟢 Original</span><span>🟠 Paraphrased</span><span>🔴 Exact Copy</span></div>", unsafe_allow_html=True)

            # Explainable Matches
            st.divider()
            st.subheader("🔍 Explainable Matches - WHY was it flagged?")
            st.caption("Unlike other tools that just show %, we explain the reasoning behind each flag.")
            
            if not matches:
                st.success("✅ No plagiarism detected - Document appears original!")
                st.balloons()
            else:
                tabs = st.tabs([f"{m['status']} #{i+1} (Score {m['similarity']:.2f})" for i,m in enumerate(matches)])
                for idx, (tab, m) in enumerate(zip(tabs, matches)):
                    with tab:
                        color = "#dc2626" if m['status'] == "Exact Copy" else "#d97706"
                        bg = "#fef2f2" if m['status'] == "Exact Copy" else "#fffbeb"
                        border = color
                        
                        concepts_html = "".join([f"<span class='concept-chip {'concept-chip-exact' if m['status']=='Exact Copy' else ''}'>{c}</span>" for c in m['common_concepts']])
                        
                        st.markdown(f"""
                        <div class="match-card" style="border-left-color:{border}; background:{bg};">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px">
                                <span style="font-weight:700; color:{color}; font-size:15px">{m['status']} • Similarity {m['similarity']:.2f}</span>
                                <span style="background:white; padding:4px 10px; border-radius:20px; font-size:12px; border:1px solid #e5e7eb">S{m['suspected_idx']+1} → O{m['original_idx']+1}</span>
                            </div>
                            <div style="margin:10px 0"><b>🧠 Common Concepts Copied:</b><br>{concepts_html}</div>
                            <div style="font-size:13px; color:#374151; background:white; padding:10px; border-radius:8px; margin:10px 0"><b>💬 Explanation:</b> {m['explanation']}</div>
                            <hr style="margin:12px 0; border:none; border-top:1px solid #e5e7eb">
                            <div style="margin-bottom:8px"><b>Suspected:</b> {m['suspected_sentence']}</div>
                            <div><b>Original:</b> <i style="color:#6b7280">{m['original_sentence']}</i></div>
                        </div>
                        """, unsafe_allow_html=True)

            # Download
            st.divider()
            report = {
                "domain": "Education",
                "uniqueness": "Explainable AI - Shows WHY flagged with common concepts + heatmap + explanation",
                "overall_similarity": round(overall, 2),
                "heatmap": sentence_status,
                "matches": matches,
                "requirements_met": ["PDF/TXT processing", "Semantic embeddings MiniLM", "Cosine similarity", "Paraphrase detection 0.82", "Matching sections", "Similarity %", "JSON report"]
            }
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button("📄 Download Explainable Report (JSON)", json.dumps(report, indent=2), "explainable_plagiarism_report.json", use_container_width=True)
            with col_d2:
                st.download_button("📝 Download Summary (TXT)", f"Plagiarism Report\nOverall: {overall:.1f}%\nMatches: {len(matches)}\nHealth: {'Original' if overall < 30 else 'Suspicious' if overall < 60 else 'High Risk'}\n\n" + "\n".join([f"- S{m['suspected_idx']+1}: {m['status']} ({m['similarity']:.2f}) Concepts: {m['common_concepts']}" for m in matches]), "summary.txt", use_container_width=True)

else:
    st.info("👆 Upload both documents above to start explainable analysis")
    with st.expander("🎯 What makes this Uniqueness 3?"):
        st.markdown("""
        **Most plagiarism checkers:** Show 87% plagiarized. That's it.
        
        **Our Explainable AI:**
        1. **Concept Chips:** Shows *which ideas* were copied (e.g., 'neural', 'network', 'learning')
        2. **Heatmap:** Shows *where* in document copying occurs - green/orange/red grid
        3. **Natural Explanation:** Shows *why* AI flagged: "Flagged because concepts ['machine','learning'] overlap with 0.91 similarity"
        
        **Judges see:** Not just a detector, but an *explainer* - like a professor marking a paper.
        """)
