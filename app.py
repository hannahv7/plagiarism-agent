
import io
import re
import html
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from docx import Document

st.set_page_config(
    page_title="TextShield",
    page_icon="♢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# EXACT TEXTSHIELD UI — independently recreated in Streamlit
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --cream:#f6f2ec;
    --paper:#fbfaf7;
    --white:#ffffff;
    --ink:#151515;
    --muted:#77736d;
    --line:#ded9d0;
    --soft:#ebe6de;
    --green:#d9eadb;
    --green-ink:#31583a;
}

* { box-sizing:border-box; }

html { scroll-behavior:smooth; }

body, .stApp {
    background:var(--cream) !important;
    color:var(--ink) !important;
    font-family:'DM Sans',sans-serif !important;
}

.stApp > header { display:none !important; }

.block-container {
    max-width:1320px !important;
    padding:0 42px 60px !important;
}

[data-testid="stSidebar"] { display:none !important; }
[data-testid="collapsedControl"] { display:none !important; }

h1,h2,h3,p { margin-top:0; }

/* ---------------- HEADER ---------------- */

.ts-header {
    height:92px;
    display:grid;
    grid-template-columns:1fr auto 1fr;
    align-items:center;
    border-bottom:1px solid var(--line);
}

.ts-brand {
    display:flex;
    align-items:center;
    gap:11px;
    min-width:0;
}

.ts-shield {
    width:31px;
    height:31px;
    border:2px solid var(--ink);
    border-radius:9px 9px 13px 13px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:20px;
    line-height:1;
    transform:rotate(0deg);
}

.ts-brand-name {
    font-family:'Space Grotesk',sans-serif;
    font-size:20px;
    font-weight:700;
    letter-spacing:-.5px;
}

.ts-pill {
    margin-left:3px;
    padding:6px 9px;
    border:1px solid #c9c3ba;
    border-radius:999px;
    font-size:8px;
    font-weight:700;
    letter-spacing:1.1px;
    color:#68635c;
    white-space:nowrap;
}

.ts-nav {
    display:flex;
    align-items:center;
    gap:37px;
    font-size:13px;
    color:#625e58;
    white-space:nowrap;
}

.ts-nav-item {
    position:relative;
    cursor:default;
}

.ts-nav-item.selected {
    color:var(--ink);
    font-weight:600;
}

.ts-nav-item.selected:after {
    content:'';
    position:absolute;
    left:0;
    right:0;
    bottom:-11px;
    height:2px;
    background:var(--ink);
}

.ts-get {
    justify-self:end;
    text-decoration:none !important;
    color:#fff !important;
    background:var(--ink);
    padding:12px 17px;
    border-radius:6px;
    font-size:12px;
    font-weight:600;
}

.ts-get span { margin-left:7px; font-size:15px; }

/* ---------------- HERO ---------------- */

.hero-wrap {
    min-height:650px;
    display:grid;
    grid-template-columns:1.02fr .98fr;
    gap:62px;
    align-items:center;
    padding:72px 0 78px;
}

.eyebrow {
    display:inline-flex;
    align-items:center;
    font-size:10px;
    letter-spacing:1.55px;
    font-weight:700;
    color:#716c65;
    margin-bottom:25px;
}

.hero-title {
    font-family:'Space Grotesk',sans-serif;
    font-size:66px;
    line-height:.98;
    letter-spacing:-3.5px;
    font-weight:600;
    max-width:650px;
    margin-bottom:28px;
}

.hero-title em {
    font-style:normal;
    position:relative;
}

.hero-title em:after {
    content:'';
    position:absolute;
    left:0;
    right:0;
    bottom:-5px;
    height:4px;
    background:#c7c0b7;
    transform:rotate(-1deg);
    border-radius:5px;
}

.hero-copy {
    max-width:555px;
    color:#706b64;
    font-size:15px;
    line-height:1.7;
    margin-bottom:30px;
}

.ts-btn {
    display:inline-block;
    text-decoration:none !important;
    padding:14px 19px;
    border-radius:6px;
    font-size:12px;
    font-weight:600;
    margin-right:8px;
}

.ts-btn-black {
    background:var(--ink);
    color:white !important;
}

.ts-btn-light {
    background:#fff;
    color:var(--ink) !important;
    border:1px solid #d8d2c9;
}

.hero-features {
    display:flex;
    gap:23px;
    margin-top:29px;
    flex-wrap:wrap;
}

.feature {
    font-size:10px;
    color:#6e6962;
    display:flex;
    align-items:center;
    gap:7px;
}

.check {
    width:16px;
    height:16px;
    border-radius:50%;
    background:#dfddd7;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    font-size:9px;
    color:#4f4b45;
}

/* ---------------- MOCKUP ---------------- */

.mock {
    background:#fff;
    border:1px solid #d7d1c8;
    border-radius:13px;
    box-shadow:0 22px 50px rgba(33,29,24,.09);
    overflow:hidden;
    transform:rotate(.15deg);
}

.mock-head {
    height:51px;
    display:grid;
    grid-template-columns:90px 1fr auto;
    align-items:center;
    border-bottom:1px solid #e5e0d8;
    padding:0 18px;
}

.mock-dots { display:flex; gap:6px; }

.dot {
    width:8px;
    height:8px;
    border-radius:50%;
    display:block;
}

.dot.r { background:#e37d72; }
.dot.y { background:#e1bd62; }
.dot.g { background:#77b77e; }

.mock-file {
    font-size:10px;
    color:#77716a;
}

.mock-live {
    font-size:8px;
    letter-spacing:1px;
    font-weight:700;
    color:#6d6861;
}

.mock-doc {
    margin:15px 17px 0;
    padding:15px;
    background:#faf9f7;
    border:1px solid #e4dfd7;
    border-radius:8px;
    display:flex;
    justify-content:space-between;
    align-items:center;
}

.mock-doc-left {
    display:flex;
    align-items:center;
    gap:11px;
}

.doc-icon {
    width:28px;
    height:34px;
    border:1.5px solid #a7a199;
    border-radius:3px;
    background:#fff;
    position:relative;
}

.doc-icon:after {
    content:'';
    position:absolute;
    width:8px;
    height:8px;
    right:-1px;
    top:-1px;
    border-left:1.5px solid #a7a199;
    border-bottom:1.5px solid #a7a199;
    background:#fff;
}

.mock-doc-title {
    font-size:11px;
    font-weight:600;
    margin-bottom:4px;
}

.mock-meta {
    font-size:8.5px;
    color:#8a847c;
}

.mock-label {
    font-size:8px;
    padding:5px 7px;
    background:#e9e5df;
    border-radius:4px;
    color:#6b665f;
}

.mock-score {
    display:grid;
    grid-template-columns:repeat(3,1fr);
    margin:16px 17px;
    border:1px solid #e3ded6;
    border-radius:8px;
    overflow:hidden;
}

.score-cell {
    min-height:90px;
    padding:15px 10px;
    text-align:center;
    border-right:1px solid #e3ded6;
}

.score-cell:last-child { border-right:0; }

.score-number {
    font-family:'Space Grotesk',sans-serif;
    font-size:29px;
    font-weight:600;
    letter-spacing:-1px;
}

.score-label {
    margin-top:6px;
    font-size:7.5px;
    letter-spacing:1px;
    color:#88827a;
    font-weight:700;
}

.risk-badge {
    display:inline-block;
    margin-top:2px;
    padding:8px 10px;
    border-radius:999px;
    background:#f0dfdf;
    color:#714a4a;
    font-size:10px;
    font-weight:600;
}

.mock-match {
    margin:0 17px 17px;
    padding:15px;
    background:#faf9f7;
    border:1px solid #e3ded6;
    border-radius:8px;
}

.match-top {
    display:flex;
    justify-content:space-between;
    gap:15px;
    font-size:9px;
    font-weight:600;
}

.overlap { color:#78716a; }

.mock-quote {
    margin-top:11px;
    padding:12px;
    background:#f0ede7;
    border-left:3px solid #9b958d;
    color:#69645d;
    font-size:10px;
    line-height:1.55;
    font-style:italic;
}

/* ---------------- ANALYZER ---------------- */

.analyzer-section {
    scroll-margin-top:25px;
    padding:86px 0 95px;
}

.analyzer-title {
    text-align:center;
    font-family:'Space Grotesk',sans-serif;
    font-size:43px;
    letter-spacing:-1.8px;
    margin-bottom:10px;
}

.analyzer-sub {
    text-align:center;
    color:#77716a;
    font-size:14px;
    margin-bottom:43px;
}

.upload-card {
    background:#fff;
    border:1px solid #ddd7ce;
    border-radius:11px;
    padding:25px;
    min-height:300px;
}

.upload-title {
    font-family:'Space Grotesk',sans-serif;
    font-size:18px;
    font-weight:600;
}

.upload-sub {
    color:#88827a;
    font-size:11px;
    margin-top:4px;
    margin-bottom:22px;
}

.upload-help {
    font-size:10px;
    color:#88827a;
    margin-top:-8px;
    margin-bottom:16px;
}

.paste-label {
    font-size:11px;
    font-weight:600;
    margin:18px 0 7px;
}

div[data-testid="stFileUploader"] {
    border:1px dashed #cfc8bf !important;
    border-radius:8px !important;
    background:#faf9f7 !important;
    padding:5px !important;
}

div[data-testid="stFileUploader"] section {
    border:0 !important;
    padding:10px !important;
}

div[data-testid="stFileUploader"] button {
    font-size:11px !important;
}

textarea {
    border-radius:7px !important;
    border:1px solid #d8d2ca !important;
    background:#fff !important;
    font-size:12px !important;
}

.analyze-row {
    text-align:center;
    margin:35px 0 0;
}

.analyze-row button {
    min-width:190px !important;
}

/* ---------------- RESULTS ---------------- */

.results-wrap {
    margin-top:55px;
    padding:28px;
    background:#fff;
    border:1px solid #ddd7ce;
    border-radius:11px;
}

.results-title {
    font-family:'Space Grotesk',sans-serif;
    font-size:26px;
    margin-bottom:22px;
}

.result-card {
    background:#faf9f7;
    border:1px solid #e2ddd5;
    border-radius:8px;
    padding:18px;
    text-align:center;
}

.result-number {
    font-family:'Space Grotesk',sans-serif;
    font-size:32px;
    font-weight:600;
}

.result-label {
    font-size:8px;
    letter-spacing:1px;
    color:#817b73;
    margin-top:4px;
}

.match-item {
    padding:16px 0;
    border-bottom:1px solid #e4dfd7;
}

.match-item:last-child { border-bottom:0; }

.match-name {
    font-size:12px;
    font-weight:600;
}

.match-score {
    font-size:10px;
    color:#77716a;
    margin-top:4px;
}

.match-text {
    margin-top:9px;
    font-size:11px;
    color:#68635c;
    line-height:1.55;
}

/* ---------------- HOW IT WORKS ---------------- */

.how-section {
    scroll-margin-top:25px;
    padding:80px 0 95px;
    border-top:1px solid var(--line);
}

.how-title {
    text-align:center;
    font-family:'Space Grotesk',sans-serif;
    font-size:43px;
    letter-spacing:-1.8px;
    margin-bottom:10px;
}

.how-sub {
    text-align:center;
    color:#77716a;
    font-size:14px;
    margin-bottom:45px;
}

.how-grid {
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:18px;
}

.how-card {
    background:#fff;
    border:1px solid #ddd7ce;
    border-radius:10px;
    padding:27px;
    min-height:245px;
}

.how-number {
    font-family:'Space Grotesk',sans-serif;
    font-size:35px;
    font-weight:600;
    letter-spacing:-1px;
    color:#a19a91;
    margin-bottom:35px;
}

.how-card h3 {
    font-family:'Space Grotesk',sans-serif;
    font-size:18px;
    margin-bottom:11px;
}

.how-card p {
    color:#77716a;
    font-size:12px;
    line-height:1.7;
}

/* ---------------- FOOTER ---------------- */

.ts-footer {
    border-top:1px solid var(--line);
    padding:28px 0 5px;
    color:#817b73;
    font-size:10px;
}

/* Streamlit buttons */
.stButton > button {
    background:#151515 !important;
    color:#fff !important;
    border:0 !important;
    border-radius:6px !important;
    font-weight:600 !important;
    font-size:12px !important;
    padding:11px 22px !important;
}

.stButton > button:hover {
    background:#2b2b2b !important;
    color:#fff !important;
}

@media (max-width: 1050px) {
    .ts-header { grid-template-columns:1fr auto; }
    .ts-nav { display:none; }
    .hero-wrap { grid-template-columns:1fr; }
    .hero-title { font-size:56px; }
}

@media (max-width: 700px) {
    .block-container { padding:0 18px 40px !important; }
    .ts-header { height:75px; }
    .ts-pill { display:none; }
    .hero-wrap { padding:55px 0; }
    .hero-title { font-size:43px; letter-spacing:-2.3px; }
    .how-grid { grid-template-columns:1fr; }
    .mock-score { grid-template-columns:1fr; }
    .score-cell { border-right:0; border-bottom:1px solid #e3ded6; }
    .score-cell:last-child { border-bottom:0; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="ts-header">
    <div class="ts-brand">
        <div class="ts-shield">♢</div>
        <div class="ts-brand-name">TextShield</div>
        <div class="ts-pill">AI SEMANTIC ENGINE</div>
    </div>

    <div class="ts-nav">
        <div class="ts-nav-item selected">Product</div>
        <div class="ts-nav-item">How It Works</div>
        <div class="ts-nav-item">Features</div>
        <div class="ts-nav-item">Dashboard Overview</div>
    </div>

    <a class="ts-get" href="#analyzer">
        Get Started <span>↗</span>
    </a>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HERO + MOCKUP
# ============================================================

st.markdown("""
<div class="hero-wrap">
    <div>
        <div class="eyebrow">✣ &nbsp; AI DOCUMENT ANALYSIS</div>

        <h1 class="hero-title">
            Understand how similar
            <br>
            your documents
            <em>really are.</em>
        </h1>

        <div class="hero-copy">
            Compare documents using semantic similarity,
            detect meaningful matching sections, and identify
            potential plagiarism risk beyond simple keyword
            matching.
        </div>

        <a class="ts-btn ts-btn-black" href="#analyzer">
            Analyze Documents &nbsp; →
        </a>

        <a class="ts-btn ts-btn-light" href="#how-it-works">
            See How It Works
        </a>

        <div class="hero-features">
            <div class="feature">
                <span class="check">✓</span>
                Vector Embeddings
            </div>
            <div class="feature">
                <span class="check">✓</span>
                Section Matching
            </div>
            <div class="feature">
                <span class="check">✓</span>
                PDF / DOCX / TXT
            </div>
        </div>
    </div>

    <div class="mock">
        <div class="mock-head">
            <div class="mock-dots">
                <span class="dot r"></span>
                <span class="dot y"></span>
                <span class="dot g"></span>
            </div>

            <div class="mock-file">analysis_report_v2.json</div>
            <div class="mock-live">LIVE MOCKUP</div>
        </div>

        <div class="mock-doc">
            <div class="mock-doc-left">
                <div class="doc-icon"></div>
                <div>
                    <div class="mock-doc-title">Document_A_Research.docx</div>
                    <div class="mock-meta">2,450 words • Reference Document</div>
                </div>
            </div>
            <div class="mock-label">Doc A</div>
        </div>

        <div class="mock-doc">
            <div class="mock-doc-left">
                <div class="doc-icon"></div>
                <div>
                    <div class="mock-doc-title">Student_Submission_B.pdf</div>
                    <div class="mock-meta">2,180 words • Target Comparison</div>
                </div>
            </div>
            <div class="mock-label">Doc B</div>
        </div>

        <div class="mock-score">
            <div class="score-cell">
                <div class="score-number">82%</div>
                <div class="score-label">SIMILARITY</div>
            </div>
            <div class="score-cell">
                <div class="risk-badge">♙ &nbsp;High</div>
                <div class="score-label">RISK LEVEL</div>
            </div>
            <div class="score-cell">
                <div class="score-number">12</div>
                <div class="score-label">MATCHES</div>
            </div>
        </div>

        <div class="mock-match">
            <div class="match-top">
                <span>Section #3 Semantic Match</span>
                <span class="overlap">86.2% Overlap</span>
            </div>
            <div class="mock-quote">
                "Deep learning algorithms enable
                automated diagnostic scans to detect
                early stage anomalous structures..."
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TEXT EXTRACTION / ANALYSIS
# ============================================================

def extract_text(uploaded):
    if uploaded is None:
        return ""

    data = uploaded.getvalue()
    name = uploaded.name.lower()

    try:
        if name.endswith(".txt"):
            return data.decode("utf-8", errors="ignore")

        if name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(data))
            return "\n".join((page.extract_text() or "") for page in reader.pages)

        if name.endswith(".docx"):
            doc = Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)

    except Exception as exc:
        st.error(f"Could not read {uploaded.name}: {exc}")

    return ""


def clean_text(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def split_sections(text):
    text = clean_text(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sections = []
    chunk = []

    for sentence in sentences:
        chunk.append(sentence)
        if len(" ".join(chunk).split()) >= 70:
            sections.append(" ".join(chunk))
            chunk = []

    if chunk:
        sections.append(" ".join(chunk))

    return sections


def analyze_documents(text_a, text_b):
    a = clean_text(text_a)
    b = clean_text(text_b)

    if not a or not b:
        return None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=15000,
    )

    matrix = vectorizer.fit_transform([a, b])
    overall = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100)

    sections_a = split_sections(a)
    sections_b = split_sections(b)

    matches = []

    if sections_a and sections_b:
        section_matrix = vectorizer.transform(sections_a + sections_b)
        a_matrix = section_matrix[:len(sections_a)]
        b_matrix = section_matrix[len(sections_a):]
        scores = cosine_similarity(a_matrix, b_matrix)

        for i in range(len(sections_a)):
            j = int(np.argmax(scores[i]))
            score = float(scores[i][j] * 100)

            if score >= 60:
                matches.append({
                    "section": i + 1,
                    "score": score,
                    "text": sections_a[i],
                    "comparison": sections_b[j],
                })

    matches.sort(key=lambda x: x["score"], reverse=True)

    if overall >= 75:
        risk = "High"
    elif overall >= 45:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "similarity": overall,
        "risk": risk,
        "matches": matches,
    }

# ============================================================
# ANALYZER
# ============================================================

st.markdown('<div id="analyzer" class="analyzer-section"></div>', unsafe_allow_html=True)

st.markdown("""
<h2 class="analyzer-title">Analyze your documents</h2>

<div class="analyzer-sub">
    Upload two documents and compare
    their semantic similarity.
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown("""
    <div class="upload-card">
        <div class="upload-title">Document A</div>
        <div class="upload-sub">Reference document</div>
    """, unsafe_allow_html=True)

    file_a = st.file_uploader(
        "Upload Document A",
        type=["pdf", "docx", "txt"],
        key="document_a",
        label_visibility="visible",
    )

    st.markdown(
        '<div class="upload-help">200MB per file • PDF, DOCX, TXT</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="paste-label">Paste Document A</div>', unsafe_allow_html=True)

    paste_a = st.text_area(
        "Paste Document A",
        height=145,
        key="paste_a",
        label_visibility="collapsed",
        placeholder="Paste your reference document text here...",
    )

    st.markdown("</div>", unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="upload-card">
        <div class="upload-title">Document B</div>
        <div class="upload-sub">Target comparison</div>
    """, unsafe_allow_html=True)

    file_b = st.file_uploader(
        "Upload Document B",
        type=["pdf", "docx", "txt"],
        key="document_b",
        label_visibility="visible",
    )

    st.markdown(
        '<div class="upload-help">200MB per file • PDF, DOCX, TXT</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="paste-label">Paste Document B</div>', unsafe_allow_html=True)

    paste_b = st.text_area(
        "Paste Document B",
        height=145,
        key="paste_b",
        label_visibility="collapsed",
        placeholder="Paste your target document text here...",
    )

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="analyze-row">', unsafe_allow_html=True)
clicked = st.button("Analyze Documents", use_container_width=False)
st.markdown("</div>", unsafe_allow_html=True)

if clicked:
    text_a = extract_text(file_a) if file_a else paste_a
    text_b = extract_text(file_b) if file_b else paste_b

    if not clean_text(text_a):
        st.warning("Please upload or paste Document A.")
    elif not clean_text(text_b):
        st.warning("Please upload or paste Document B.")
    else:
        with st.spinner("Analyzing documents..."):
            result = analyze_documents(text_a, text_b)

        st.markdown('<div class="results-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="results-title">Analysis Results</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-number">{result["similarity"]:.1f}%</div>
                <div class="result-label">SIMILARITY</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-number">{result["risk"]}</div>
                <div class="result-label">RISK LEVEL</div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-number">{len(result["matches"])}</div>
                <div class="result-label">MATCHES</div>
            </div>
            """, unsafe_allow_html=True)

        if result["matches"]:
            st.markdown("#### Matching Sections")

            for idx, match in enumerate(result["matches"][:12], start=1):
                st.markdown(f"""
                <div class="match-item">
                    <div class="match-name">
                        Section #{match["section"]} Semantic Match
                    </div>
                    <div class="match-score">
                        {match["score"]:.1f}% Overlap
                    </div>
                    <div class="match-text">
                        {html.escape(match["text"][:700])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No sections crossed the semantic-match threshold.")

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown('<div id="how-it-works" class="how-section"></div>', unsafe_allow_html=True)

st.markdown("""
<h2 class="how-title">
    How It Works
</h2>

<div class="how-sub">
    TextShield analyzes your documents in three simple stages.
</div>

<div class="how-grid">

    <div class="how-card">
        <div class="how-number">01</div>

        <h3>
            Upload Documents
        </h3>

        <p>
            Upload your reference and target documents
            in PDF, DOCX, or TXT format, or paste the
            text directly.
        </p>
    </div>

    <div class="how-card">
        <div class="how-number">02</div>

        <h3>
            Analyze Similarity
        </h3>

        <p>
            TextShield processes the documents,
            divides them into sections, and calculates
            similarity between their content.
        </p>
    </div>

    <div class="how-card">
        <div class="how-number">03</div>

        <h3>
            Review Matches
        </h3>

        <p>
            Review the overall similarity score,
            risk level, matching sections, and
            similarity matrix.
        </p>
    </div>

</div>
""", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="ts-footer">
    TextShield • AI Semantic Document Analysis
</div>
""", unsafe_allow_html=True)
