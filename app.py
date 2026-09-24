import io
import re
import html
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document
except ImportError:
    Document = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TextShield — AI Semantic Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:ital,wght@500;600&display=swap');


/* ============================================================
   GLOBAL
   ============================================================ */

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif !important;
}

.stApp {
    background: #f6f2ec;
    color: #181818;
}

.block-container {
    max-width: 1410px !important;
    padding: 0 0 80px !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

section[data-testid="stSidebar"] {
    display: none;
}

[data-testid="stToolbar"] {
    display: none;
}

[data-testid="stDecoration"] {
    display: none;
}

footer {
    visibility: hidden;
}


/* ============================================================
   HEADER
   ============================================================ */

.ts-header {
    height: 84px;
    border-bottom: 1px solid #ddd8d1;

    display: grid;
    grid-template-columns: 1fr auto 1fr;

    align-items: center;
}


.ts-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}


.ts-shield {
    width: 44px;
    height: 44px;

    background: #171717;
    color: white;

    border-radius: 14px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 22px;

    box-shadow: 0 4px 10px rgba(0,0,0,.10);
}


.ts-brand-name {
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -1px;
}


.ts-pill {
    background: #e9e5de;

    border: 1px solid #dfdad3;

    color: #716c66;

    border-radius: 20px;

    padding: 6px 11px;

    font-family: "DM Mono", monospace;

    font-size: 10px;
}


/* Navigation */

.ts-nav {
    display: flex;
    align-items: center;

    gap: 37px;

    height: 100%;

    color: #716c66;

    font-size: 15px;
}


.ts-nav .selected {
    color: #171717;

    position: relative;
}


.ts-nav .selected:after {
    content: "";

    position: absolute;

    left: 0;
    right: 0;

    bottom: -31px;

    height: 2px;

    background: #171717;
}


/* Get Started */

.ts-get {
    justify-self: end;

    background: #171717;

    color: white !important;

    text-decoration: none !important;

    padding: 12px 20px;

    border-radius: 15px;

    font-size: 14px;

    font-weight: 600;

    box-shadow: 0 4px 9px rgba(0,0,0,.10);
}

.ts-get span {
    font-size: 17px;
    margin-left: 8px;
}


/* ============================================================
   HERO
   ============================================================ */

.hero-wrap {

    display: grid;

    grid-template-columns: 1.03fr .83fr;

    gap: 70px;

    padding-top: 92px;

    align-items: center;
}


/* Eyebrow */

.eyebrow {

    display: inline-flex;

    align-items: center;

    background: #ebe7e1;

    border: 1px solid #dfd9d2;

    border-radius: 22px;

    padding: 8px 14px;

    font-size: 13px;

    font-weight: 500;

    margin-bottom: 32px;
}


/* Main heading */

.hero-title {

    margin: 0;

    font-size: 68px;

    line-height: .98;

    letter-spacing: -4px;

    font-weight: 700;
}


.hero-title em {

    font-family: "Playfair Display", serif;

    color: #77716b;

    font-weight: 500;

    letter-spacing: -3px;

    white-space: nowrap;
}


/* Description */

.hero-copy {

    margin: 32px 0 36px;

    max-width: 710px;

    color: #6d6862;

    font-size: 19px;

    line-height: 1.65;
}


/* Buttons */

.ts-btn {

    display: inline-flex;

    align-items: center;

    justify-content: center;

    min-width: 255px;

    padding: 17px 27px;

    border-radius: 15px;

    text-decoration: none !important;

    font-size: 16px;

    font-weight: 600;

    margin-right: 12px;
}


.ts-btn-black {

    background: #171717;

    color: white !important;

    box-shadow: 0 8px 15px rgba(0,0,0,.12);
}


.ts-btn-light {

    background: #faf9f7;

    color: #171717 !important;

    border: 1px solid #ddd8d1;
}


/* Features */

.hero-features {

    border-top: 1px solid #ddd8d1;

    margin-top: 28px;

    padding-top: 25px;

    display: flex;

    gap: 70px;

    color: #6e6963;

    font-size: 14px;
}


.feature {

    display: flex;

    align-items: center;

    white-space: nowrap;
}


.check {

    width: 18px;
    height: 18px;

    border: 1.5px solid #242424;

    border-radius: 50%;

    margin-right: 9px;

    display: inline-flex;

    justify-content: center;

    align-items: center;

    font-size: 11px;
}


/* ============================================================
   RIGHT MOCKUP
   ============================================================ */

.mock {

    background: #fbfaf8;

    border: 1px solid #e1dcd5;

    border-radius: 23px;

    padding: 30px;

    box-shadow: 0 12px 30px rgba(40,35,28,.055);
}


/* Mock header */

.mock-head {

    height: 45px;

    border-bottom: 1px solid #dfdad3;

    display: flex;

    align-items: flex-start;

    justify-content: space-between;
}


.mock-dots {

    display: flex;

    gap: 8px;

    padding-top: 9px;
}


.dot {

    width: 14px;

    height: 14px;

    border-radius: 50%;
}


.dot.r {
    background: #ff6173;
}

.dot.y {
    background: #ffbd2e;
}

.dot.g {
    background: #00c98b;
}


.mock-file {

    font-family: "DM Mono", monospace;

    color: #716b65;

    font-size: 12px;

    padding-top: 9px;
}


.mock-live {

    background: #171717;

    color: white;

    border-radius: 9px;

    font-size: 10px;

    font-weight: 700;

    padding: 7px 11px;
}


/* Documents */

.mock-doc {

    margin-top: 19px;

    border: 1px solid #e0dbd4;

    background: #f1eee9;

    border-radius: 16px;

    min-height: 72px;

    padding: 17px;

    display: flex;

    justify-content: space-between;

    align-items: center;
}


.mock-doc-left {

    display: flex;

    gap: 12px;

    align-items: center;
}


.doc-icon {

    width: 20px;

    height: 25px;

    border: 2px solid #6d6862;

    border-radius: 4px;

    position: relative;
}


.doc-icon:after {

    content: "";

    position: absolute;

    left: 4px;

    right: 4px;

    top: 8px;

    border-top: 1px solid #aaa39b;

    box-shadow: 0 5px 0 #aaa39b;
}


.mock-doc-title {

    font-size: 14px;

    font-weight: 700;
}


.mock-meta {

    color: #8b857e;

    font-size: 11px;

    margin-top: 4px;
}


.mock-label {

    font-family: "DM Mono", monospace;

    font-size: 11px;
}


/* Score */

.mock-score {

    margin-top: 19px;

    background: #171717;

    color: white;

    border-radius: 17px;

    height: 101px;

    display: grid;

    grid-template-columns: 1fr 1fr 1fr;

    align-items: center;

    text-align: center;
}


.score-cell + .score-cell {

    border-left: 1px solid #353535;
}


.score-number {

    font-size: 25px;

    font-weight: 700;

    line-height: 1;
}


.score-label {

    color: #b7b2ac;

    font-size: 10px;

    margin-top: 8px;
}


.risk-badge {

    display: inline-block;

    background: #541c2a;

    color: #ffb1bd;

    border-radius: 8px;

    padding: 6px 10px;

    font-size: 10px;
}


/* Match */

.mock-match {

    margin-top: 20px;

    border: 1px solid #dfdad3;

    border-radius: 15px;

    padding: 16px;
}


.match-top {

    display: flex;

    justify-content: space-between;

    align-items: center;

    font-size: 12px;

    font-weight: 600;
}


.overlap {

    color: #f05c6d;

    border: 1px solid #ffb6c0;

    background: #fff4f5;

    padding: 5px 8px;

    border-radius: 7px;

    font-size: 11px;
}


.mock-quote {

    margin-top: 12px;

    background: #f2eee8;

    border: 1px solid #e3ddd5;

    border-radius: 6px;

    padding: 10px;

    color: #655f59;

    font-family: "DM Mono", monospace;

    font-size: 10px;

    line-height: 1.6;
}


/* ============================================================
   ANALYZER
   ============================================================ */

#analyzer {
    scroll-margin-top: 30px;
}


.analyzer {

    margin-top: 105px;

    padding-top: 55px;

    border-top: 1px solid #ddd8d1;
}


.analyzer h2 {

    font-size: 42px;

    letter-spacing: -2px;

    margin: 0 0 7px;
}


.analyzer-sub {

    color: #716c66;

    margin-bottom: 28px;
}


/* Upload cards */

.upload-card {

    background: #faf9f7;

    border: 1px solid #dfdad3;

    border-radius: 18px;

    padding: 24px;
}


.upload-title {

    font-size: 18px;

    font-weight: 700;

    margin-bottom: 4px;
}


.upload-sub {

    color: #7b756f;

    font-size: 13px;

    margin-bottom: 15px;
}


div[data-testid="stFileUploader"] {

    background: #f4f1ec !important;

    border: 1px dashed #cfc8bf !important;

    border-radius: 13px !important;
}


/* Text area */

textarea {

    background: #faf9f7 !important;
}


/* Buttons */

div.stButton > button {

    border-radius: 14px !important;

    min-height: 50px !important;

    font-weight: 600 !important;
}


/* ============================================================
   HOW IT WORKS
   ============================================================ */

#how-it-works {
    scroll-margin-top: 30px;
}

.how-section {
    margin-top: 110px;
    padding-top: 65px;
    border-top: 1px solid #ddd8d1;
}

.how-title {
    font-size: 42px;
    letter-spacing: -2px;
    margin: 0 0 7px;
}

.how-sub {
    color: #716c66;
    margin-bottom: 34px;
}

.how-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
}

.how-card {
    background: #faf9f7;
    border: 1px solid #ded9d2;
    border-radius: 18px;
    padding: 26px;
    min-height: 220px;
}

.how-number {
    font-family: "DM Mono", monospace;
    font-size: 12px;
    letter-spacing: 1px;
    color: #77716a;
    margin-bottom: 32px;
}

.how-card h3 {
    font-size: 20px;
    margin: 0 0 10px;
}

.how-card p {
    color: #716c66;
    line-height: 1.7;
    margin: 0;
}

@media(max-width: 1000px) {
    .how-grid {
        grid-template-columns: 1fr;
    }
}


/* ============================================================
   RESULTS
   ============================================================ */

.result {

    margin-top: 40px;
}


.result-card {

    background: #faf9f7;

    border: 1px solid #ded9d2;

    border-radius: 18px;

    padding: 23px;
}


.metric {

    font-size: 39px;

    font-weight: 700;

    letter-spacing: -2px;
}


.muted {

    color: #77716a;

    font-size: 12px;

    letter-spacing: .3px;
}


.high {
    color: #bd3345;
}

.medium {
    color: #a66b00;
}

.low {
    color: #28744e;
}


.match-card {

    background: #faf9f7;

    border: 1px solid #ded9d2;

    border-radius: 16px;

    padding: 18px;

    margin: 12px 0;
}


.match-card .quote {

    background: #f1eee9;

    border: 1px solid #e1dcd4;

    border-radius: 8px;

    padding: 12px;

    margin-top: 7px;

    color: #625d57;

    font-family: "DM Mono", monospace;

    font-size: 11px;

    line-height: 1.6;
}


.footer {

    text-align: center;

    color: #89827b;

    font-size: 12px;

    padding-top: 50px;
}


/* ============================================================
   RESPONSIVE
   ============================================================ */

@media(max-width: 1000px) {

    .ts-header {

        grid-template-columns: 1fr auto;
    }

    .ts-nav {
        display: none;
    }

    .hero-wrap {

        grid-template-columns: 1fr;

        gap: 45px;

        padding-top: 55px;
    }

    .hero-title {

        font-size: 52px;

        letter-spacing: -2.7px;
    }

    .hero-features {

        gap: 25px;

        flex-wrap: wrap;
    }
}


@media(max-width: 600px) {

    .hero-title {

        font-size: 42px;
    }

    .hero-title em {

        white-space: normal;
    }

    .hero-copy {

        font-size: 17px;
    }

    .mock {

        padding: 20px;
    }

    .ts-brand-name {

        font-size: 21px;
    }

    .ts-pill {

        display: none;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FILE READING
# ============================================================

def read_file(uploaded_file):

    if uploaded_file is None:
        return ""

    data = uploaded_file.getvalue()

    filename = uploaded_file.name.lower()

    # TXT
    if filename.endswith(".txt"):

        return data.decode(
            "utf-8",
            errors="ignore"
        )


    # PDF
    if filename.endswith(".pdf"):

        if PyPDF2 is None:
            return ""

        reader = PyPDF2.PdfReader(
            io.BytesIO(data)
        )

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)


    # DOCX
    if filename.endswith(".docx"):

        if Document is None:
            return ""

        document = Document(
            io.BytesIO(data)
        )

        paragraphs = []

        for paragraph in document.paragraphs:

            if paragraph.text.strip():

                paragraphs.append(
                    paragraph.text
                )

        return "\n".join(paragraphs)


    return ""


# ============================================================
# TEXT PROCESSING
# ============================================================

def clean_text(text):

    return re.sub(
        r"\s+",
        " ",
        text or ""
    ).strip()


def split_sections(text):

    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sections = []

    # Three sentences per section
    for i in range(
        0,
        len(sentences),
        3
    ):

        section = " ".join(
            sentences[i:i + 3]
        ).strip()

        if section:

            sections.append(
                section
            )

    return sections


# ============================================================
# SEMANTIC ANALYSIS
# ============================================================

def analyze_documents(
    document_a,
    document_b
):

    sections_a = split_sections(
        document_a
    )

    sections_b = split_sections(
        document_b
    )

    if not sections_a or not sections_b:

        return None


    vectorizer = TfidfVectorizer(

        stop_words="english",

        ngram_range=(1, 2),

        max_features=15000
    )


    combined = (
        sections_a +
        sections_b
    )


    matrix = vectorizer.fit_transform(
        combined
    )


    vectors_a = matrix[
        :len(sections_a)
    ]

    vectors_b = matrix[
        len(sections_a):
    ]


    similarity_matrix = cosine_similarity(
        vectors_a,
        vectors_b
    )


    # Overall similarity
    full_a = vectorizer.transform(
        [clean_text(document_a)]
    )

    full_b = vectorizer.transform(
        [clean_text(document_b)]
    )


    overall_similarity = float(

        cosine_similarity(
            full_a,
            full_b
        )[0][0] * 100
    )


    # Matching sections
    match_count = int(
        (
            similarity_matrix >= 0.60
        ).sum()
    )


    # Risk level
    if overall_similarity >= 75:

        risk = "High"

    elif overall_similarity >= 45:

        risk = "Medium"

    else:

        risk = "Low"


    # Find strongest matches
    matches = []

    flattened = np.argsort(
        similarity_matrix.ravel()
    )[::-1]


    used = set()


    for index in flattened:

        i, j = np.unravel_index(
            index,
            similarity_matrix.shape
        )

        score = (
            similarity_matrix[i][j]
            * 100
        )


        if score < 40:

            break


        if (i, j) in used:

            continue


        used.add(
            (i, j)
        )


        matches.append({

            "a_index": i,

            "b_index": j,

            "score": score,

            "a_text": sections_a[i],

            "b_text": sections_b[j]

        })


        if len(matches) >= 8:

            break


    return {

        "overall": overall_similarity,

        "risk": risk,

        "matches": match_count,

        "sections_a": sections_a,

        "sections_b": sections_b,

        "top_matches": matches,

        "matrix": similarity_matrix

    }


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="ts-header">

    <div class="ts-brand">

        <div class="ts-shield">
            ♢
        </div>

        <div class="ts-brand-name">
            TextShield
        </div>

        <div class="ts-pill">
            AI SEMANTIC ENGINE
        </div>

    </div>


    <div class="ts-nav">

        <div class="selected">
            Product
        </div>

        <div>
            How It Works
        </div>

        <div>
            Features
        </div>

        <div>
            Dashboard Overview
        </div>

    </div>


    <a
        class="ts-get"
        href="#analyzer"
    >
        Get Started
        <span>↗</span>
    </a>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
<div class="hero-wrap">

    <div>

        <div class="eyebrow">
            ✣ &nbsp; AI DOCUMENT ANALYSIS
        </div>


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


        <a
            class="ts-btn ts-btn-black"
            href="#analyzer"
        >
            Analyze Documents &nbsp; →
        </a>


        <a
            class="ts-btn ts-btn-light"
            href="#how-it-works"
        >
            See How It Works
        </a>


        <div class="hero-features">

            <div class="feature">

                <span class="check">
                    ✓
                </span>

                Vector Embeddings

            </div>


            <div class="feature">

                <span class="check">
                    ✓
                </span>

                Section Matching

            </div>


            <div class="feature">

                <span class="check">
                    ✓
                </span>

                PDF / DOCX / TXT

            </div>

        </div>

    </div>


    <!-- RIGHT MOCKUP -->

    <div class="mock">

        <div class="mock-head">

            <div class="mock-dots">

                <span class="dot r"></span>

                <span class="dot y"></span>

                <span class="dot g"></span>

            </div>


            <div class="mock-file">
                analysis_report_v2.json
            </div>


            <div class="mock-live">
                LIVE MOCKUP
            </div>

        </div>


        <!-- DOCUMENT A -->

        <div class="mock-doc">

            <div class="mock-doc-left">

                <div class="doc-icon"></div>

                <div>

                    <div class="mock-doc-title">
                        Document_A_Research.docx
                    </div>

                    <div class="mock-meta">
                        2,450 words • Reference Document
                    </div>

                </div>

            </div>


            <div class="mock-label">
                Doc A
            </div>

        </div>


        <!-- DOCUMENT B -->

        <div class="mock-doc">

            <div class="mock-doc-left">

                <div class="doc-icon"></div>

                <div>

                    <div class="mock-doc-title">
                        Student_Submission_B.pdf
                    </div>

                    <div class="mock-meta">
                        2,180 words • Target Comparison
                    </div>

                </div>

            </div>


            <div class="mock-label">
                Doc B
            </div>

        </div>


        <!-- SCORE -->

        <div class="mock-score">

            <div class="score-cell">

                <div class="score-number">
                    82%
                </div>

                <div class="score-label">
                    SIMILARITY
                </div>

            </div>


            <div class="score-cell">

                <div class="risk-badge">
                    ♙ &nbsp;High
                </div>

                <div class="score-label">
                    RISK LEVEL
                </div>

            </div>


            <div class="score-cell">

                <div class="score-number">
                    12
                </div>

                <div class="score-label">
                    MATCHES
                </div>

            </div>

        </div>


        <!-- MATCH -->

        <div class="mock-match">

            <div class="match-top">

                <span>
                    Section #3 Semantic Match
                </span>

                <span class="overlap">
                    86.2% Overlap
                </span>

            </div>


            <div class="mock-quote">

                "Deep learning algorithms enable
                automated diagnostic scans to detect
                early stage anomalous structures..."

            </div>

        </div>

    </div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# ANALYZER SECTION
# ============================================================

st.markdown(
    """
<div
    class="analyzer"
    id="analyzer"
>

    <h2>
        Analyze your documents
    </h2>

    <div class="analyzer-sub">

        Upload two documents and compare
        their semantic similarity.

    </div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

column_a, column_b = st.columns(
    2,
    gap="large"
)


# DOCUMENT A

with column_a:

    st.markdown(
        """
        <div class="upload-card">

            <div class="upload-title">
                Document A
            </div>

            <div class="upload-sub">
                Reference document
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    file_a = st.file_uploader(

        "Upload Document A",

        type=[
            "pdf",
            "docx",
            "txt"
        ],

        key="document_a"
    )


    text_a = st.text_area(

        "Paste Document A",

        height=150,

        placeholder=
        "Paste your reference text here..."

    )


# DOCUMENT B

with column_b:

    st.markdown(
        """
        <div class="upload-card">

            <div class="upload-title">
                Document B
            </div>

            <div class="upload-sub">
                Target comparison
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    file_b = st.file_uploader(

        "Upload Document B",

        type=[
            "pdf",
            "docx",
            "txt"
        ],

        key="document_b"
    )


    text_b = st.text_area(

        "Paste Document B",

        height=150,

        placeholder=
        "Paste your target text here..."

    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(

    "Run Semantic Analysis  →",

    type="primary",

    use_container_width=True

):

    document_a = clean_text(

        read_file(file_a)
        if file_a
        else text_a

    )


    document_b = clean_text(

        read_file(file_b)
        if file_b
        else text_b

    )


    if not document_a:

        st.error(
            "Please provide Document A."
        )


    elif not document_b:

        st.error(
            "Please provide Document B."
        )


    else:

        with st.spinner(
            "Analyzing semantic similarity..."
        ):

            result = analyze_documents(

                document_a,

                document_b

            )


        if result is None:

            st.error(
                "Not enough readable text was found."
            )

        else:

            st.session_state[
                "analysis_result"
            ] = result


# ============================================================
# RESULTS
# ============================================================

if "analysis_result" in st.session_state:

    result = st.session_state[
        "analysis_result"
    ]


    overall = result["overall"]

    risk = result["risk"]

    matches = result["matches"]

    top_matches = result[
        "top_matches"
    ]

    similarity_matrix = result[
        "matrix"
    ]


    st.markdown(
        """
        <div class="result">

            <h2>
                Dashboard Overview
            </h2>

        </div>
        """,
        unsafe_allow_html=True
    )


    # Metrics

    metric1, metric2, metric3 = st.columns(3)


    with metric1:

        st.markdown(
            f"""
            <div class="result-card">

                <div class="muted">
                    SIMILARITY
                </div>

                <div class="metric">
                    {overall:.1f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with metric2:

        st.markdown(
            f"""
            <div class="result-card">

                <div class="muted">
                    RISK LEVEL
                </div>

                <div class="metric {risk.lower()}">
                    {risk}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with metric3:

        st.markdown(
            f"""
            <div class="result-card">

                <div class="muted">
                    MATCHING SECTIONS
                </div>

                <div class="metric">
                    {matches}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # MATCHES
    # ========================================================

    st.markdown(
        "### Semantic Matches"
    )


    for number, match in enumerate(
        top_matches,
        start=1
    ):

        score = match["score"]

        text_from_a = html.escape(
            match["a_text"]
        )

        text_from_b = html.escape(
            match["b_text"]
        )


        st.markdown(
            f"""
            <div class="match-card">

                <div
                    style="
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                    "
                >

                    <strong>
                        Section #{number}
                        Semantic Match
                    </strong>

                    <span class="overlap">
                        {score:.1f}% Overlap
                    </span>

                </div>


                <div
                    class="muted"
                    style="margin-top:15px;"
                >
                    DOCUMENT A
                </div>


                <div class="quote">

                    {text_from_a[:1000]}

                </div>


                <div
                    class="muted"
                    style="margin-top:14px;"
                >
                    DOCUMENT B
                </div>


                <div class="quote">

                    {text_from_b[:1000]}

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # SIMILARITY MATRIX
    # ========================================================

    st.markdown(
        "### Similarity Matrix"
    )


    st.dataframe(

        np.round(
            similarity_matrix * 100,
            1
        ),

        use_container_width=True,

        hide_index=True

    )


    st.caption(

        "This version uses TF-IDF + cosine similarity "
        "as a lightweight local baseline. For stronger "
        "semantic matching, the engine can be upgraded "
        "to sentence embeddings."

    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    """
<section class="how-section" id="how-it-works">

    <h2 class="how-title">
        How It Works
    </h2>

    <div class="how-sub">
        TextShield analyzes your documents in three simple stages.
    </div>

    <div class="how-grid">

        <div class="how-card">

            <div class="how-number">
                01
            </div>

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

            <div class="how-number">
                02
            </div>

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

            <div class="how-number">
                03
            </div>

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

</section>
""",
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        TextShield • AI Semantic Document Analysis

    </div>
    """,
    unsafe_allow_html=True
)
