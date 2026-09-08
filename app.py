import streamlit as st
import joblib
import re
import os


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(PROJECT_DIR, "fake_review_model.pkl"))
tfidf = joblib.load(os.path.join(PROJECT_DIR, "tfidf_vectorizer.pkl"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


st.set_page_config(
    page_title="Fake Review Detector | AI Authenticity Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown(
    "<style>.stApp{background:linear-gradient(135deg,#0f172a,#1e1b4b,#0f172a);color:#e2e8f0;font-family:'Segoe UI',sans-serif;}"
    "#MainMenu,footer,header{visibility:hidden;}"
    ".block-container{max-width:1100px;padding-top:2.5rem;}"
    ".hero-title{text-align:center;font-size:54px;font-weight:800;background:linear-gradient(90deg,#22d3ee,#a855f7,#f472b6);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;}"
    ".hero-subtitle{text-align:center;color:#94a3b8;font-size:19px;}"
    ".stTextArea textarea{background:rgba(255,255,255,0.06)!important;border:1.5px solid rgba(148,163,184,0.3)!important;"
    "border-radius:14px!important;color:#f1f5f9!important;font-size:16px!important;}"
    ".stButton>button{width:100%;height:54px;border-radius:14px;font-size:18px;font-weight:700;color:#fff!important;border:none!important;"
    "background:linear-gradient(90deg,#06b6d4,#8b5cf6)!important;box-shadow:0 8px 24px rgba(139,92,246,0.4);}"
    ".stButton>button:hover{transform:translateY(-2px);box-shadow:0 12px 32px rgba(139,92,246,0.6);}"
    ".section-header{font-size:22px;font-weight:700;color:#f1f5f9;margin:8px 0 4px 0;}"
    ".result-banner{border-radius:16px;padding:22px;text-align:center;font-size:30px;font-weight:800;margin:6px 0 14px 0;}"
    ".result-fake{background:linear-gradient(135deg,rgba(239,68,68,0.2),rgba(127,29,29,0.3));border:1.5px solid rgba(239,68,68,0.6);color:#fca5a5;}"
    ".result-genuine{background:linear-gradient(135deg,rgba(16,185,129,0.2),rgba(6,78,59,0.3));border:1.5px solid rgba(16,185,129,0.6);color:#6ee7b7;}"
    ".metric-card{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:16px;padding:18px;text-align:center;}"
    ".metric-label{color:#94a3b8;font-size:13px;font-weight:600;text-transform:uppercase;}"
    ".metric-value{font-size:34px;font-weight:800;color:#f8fafc;}"
    ".info-card{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:16px;padding:18px;text-align:center;}"
    ".info-icon{font-size:34px;}.info-title{color:#f1f5f9;font-size:17px;font-weight:700;}.info-desc{color:#94a3b8;font-size:13px;}"
    ".footer-note{text-align:center;color:#475569;font-size:13px;margin-top:20px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.08);}"
    "</style>",
    unsafe_allow_html=True,
)


st.markdown('<div class="hero-title">🛡️ Fake Review Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">AI-Powered Review Authenticity · NLP + TF-IDF + Logistic Regression</div>', unsafe_allow_html=True)
st.divider()

st.markdown('<div class="section-header">📝 Enter Product Review</div>', unsafe_allow_html=True)
review = st.text_area(
    "Your review",
    placeholder="Paste or type a product review here to analyze its authenticity...",
    height=170,
    label_visibility="collapsed",
)

col_btn = st.columns([1, 2, 1])[1]
with col_btn:
    analyze_clicked = st.button("🔍 Analyze Review", use_container_width=True)

if analyze_clicked:
    if not review.strip():
        st.warning("⚠️ Please enter a review before analyzing.")
    else:
        cleaned = clean_text(review)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]
        confidence = max(probabilities) * 100

        st.markdown('<div class="section-header">📊 Analysis Result</div>', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown('<div class="result-banner result-fake">🚨 FAKE REVIEW DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-banner result-genuine">✅ GENUINE REVIEW</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">{confidence:.2f}%</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Verdict</div><div class="metric-value" style="font-size:24px;color:{"#f87171" if prediction == 1 else "#34d399"};">{"FAKE" if prediction == 1 else "GENUINE"}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Words Analyzed</div><div class="metric-value" style="font-size:24px;">{len(cleaned.split())}</div></div>', unsafe_allow_html=True)

        st.progress(int(confidence))

st.divider()
st.markdown('<div class="section-header">🧠 Model Information</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="info-card"><div class="info-icon">🧬</div><div class="info-title">NLP</div><div class="info-desc">Text preprocessing &amp; feature extraction</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="info-card"><div class="info-icon">📊</div><div class="info-title">TF-IDF</div><div class="info-desc">Term frequency-inverse document frequency</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="info-card"><div class="info-icon">📈</div><div class="info-title">Logistic Regression</div><div class="info-desc">Classification model for authenticity</div></div>', unsafe_allow_html=True)

st.markdown('<div class="footer-note">🛡️ Built with Streamlit · Apply machine learning to identify potentially deceptive reviews</div>', unsafe_allow_html=True)
