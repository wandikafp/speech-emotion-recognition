"""
Phase 7 — Streamlit Dashboard for Speech Emotion Recognition.
Modularised App Entry Point.

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
import glob
import streamlit as st

st.set_page_config(
    page_title="SER System — Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# ── Custom CSS ────────────────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "dashboard", "style.css")) as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎙️ Speech Emotion Recognition System</h1>
    <p>Contact Center Agent Routing & Psychologist Pre-Assessment</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### Welcome to the SER Dashboard!
Please navigate to the different features using the sidebar on the left.
- **🎤 Live Demo**: Try emotion detection with real-time audio uploads.
- **📞 Contact Center**: Simulate call routing based on emotional analysis.
- **🧠 Psych Assessment**: Generate psychological pre-assessment reports.
- **📊 Model Analytics**: Evaluate model performance and view training history.
""")

# ── Sidebar: Model loader ─────────────────────────────────────────────────────
st.sidebar.title("⚙️ Model Configuration")

MODELS_DIR = os.path.join(BASE_DIR, "models")
model_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.keras")), key=os.path.getmtime, reverse=True)
model_labels = [os.path.basename(f) for f in model_files]

selected_model = st.sidebar.selectbox(
    "Pilih Model", model_labels if model_labels else ["(no model found)"],
    help="Model .keras yang sudah ditraining"
)

@st.cache_resource
def load_predictor(model_name: str):
    from src.predict import SERPredictor
    path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(path):
        return SERPredictor(path)
    return None

predictor = None
if model_files and selected_model != "(no model found)":
    with st.sidebar:
        with st.spinner("Loading model..."):
            predictor = load_predictor(selected_model)
        if predictor:
            st.success(f"✅ Model loaded")
            st.caption(f"Classes: {', '.join(predictor.classes)}")
            st.session_state["predictor"] = predictor
        else:
            st.error("❌ Failed to load model")
else:
    st.sidebar.warning("No trained model found. Run notebook 03 first.")

st.sidebar.markdown("---")
st.sidebar.markdown("**📖 Navigation**")
st.sidebar.caption("Pilih halaman di atas untuk menggunakan fitur yang diinginkan.")
