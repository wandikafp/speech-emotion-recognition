import streamlit as st
import pandas as pd
import plotly.express as px
from utils.helpers import predict_from_bytes
from utils.plotting import AGENT_COLORS, EMOTION_COLORS
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
from api.router import route_contact_center, get_dominant_routing

st.header("📞 Contact Center Agent Routing")
st.caption("Simulasi routing panggilan berdasarkan deteksi emosi pelanggan.")

predictor = st.session_state.get("predictor")

if not predictor:
    st.warning("⚠️ Load model terlebih dahulu di sidebar.")
else:
    st.subheader("Upload Audio Panggilan")
    col_up, col_info = st.columns([2, 1])

    with col_up:
        call_files = st.file_uploader(
            "Upload 1 atau lebih segmen audio panggilan (.wav)",
            type=["wav"], accept_multiple_files=True,
            key="cc_upload"
        )

    with col_info:
        st.info("💡 **Cara kerja:**\n1. Upload rekaman panggilan\n2. Model deteksi emosi tiap segmen\n3. Business logic tentukan tipe agent\n4. Sistem routing ke agent terbaik")

    if call_files:
        segment_results, routed_results = [], []

        prog = st.progress(0, "Menganalisis segmen...")
        for i, f in enumerate(call_files):
            audio_bytes = f.read()
            pred   = predict_from_bytes(predictor, audio_bytes)
            routed = route_contact_center(pred)
            segment_results.append({**pred, "filename": f.name})
            routed_results.append(routed)
            prog.progress((i+1)/len(call_files), f"Segmen {i+1}/{len(call_files)}")

        dominant = get_dominant_routing(routed_results)

        # ── Routing result ────────────────────────────────────────────────
        st.markdown("---")
        agent = dominant["agent_type"]
        color = AGENT_COLORS.get(agent, "#666")
        st.markdown(f"""
        <div style="background:{color}22; border:2px solid {color};
                    border-radius:12px; padding:1.5rem; text-align:center;">
            <h2 style="color:{color}; margin:0">🎯 {dominant['agent_description']}</h2>
            <p style="color:#a8b2d8; margin:0.5rem 0 0">
            Priority {dominant['routing_priority']} |
            Escalation: {'✅ Required' if dominant['requires_escalation'] else '❌ Not required'}
            </p>
            <p style="color:#ccc; font-size:0.9rem">{dominant['routing_reason']}</p>
        </div>""", unsafe_allow_html=True)

        # ── Session summary ───────────────────────────────────────────────
        st.markdown("### 📊 Session Emotion Timeline")
        summary = dominant.get("session_summary", {})
        ec = summary.get("emotion_counts", {})
        if ec:
            fig_pie = px.pie(
                names=list(ec.keys()), values=list(ec.values()),
                color=list(ec.keys()),
                color_discrete_map=EMOTION_COLORS,
                title="Distribusi Emosi Sepanjang Panggilan"
            )
            fig_pie.update_layout(
                paper_bgcolor="#0d1117", font=dict(color="#a8b2d8"), height=350
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Per-segment table ─────────────────────────────────────────────
        with st.expander("📋 Detail per Segmen"):
            rows = []
            for i, (seg, rt) in enumerate(zip(segment_results, routed_results)):
                rows.append({
                    "Segmen": i + 1,
                    "File": seg["filename"],
                    "Emosi": seg["dominant_emotion"],
                    "Confidence": f"{seg['confidence']*100:.1f}%",
                    "Agent Type": rt["agent_type"],
                    "Priority": rt["routing_priority"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
