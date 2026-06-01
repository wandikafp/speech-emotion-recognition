import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import predict_from_bytes
from utils.plotting import EMOTION_COLORS
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
from api.psych_report import generate_psych_report, format_report_text

st.header("🧠 Psychologist Pre-Assessment Report")
st.caption("Analisis emosi rekaman audio klien untuk gambaran awal kondisi emosional.")

predictor = st.session_state.get("predictor")

if not predictor:
    st.warning("⚠️ Load model terlebih dahulu di sidebar.")
else:
    col_form, col_hint = st.columns([2, 1])
    with col_form:
        client_id  = st.text_input("Client ID",  placeholder="C001")
        session_id = st.text_input("Session ID", placeholder="S2026-05-19")
        psych_files = st.file_uploader(
            "Upload rekaman sesi (.wav) — bisa multiple segmen",
            type=["wav"], accept_multiple_files=True, key="psych_upload"
        )
    with col_hint:
        st.info("💡 **Tips:**\nUpload rekaman dalam beberapa segmen (misal: setiap 30 detik) untuk analisis timeline yang lebih akurat.")

    if psych_files and st.button("🔍 Generate Report", type="primary"):
        preds = []
        prog2 = st.progress(0)
        for i, f in enumerate(psych_files):
            pred = predict_from_bytes(predictor, f.read())
            preds.append(pred)
            prog2.progress((i+1)/len(psych_files))

        report = generate_psych_report(preds, client_id=client_id or None,
                                        session_id=session_id or None)

        # ── Risk badge ───────────────────────────────────────────────────
        risk = report["risk_assessment"]["overall_risk_level"]
        risk_cls = f"risk-{risk.lower()}"
        dom_e = report["dominant_emotion"]
        st.markdown(f"""
        <div style="margin:1rem 0">
            <span class="agent-badge {risk_cls}">⚠️ Risk: {risk}</span>
            &nbsp;&nbsp;
            <b>Dominant:</b> {dom_e['emotion'].upper()} ({dom_e['percent']})
        </div>""", unsafe_allow_html=True)

        # ── Arousal-Valence ───────────────────────────────────────────────
        av = report["arousal_valence"]
        col_av1, col_av2, col_av3 = st.columns(3)
        col_av1.metric("Arousal",  av["dominant_arousal"])
        col_av2.metric("Valence",  av["dominant_valence"])
        col_av3.metric("Variabilitas", report["emotional_variability"]["level"])
        st.caption(av["interpretation"])

        # ── Emotion distribution chart ────────────────────────────────────
        ed = report["emotion_distribution"]
        fig_ed = go.Figure(go.Bar(
            x=list(ed.keys()),
            y=[v["percent"].replace("%","") for v in ed.values()],
            marker_color=[EMOTION_COLORS.get(e,"#666") for e in ed.keys()],
            text=[v["percent"] for v in ed.values()], textposition="outside",
        ))
        fig_ed.update_layout(
            title="Distribusi Emosi Klien", yaxis_title="Persentase (%)",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            font=dict(color="#a8b2d8"), height=320
        )
        st.plotly_chart(fig_ed, use_container_width=True)

        # ── Timeline ─────────────────────────────────────────────────────
        timeline = report["timeline"]
        if len(timeline) > 1:
            tl_df = pd.DataFrame(timeline)
            fig_tl = px.scatter(tl_df, x="segment", y="emotion", color="emotion",
                                size="confidence", color_discrete_map=EMOTION_COLORS,
                                title="Emotion Timeline per Segmen")
            fig_tl.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#a8b2d8"), height=300
            )
            st.plotly_chart(fig_tl, use_container_width=True)

        # ── Risk flags ───────────────────────────────────────────────────
        if report["risk_assessment"]["risk_flags"]:
            st.markdown("### ⚠️ Risk Flags")
            for flag in report["risk_assessment"]["risk_flags"]:
                color = {"HIGH":"#c0392b","MODERATE":"#e67e22","LOW":"#27ae60"}.get(flag["level"],"#666")
                st.markdown(f"""<div style="background:{color}22; border-left:4px solid {color};
                    padding:0.7rem 1rem; border-radius:6px; margin:0.3rem 0">
                    <b style="color:{color}">[{flag['level']}]</b>
                    {flag['flag'].replace('_',' ').title()} —
                    <i>{', '.join(flag['emotions_involved'])}</i> ({flag['ratio']*100:.0f}%)</div>""",
                    unsafe_allow_html=True)

        # ── Clinical notes & recommendations ─────────────────────────────
        with st.expander("📋 Catatan Klinis & Rekomendasi"):
            for note in report["risk_assessment"].get("clinical_notes", []):
                st.warning(note)
            st.markdown("**Rekomendasi:**")
            for i, rec in enumerate(report["recommendations"], 1):
                st.markdown(f"{i}. {rec}")

        # ── Download report ───────────────────────────────────────────────
        report_txt = format_report_text(report)
        st.download_button(
            "⬇️ Download Report (TXT)", report_txt,
            file_name=f"psych_report_{client_id or 'anon'}.txt",
            mime="text/plain"
        )
