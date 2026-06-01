import streamlit as st
import tempfile
import os
import time
import librosa
from utils.helpers import predict_from_bytes
from utils.plotting import plot_emotion_scores, plot_waveform, plot_spectrogram

st.header("🎤 Live Emotion Detection")
st.caption("Pilih metode input audio untuk mendeteksi emosi secara langsung.")

predictor = st.session_state.get("predictor")

def process_audio(audio_bytes, ext="wav"):
    st.audio(audio_bytes, format=f"audio/{ext}")

    with st.spinner("Menganalisis emosi..."):
        t0 = time.perf_counter()
        result = predict_from_bytes(predictor, audio_bytes, ext)
        latency = (time.perf_counter() - t0) * 1000

    # ── Result cards ─────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    dom = result["dominant_emotion"]
    conf = result["confidence"]
    with col1:
        st.markdown(f"""<div class="emotion-card">
            <div class="label">Dominant Emotion</div>
            <div class="value">{dom.upper()}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="emotion-card">
            <div class="label">Confidence</div>
            <div class="value">{conf*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="emotion-card">
            <div class="label">Latency</div>
            <div class="value">{latency:.0f}ms</div>
        </div>""", unsafe_allow_html=True)

    st.progress(conf, text=f"Confidence: {conf*100:.1f}%")

    # ── Charts ───────────────────────────────────────────────────────
    st.plotly_chart(plot_emotion_scores(result["emotion_scores"]),
                    use_container_width=True)

    with st.expander("🔬 Audio Visualisation"):
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            audio_arr, sr = librosa.load(tmp_path, sr=22050)
            col_w, col_s = st.columns(2)
            with col_w:
                st.pyplot(plot_waveform(audio_arr, sr))
            with col_s:
                st.pyplot(plot_spectrogram(audio_arr, sr))
        except Exception as e:
            st.error(f"Gagal memvisualisasikan audio: {e}")
        finally:
            os.unlink(tmp_path)

if not predictor:
    st.warning("⚠️ Load model terlebih dahulu di sidebar.")
else:
    tab_upload, tab_record = st.tabs(["📂 Upload File", "🎙️ Rekam Audio"])
    
    with tab_upload:
        uploaded = st.file_uploader("Upload file audio (.wav)", type=["wav", "mp3", "ogg"])
        if uploaded:
            ext = uploaded.name.split('.')[-1]
            process_audio(uploaded.read(), ext)
            
    with tab_record:
        audio_value = st.audio_input("Rekam suara Anda secara langsung")
        if audio_value:
            process_audio(audio_value.read(), "wav")
