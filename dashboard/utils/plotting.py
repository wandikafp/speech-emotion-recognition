import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import librosa
import librosa.display

EMOTION_COLORS = {
    "neutral": "#95a5a6", "calm": "#3498db", "happy": "#f1c40f",
    "sad": "#2980b9", "angry": "#e74c3c", "fearful": "#9b59b6",
    "disgust": "#e67e22", "surprised": "#1abc9c",
}

AGENT_COLORS = {
    "de_escalation": "#e74c3c", "empathetic": "#9b59b6",
    "quality": "#e67e22", "standard": "#27ae60",
    "general": "#3498db", "manual_review": "#95a5a6",
}

def plot_emotion_scores(scores: dict, title="Emotion Probabilities") -> go.Figure:
    emotions = list(scores.keys())
    values   = [scores[e] * 100 for e in emotions]
    colors   = [EMOTION_COLORS.get(e, "#666") for e in emotions]
    fig = go.Figure(go.Bar(
        x=emotions, y=values,
        marker_color=colors, text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title, yaxis_title="Probability (%)", yaxis_range=[0, 110],
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#a8b2d8"), height=350,
    )
    return fig

def plot_waveform(audio_array, sr) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    librosa.display.waveshow(audio_array, sr=sr, ax=ax, color="#e94560")
    ax.set_title("Waveform", color="#a8b2d8"); ax.tick_params(colors="#a8b2d8")
    for spine in ax.spines.values(): spine.set_color("#333")
    plt.tight_layout()
    return fig

def plot_spectrogram(audio_array, sr) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_array)), ref=np.max)
    img = librosa.display.specshow(D, y_axis="log", x_axis="time", sr=sr, ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Spectrogram", color="#a8b2d8"); ax.tick_params(colors="#a8b2d8")
    for spine in ax.spines.values(): spine.set_color("#333")
    plt.tight_layout()
    return fig
