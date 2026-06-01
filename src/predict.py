"""
Inference module — predicts emotion from a single audio file or numpy array.
Used by both the API and the dashboard.
"""

import os
import sys
import numpy as np
import joblib
import librosa
import soundfile as sf
import tempfile
import base64
from typing import Union

import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.extractor import preprocess_audio, extract_features
from src.model import SoftAttention

# ── Default paths ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(BASE_DIR, "models")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed", "features_random")
SCALER_PATH  = os.path.join(FEATURES_DIR, "train", "scaler.pkl")
ENCODER_PATH = os.path.join(FEATURES_DIR, "train", "label_encoder.pkl")


class SERPredictor:
    """
    Speech Emotion Recognition inference engine.
    Loads model + scaler + label encoder once, then serves predictions.
    """

    def __init__(self, model_path: str,
                 scaler_path: str = SCALER_PATH,
                 encoder_path: str = ENCODER_PATH):
        print(f"Loading model from: {model_path}")
        self.model   = keras.models.load_model(model_path, custom_objects={'SoftAttention': SoftAttention})
        self.scaler  = joblib.load(scaler_path)
        encoder      = joblib.load(encoder_path)
        self.label2idx = encoder["label2idx"]
        self.idx2label = encoder.get("idx2label", {v: k for k, v in self.label2idx.items()})
        self.classes   = [self.idx2label[i] for i in range(len(self.idx2label))]
        print(f"Model loaded. Classes: {self.classes}")

    # ── Core prediction ───────────────────────────────────────────────────────

    def _audio_to_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Preprocess raw audio and extract normalised features."""
        # Re-use pipeline from extractor (resample / trim / pad)
        import io
        from src.features.extractor import (
            preprocess_audio, extract_features,
            TARGET_SR, N_SAMPLES
        )

        # Preprocess starting from raw waveform (already loaded)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        audio = librosa.effects.preemphasis(audio, coef=0.97)
        audio, _ = librosa.effects.trim(audio, top_db=30)
        if len(audio) > N_SAMPLES:
            audio = audio[:N_SAMPLES]
        else:
            audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode="constant")

        # Feature extraction
        feat = extract_features(audio, sr=TARGET_SR)        # (T, F)

        # Normalise with training scaler
        T, F   = feat.shape
        normed = self.scaler.transform(feat).astype(np.float32)

        # Add batch dimension → (1, T, F)
        return normed[np.newaxis, ...]

    def predict_from_array(self, audio: np.ndarray, sr: int) -> dict:
        """Predict emotion from a numpy audio array."""
        X    = self._audio_to_features(audio, sr)
        prob = self.model.predict(X, verbose=0)[0]          # (n_classes,)
        return self._format_result(prob)

    def predict_from_file(self, file_path: str) -> dict:
        """Predict emotion from an audio file path."""
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        return self.predict_from_array(audio, sr)

    def predict_from_base64(self, audio_b64: str,
                             audio_format: str = "wav") -> dict:
        """Predict emotion from a base64-encoded audio string."""
        raw_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            result = self.predict_from_file(tmp_path)
        finally:
            os.unlink(tmp_path)
        return result

    # ── Result formatting ─────────────────────────────────────────────────────

    def _format_result(self, prob: np.ndarray) -> dict:
        dominant_idx      = int(np.argmax(prob))
        emotion_scores    = {cls: float(prob[i]) for i, cls in enumerate(self.classes)}
        return {
            "dominant_emotion": self.idx2label[dominant_idx],
            "confidence":       float(prob[dominant_idx]),
            "emotion_scores":   emotion_scores,
        }


# ── CLI quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, glob, time

    parser = argparse.ArgumentParser(description="Predict emotion from audio")
    parser.add_argument("--model",  required=True, help="Path to .keras model")
    parser.add_argument("--audio",  required=True, help="Path to .wav file")
    args = parser.parse_args()

    predictor = SERPredictor(args.model)
    t0 = time.time()
    result = predictor.predict_from_file(args.audio)
    latency = (time.time() - t0) * 1000

    print(f"\nPrediction result:")
    print(f"  Dominant emotion : {result['dominant_emotion']}")
    print(f"  Confidence       : {result['confidence']:.4f}")
    print(f"  Latency          : {latency:.1f} ms")
    print(f"\nAll scores:")
    for emotion, score in sorted(result['emotion_scores'].items(),
                                 key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 30)
        print(f"  {emotion:10s}: {score:.4f} {bar}")
