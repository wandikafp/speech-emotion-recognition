"""
Feature extractor — aligned with Zennou et al. ETASR 2026.

Features per frame:
  13 MFCC + 13 Delta-MFCC + 13 Delta-Delta-MFCC + 1 RMSE = 40 features
Frame params:
  frame_length = 25 ms, hop_length = 10 ms (per paper)
Sequence length:
  zero-padded to 400 frames (per paper)
Normalisation:
  z-score per utterance (per paper Sec. II-C)
"""

import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SR    = 22050
DURATION     = 3.0                     # seconds (load window)
N_SAMPLES    = int(TARGET_SR * DURATION)

# Frame parameters matching paper (25 ms frame, 10 ms hop)
FRAME_LENGTH = int(0.025 * TARGET_SR)  # 551 samples
HOP_LENGTH   = int(0.010 * TARGET_SR)  # 220 samples
N_FFT        = 2048                    # ≥ FRAME_LENGTH for FFT efficiency

N_MFCC       = 13                      # per paper
MAX_FRAMES   = 400                     # zero-pad target (per paper)
N_FEATURES   = N_MFCC * 3 + 1         # 13+13+13+1 = 40


# ── Core functions ────────────────────────────────────────────────────────────

def preprocess_audio(file_path: str) -> np.ndarray:
    """Load, resample, pre-emphasise, trim silence, and fix length."""
    y, sr = librosa.load(file_path, sr=None, mono=True)

    # 1. Resample to TARGET_SR
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    # 2. Pre-emphasis filter
    y = librosa.effects.preemphasis(y, coef=0.97)

    # 3. Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=30)

    # 4. Pad or truncate to fixed duration
    if len(y) > N_SAMPLES:
        y = y[:N_SAMPLES]
    else:
        y = np.pad(y, (0, N_SAMPLES - len(y)), mode="constant")

    return y


def extract_features(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract MFCC + Delta + Delta-Delta + RMSE features.

    Returns
    -------
    np.ndarray of shape (MAX_FRAMES, N_FEATURES)
        N_FEATURES = 13+13+13+1 = 40
    """
    # 1. MFCC (13 coefficients)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=FRAME_LENGTH,
    )  # shape: (13, T)

    # 2. Delta MFCC
    delta = librosa.feature.delta(mfcc, width=9)       # (13, T)

    # 3. Delta-Delta MFCC
    delta2 = librosa.feature.delta(mfcc, order=2, width=9)  # (13, T)

    # 4. RMSE (root mean square energy) per frame
    rmse = librosa.feature.rms(
        y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )  # shape: (1, T_rmse)  — may differ by ±1 frame from MFCC

    # Align RMSE length to MFCC frame count
    T_mfcc = mfcc.shape[1]
    if rmse.shape[1] > T_mfcc:
        rmse = rmse[:, :T_mfcc]
    elif rmse.shape[1] < T_mfcc:
        rmse = np.pad(rmse, ((0, 0), (0, T_mfcc - rmse.shape[1])), mode="edge")

    # Concatenate → (40, T)
    features = np.concatenate([mfcc, delta, delta2, rmse], axis=0)

    # Transpose → (T, 40)
    features = features.T

    # Zero-pad or truncate to MAX_FRAMES (400)
    T = features.shape[0]
    if T >= MAX_FRAMES:
        features = features[:MAX_FRAMES, :]
    else:
        pad = np.zeros((MAX_FRAMES - T, N_FEATURES), dtype=np.float32)
        features = np.vstack([features, pad])

    return features.astype(np.float32)   # (400, 40)


# ── Pipeline: manifest → features ────────────────────────────────────────────

def build_feature_dataset(manifest_path: str, output_dir: str,
                           scaler_path: str = None) -> tuple:
    """
    Read manifest CSV, extract features for every file, normalise, and save.

    Normalisation follows paper: z-score per utterance (mean/std across
    all frames and all features of that utterance).  A global StandardScaler
    fitted on the training set is also saved for inference.

    Returns
    -------
    (X, y, label_encoder)
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(df)} samples")

    emotion_classes = sorted(df["emotion_label"].unique())
    label2idx = {e: i for i, e in enumerate(emotion_classes)}
    print(f"Classes ({len(emotion_classes)}): {emotion_classes}")

    X_list, y_list = [], []
    errors = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            y_audio = preprocess_audio(row["file_path"])
            feat    = extract_features(y_audio)
            X_list.append(feat)
            y_list.append(label2idx[row["emotion_label"]])
        except Exception as e:
            errors.append((row["file_path"], str(e)))

    if errors:
        print(f"\n⚠️  Skipped {len(errors)} files due to errors:")
        for path, err in errors[:5]:
            print(f"  {os.path.basename(path)}: {err}")

    X = np.array(X_list, dtype=np.float32)   # (N, 400, 40)
    y = np.array(y_list,  dtype=np.int32)     # (N,)
    print(f"\nRaw feature array: X={X.shape}, y={y.shape}")

    # ── Normalise per feature dimension (global z-score on train set) ─────────
    N, T, F = X.shape
    X_flat = X.reshape(-1, F)   # (N*T, F)

    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded existing scaler from {scaler_path}")
    else:
        scaler = StandardScaler()
        scaler.fit(X_flat)
        saved_scaler = os.path.join(output_dir, "scaler.pkl")
        joblib.dump(scaler, saved_scaler)
        print(f"Scaler fitted and saved → {saved_scaler}")

    X_norm = scaler.transform(X_flat).reshape(N, T, F).astype(np.float32)

    # ── Save arrays ───────────────────────────────────────────────────────────
    np.save(os.path.join(output_dir, "X.npy"), X_norm)
    np.save(os.path.join(output_dir, "y.npy"), y)
    joblib.dump(
        {"label2idx": label2idx, "idx2label": {v: k for k, v in label2idx.items()}},
        os.path.join(output_dir, "label_encoder.pkl"),
    )

    print(f"Saved features → {output_dir}")
    return X_norm, y, label2idx


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract RAVDESS features (ETASR config)")
    parser.add_argument("--manifest", default="../data/processed/manifest.csv")
    parser.add_argument("--out_dir",  default="../data/processed/features")
    args = parser.parse_args()

    build_feature_dataset(args.manifest, args.out_dir)
