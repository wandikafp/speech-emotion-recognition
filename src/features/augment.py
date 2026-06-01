"""
Data augmentation for Speech Emotion Recognition — ETASR 2026 paper config.

⚠️  IMPORTANT — HOW TO USE CORRECTLY:
Augmentation must be applied AT TRAINING TIME (each batch, different random
transform), NOT baked into saved .npy feature files.
Baking augmented audio into features creates a distribution mismatch:
  - Train set: augmented (noisy) MFCC features
  - Val/Test set: clean MFCC features
  → model learns "noisy" domain → poor val/test accuracy

Recommended usage:
  1. Extract features from CLEAN audio → save X_train.npy
  2. During model.fit(), apply augmentation via tf.data.Dataset or a custom
     Keras data generator that calls augment() on raw audio before feature
     extraction for each batch.

Alternatively, apply feature-space augmentation (e.g., SpecAugment) directly
on MFCC arrays — this is simpler and avoids the length-change problem of
time_stretch on raw audio.

Augmentation techniques (per paper Sec. II-B):
  - Noise addition  : inject Gaussian noise (simulate real acoustic env)
  - Time stretching : change duration without altering pitch (rate 0.8–1.2)
  - Time shifting   : apply small time shifts (±0.5 s)
  - Pitch shifting  : slightly adjust pitch (±2 semitones)
"""


import numpy as np
import librosa


def add_noise(y: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Add Gaussian noise at a given SNR (dB)."""
    signal_power = np.mean(y ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=y.shape)
    return (y + noise).astype(np.float32)


def time_stretch(y: np.ndarray,
                 rate_min: float = 0.8,
                 rate_max: float = 1.2) -> np.ndarray:
    """Randomly stretch / compress the time axis without affecting pitch."""
    rate = np.random.uniform(rate_min, rate_max)
    return librosa.effects.time_stretch(y.astype(np.float32), rate=rate)


def time_shift(y: np.ndarray, sr: int = 22050,
               shift_max: float = 0.5) -> np.ndarray:
    """Randomly shift the signal left or right by up to shift_max seconds."""
    shift = int(np.random.uniform(-shift_max, shift_max) * sr)
    return np.roll(y, shift).astype(np.float32)


def pitch_shift(y: np.ndarray, sr: int = 22050,
                semitones_max: float = 2.0) -> np.ndarray:
    """Randomly shift pitch by ±semitones_max semitones."""
    n_steps = np.random.uniform(-semitones_max, semitones_max)
    return librosa.effects.pitch_shift(
        y.astype(np.float32), sr=sr, n_steps=n_steps
    )


# ── Augmentation strategy ─────────────────────────────────────────────────────

# Each entry: (function, kwargs, probability)
AUGMENTATIONS = [
    (add_noise,    {"snr_db": 20.0},                          0.5),
    (time_stretch, {"rate_min": 0.85, "rate_max": 1.15},      0.4),
    (time_shift,   {"sr": 22050, "shift_max": 0.3},           0.4),
    (pitch_shift,  {"sr": 22050, "semitones_max": 2.0},       0.3),
]


def augment(y: np.ndarray, sr: int = 22050,
            p_apply: float = 1.0) -> np.ndarray:
    """
    Apply a random subset of augmentations to an audio array.

    Parameters
    ----------
    y        : Raw audio signal (float32 array)
    sr       : Sample rate (default 22050)
    p_apply  : Overall probability of applying any augmentation at all.
               Set to 1.0 to always augment (typical for training).

    Returns
    -------
    Augmented audio signal (same dtype as input, length may vary slightly).
    """
    if np.random.random() > p_apply:
        return y

    y_aug = y.copy()
    for fn, kwargs, prob in AUGMENTATIONS:
        if np.random.random() < prob:
            try:
                y_aug = fn(y_aug, **kwargs)
            except Exception:
                # Fallback: skip this augmentation if it fails
                pass

    return y_aug.astype(np.float32)
