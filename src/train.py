"""
Training pipeline for Speech Emotion Recognition (SER).
Aligned with Zennou et al. ETASR 2026:
  - batch_size = 32, epochs = 300
  - EarlyStopping monitors val_loss
  - Data augmentation applied on train set

Usage:
    python src/train.py                   # train attention model (A5, default)
    python src/train.py --model baseline  # train baseline (A4)
    python src/train.py --model attention # train attention model (A5)
    python src/train.py --tune            # random-search hyperparameter tuning
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import joblib
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

# Allow running from project root or src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_baseline_model, build_attention_model

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR  = os.path.join(BASE_DIR, "data", "processed", "features")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(split_name: str) -> tuple:
    """Load pre-extracted feature arrays for a given split (train/val/test)."""
    split_dir = os.path.join(FEATURES_DIR, split_name)
    X = np.load(os.path.join(split_dir, "X.npy"))
    y = np.load(os.path.join(split_dir, "y.npy"))
    return X, y


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(model_name: str,
                  patience_es: int = 20,
                  patience_lr: int = 10):
    """Return standard Keras callbacks (ETASR paper config)."""
    ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_best.keras")
    return [
        # Paper: early stopping based on validation loss
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model_type: str = "attention",
    epochs:     int   = 300,
    batch_size: int   = 32,
    **hparams,
) -> tuple:
    """
    Load data, build model, compute class weights, and train.

    Returns
    -------
    (model, history, timestamp)
    """
    print(f"\n{'='*60}")
    print(f"  Training: {model_type.upper()} model  (ETASR config)")
    print(f"  epochs={epochs}, batch_size={batch_size}")
    print(f"{'='*60}")

    # ── Load data ──────────────────────────────────────────────────────────────
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    print(f"Train : X={X_train.shape}, y={y_train.shape}")
    print(f"Val   : X={X_val.shape},   y={y_val.shape}")

    n_classes   = len(np.unique(y_train))
    input_shape = X_train.shape[1:]   # (400, 40)

    # ── Class weights ──────────────────────────────────────────────────────────
    cw_vals = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(cw_vals))
    print(f"\nClass weights: { {k: round(v, 3) for k, v in class_weights.items()} }")

    # ── Build model ────────────────────────────────────────────────────────────
    if model_type == "baseline":
        model = build_baseline_model(input_shape, n_classes, **hparams)
    else:
        model = build_attention_model(input_shape, n_classes, **hparams)

    model.summary(line_length=100)

    # ── Train ──────────────────────────────────────────────────────────────────
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = get_callbacks(f"{model_type}_{ts}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save final model & history ─────────────────────────────────────────────
    final_path = os.path.join(MODELS_DIR, f"{model_type}_{ts}_final.keras")
    model.save(final_path)

    hist_path = os.path.join(MODELS_DIR, f"{model_type}_{ts}_history.json")
    hist_dict = {k: [float(v) for v in vals]
                 for k, vals in history.history.items()}
    with open(hist_path, "w") as f:
        json.dump(hist_dict, f, indent=2)

    print(f"\nModel saved   → {final_path}")
    print(f"History saved → {hist_path}")
    return model, history, ts


# ── Random-search hyperparameter tuning ──────────────────────────────────────

SEARCH_SPACE = {
    "conv1_filters":  [64, 128],
    "conv2_filters":  [128, 256],
    "conv3_filters":  [256, 512],
    "lstm_units":     [64, 128],
    "dense1_units":   [128, 256],
    "dense2_units":   [64, 128],
    "dropout_conv":   [0.2, 0.3, 0.4],
    "dropout_dense":  [0.2, 0.3, 0.4],
    "learning_rate":  [0.0001, 0.001],
    "batch_size":     [16, 32],
}


def random_search(n_trials: int = 10, epochs: int = 50) -> dict:
    """Run n_trials of random hyperparameter search and return best params."""
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    n_classes   = len(np.unique(y_train))
    input_shape = X_train.shape[1:]

    cw_vals       = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(cw_vals))

    best_val_acc = 0.0
    best_params  = {}
    results      = []

    for trial in range(1, n_trials + 1):
        params     = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
        batch_size = params.pop("batch_size")

        print(f"\n{'─'*60}")
        print(f"Trial {trial}/{n_trials}: {params}")

        model = build_attention_model(input_shape, n_classes, **params)
        cb = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
        ]
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=cb,
            verbose=0,
        )

        val_acc = max(hist.history["val_accuracy"])
        print(f"  → best val_accuracy: {val_acc:.4f}")
        results.append({
            "params":   {**params, "batch_size": batch_size},
            "val_acc":  val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params  = {**params, "batch_size": batch_size}
            best_model   = model

        keras.backend.clear_session()

    # Save results
    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(MODELS_DIR, f"random_search_{ts}.json")
    with open(results_path, "w") as f:
        json.dump({
            "best_val_acc": best_val_acc,
            "best_params":  best_params,
            "all_results":  results,
        }, f, indent=2)

    best_model.save(os.path.join(MODELS_DIR, f"best_tuned_{ts}.keras"))
    print(f"\n{'='*60}")
    print(f"Best val_accuracy : {best_val_acc:.4f}")
    print(f"Best params       : {best_params}")
    print(f"Results saved     → {results_path}")
    return best_params


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SER model (ETASR config)")
    parser.add_argument("--model",  choices=["baseline", "attention"],
                        default="attention",
                        help="A4 (baseline) or A5 (attention) model")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Max training epochs (paper: 300)")
    parser.add_argument("--batch",  type=int, default=32,
                        help="Batch size (paper: 32)")
    parser.add_argument("--tune",   action="store_true",
                        help="Run random-search hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of random-search trials")
    args = parser.parse_args()

    if args.tune:
        random_search(n_trials=args.trials, epochs=50)
    else:
        train(model_type=args.model, epochs=args.epochs, batch_size=args.batch)
