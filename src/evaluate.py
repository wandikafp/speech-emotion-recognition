"""
Evaluation module for Speech Emotion Recognition (SER).

Usage:
    python src/evaluate.py --model models/<model>.keras
"""

import os
import sys
import json
import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score,
    roc_curve, auc
)
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
MODELS_DIR   = os.path.join(BASE_DIR, "models")


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_split(split_name: str) -> tuple:
    split_dir = os.path.join(FEATURES_DIR, split_name)
    X = np.load(os.path.join(split_dir, "X.npy"))
    y = np.load(os.path.join(split_dir, "y.npy"))
    return X, y


def load_label_encoder() -> dict:
    enc_path = os.path.join(FEATURES_DIR, "train", "label_encoder.pkl")
    return joblib.load(enc_path)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix",
                           save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{title} — Counts")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalised
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"{title} — Normalised")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved → {save_path}")
    plt.show()


def plot_training_history(history_path: str, save_path=None):
    """Plot accuracy & loss curves from a saved history JSON."""
    with open(history_path) as f:
        hist = json.load(f)

    epochs = range(1, len(hist["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(epochs, hist["accuracy"],     label="Train Accuracy")
    axes[0].plot(epochs, hist["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Training & Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(epochs, hist["loss"],     label="Train Loss")
    axes[1].plot(epochs, hist["val_loss"], label="Val Loss")
    axes[1].set_title("Training & Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved → {save_path}")
    plt.show()


def plot_per_class_metrics(report_dict: dict, class_names: list, save_path=None):
    """Bar chart of precision, recall, F1 per emotion class."""
    metrics = {
        "Precision": [report_dict[c]["precision"]    for c in class_names],
        "Recall":    [report_dict[c]["recall"]       for c in class_names],
        "F1-Score":  [report_dict[c]["f1-score"]     for c in class_names],
    }
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (metric, vals) in enumerate(metrics.items()):
        ax.bar(x + i * width, vals, width, label=metric)

    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=30)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class Metrics")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Per-class metrics saved → {save_path}")
    plt.show()


def plot_roc_curves(y_true, y_prob, class_names, save_path=None):
    """One-vs-rest ROC curves for all classes."""
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, cls in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curves saved → {save_path}")
    plt.show()


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate(model_path: str, split: str = "test",
             save_dir: str = None) -> dict:
    """
    Evaluate a saved model on the given split and print/plot all metrics.

    Returns
    -------
    dict with accuracy, macro_f1, per_class report, and roc_auc
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating: {os.path.basename(model_path)}")
    print(f"  Split     : {split}")
    print(f"{'='*60}")

    # ── Load model & data ─────────────────────────────────────────────────────
    model = keras.models.load_model(model_path)
    X, y_true = load_split(split)
    enc       = load_label_encoder()
    idx2label = enc["idx2label"]
    class_names = [idx2label[i] for i in range(len(idx2label))]

    # ── Predictions ───────────────────────────────────────────────────────────
    y_prob = model.predict(X, verbose=0)      # (N, n_classes)
    y_pred = np.argmax(y_prob, axis=1)

    # ── Scalar metrics ────────────────────────────────────────────────────────
    acc       = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro")
    roc_auc   = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    report    = classification_report(y_true, y_pred,
                                      target_names=class_names, output_dict=True)

    print(f"\nTest Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Macro ROC-AUC : {roc_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ── Plots ─────────────────────────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plot_confusion_matrix(y_true, y_pred, class_names,
                          save_path=os.path.join(save_dir, "confusion_matrix.png") if save_dir else None)

    plot_per_class_metrics(report, class_names,
                           save_path=os.path.join(save_dir, "per_class_metrics.png") if save_dir else None)

    plot_roc_curves(y_true, y_prob, class_names,
                    save_path=os.path.join(save_dir, "roc_curves.png") if save_dir else None)

    # ── Error analysis: top confused pairs ───────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)   # remove correct predictions
    flat_idx = np.argsort(cm.flatten())[::-1][:5]
    print("\nTop-5 Most Confused Emotion Pairs:")
    for idx in flat_idx:
        true_idx = idx // len(class_names)
        pred_idx = idx  % len(class_names)
        print(f"  True='{class_names[true_idx]}' → Pred='{class_names[pred_idx]}' ({cm[true_idx, pred_idx]} times)")

    results = {
        "accuracy":  acc,
        "macro_f1":  macro_f1,
        "roc_auc":   roc_auc,
        "report":    report,
    }

    if save_dir:
        res_path = os.path.join(save_dir, "eval_results.json")
        with open(res_path, "w") as f:
            json.dump({k: v for k, v in results.items() if k != "report"}, f, indent=2)
        print(f"\nEval results saved → {res_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SER model")
    parser.add_argument("--model",    required=True, help="Path to .keras model file")
    parser.add_argument("--split",    default="test", choices=["train", "val", "test"])
    parser.add_argument("--save_dir", default=None,
                        help="Directory to save plots and metrics (optional)")
    parser.add_argument("--history",  default=None,
                        help="Path to training history JSON to plot curves")
    args = parser.parse_args()

    evaluate(args.model, split=args.split, save_dir=args.save_dir)

    if args.history:
        plot_training_history(args.history,
                              save_path=os.path.join(args.save_dir, "training_curves.png")
                              if args.save_dir else None)
