import streamlit as st
import os
import sys
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

st.header("📊 Model Analytics")

predictor = st.session_state.get("predictor")

if not predictor:
    st.warning("⚠️ Load model terlebih dahulu di sidebar.")
else:
    FEAT_DIR = os.path.join(BASE_DIR, "data", "processed", "features_random")

    # Model info
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Model Name", predictor.model.name)
    col_m2.metric("Classes", len(predictor.classes))
    col_m3.metric("Input Shape", str(predictor.model.input_shape[1:]))

    # ── Confusion matrix on test set ──────────────────────────────────────
    st.subheader("Confusion Matrix — Test Set")
    if st.button("🔄 Run Evaluation on Test Set"):
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

        X_test = np.load(os.path.join(FEAT_DIR, "test", "X.npy"))
        y_test = np.load(os.path.join(FEAT_DIR, "test", "y.npy"))

        with st.spinner("Running predictions..."):
            y_prob = predictor.model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_prob, axis=1)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="macro")

        col_e1, col_e2 = st.columns(2)
        col_e1.metric("Test Accuracy", f"{acc*100:.2f}%")
        col_e2.metric("Macro F1", f"{f1:.4f}")

        cm      = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig_cm, ax = plt.subplots(figsize=(10, 7), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=predictor.classes, yticklabels=predictor.classes, ax=ax)
        ax.set_title("Normalised Confusion Matrix", color="#a8b2d8")
        ax.set_xlabel("Predicted", color="#a8b2d8"); ax.set_ylabel("True", color="#a8b2d8")
        ax.tick_params(colors="#a8b2d8")
        st.pyplot(fig_cm)

        with st.expander("📋 Classification Report"):
            report_str = classification_report(y_test, y_pred, target_names=predictor.classes)
            st.code(report_str)

    # ── Training history ──────────────────────────────────────────────────
    st.subheader("Training History")
    hist_files = sorted(glob.glob(os.path.join(BASE_DIR, "models", "*history*.json")),
                        key=os.path.getmtime, reverse=True)
    if hist_files:
        selected_hist = st.selectbox("Pilih history file:", [os.path.basename(h) for h in hist_files])
        hist_path = os.path.join(BASE_DIR, "models", selected_hist)
        with open(hist_path) as f:
            hist = json.load(f)

        epochs = list(range(1, len(hist["accuracy"]) + 1))
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=epochs, y=hist["accuracy"],     name="Train Acc", line=dict(color="#e94560")))
        fig_hist.add_trace(go.Scatter(x=epochs, y=hist["val_accuracy"], name="Val Acc",   line=dict(color="#3498db")))
        fig_hist.add_trace(go.Scatter(x=epochs, y=hist["loss"],         name="Train Loss",line=dict(color="#e94560", dash="dot")))
        fig_hist.add_trace(go.Scatter(x=epochs, y=hist["val_loss"],     name="Val Loss",  line=dict(color="#3498db", dash="dot")))
        fig_hist.update_layout(
            title="Training Curves", xaxis_title="Epoch", yaxis_title="Value",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(color="#a8b2d8"),
            height=400,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Belum ada file history. Jalankan training terlebih dahulu.")
