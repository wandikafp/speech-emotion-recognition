"""
CNN-BiLSTM + Attention model for Speech Emotion Recognition.

Aligned with Zennou et al., "Real-Time Speech Emotion Recognition with a
CNN-BiLSTM-Attention Deep Learning Model", ETASR Vol.16 No.3, 2026.

Architecture (A5 — full proposed model):
  Input: (400 frames, 40 features)
    → Conv1D(64,  k=3, PReLU, MaxPool(2), Dropout(0.3))   [Block 1]
    → Conv1D(128, k=5, PReLU, MaxPool(2), Dropout(0.3))   [Block 2]
    → Conv1D(256, k=7, PReLU, MaxPool(2), Dropout(0.3))   [Block 3]
    → Bidirectional LSTM(64, return_sequences=True)
    → Soft Attention (weighted sum over time)
    → Dense(128, PReLU, Dropout(0.3))
    → Dense(64,  PReLU, Dropout(0.3))
    → Dense(n_classes, Softmax)

Baseline (A4 — no attention):
  Same as above but BiLSTM returns final state (no attention).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ── Soft Attention Layer (ETASR paper eq. 5–6) ────────────────────────────────

@keras.utils.register_keras_serializable()
class SoftAttention(layers.Layer):
    """
    Soft attention mechanism over BiLSTM outputs.

    For each time step t:
        score_t = tanh(W · h_t)          [learned weight vector W]
        alpha_t = softmax(score_t)        [normalised weights]
        context = Σ alpha_t * h_t         [weighted sum]

    This matches the formulation in Zennou et al. (2026), Eq. 5–6.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # W: (feature_dim,) — one weight per feature dimension
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, encoder_output):
        # encoder_output: (batch, T, features)
        score   = tf.nn.tanh(encoder_output * self.W)   # (batch, T, features)
        score   = tf.reduce_sum(score, axis=-1, keepdims=True)  # (batch, T, 1)
        weights = tf.nn.softmax(score, axis=1)           # (batch, T, 1)
        context = tf.reduce_sum(weights * encoder_output, axis=1)  # (batch, features)
        return context, weights

    def get_config(self):
        return super().get_config()


# ── Helper: PReLU block ────────────────────────────────────────────────────────

def prelu_block(x, filters: int, kernel_size: int, dropout: float,
                name_prefix: str, l2: float = 1e-4):
    """Conv1D → PReLU → MaxPool(2) → Dropout (paper Table I)."""
    x = layers.Conv1D(
        filters, kernel_size, padding="same",
        kernel_regularizer=regularizers.l2(l2),
        name=f"{name_prefix}_conv"
    )(x)
    x = layers.PReLU(name=f"{name_prefix}_prelu")(x)
    x = layers.MaxPooling1D(2, name=f"{name_prefix}_pool")(x)
    x = layers.Dropout(dropout, name=f"{name_prefix}_drop")(x)
    return x


# ── Baseline model: A4 (CNN + BiLSTM, no attention) ──────────────────────────

def build_baseline_model(
    input_shape: tuple,
    n_classes: int,
    conv1_filters: int   = 64,   conv1_kernel: int = 3,
    conv2_filters: int   = 128,  conv2_kernel: int = 5,
    conv3_filters: int   = 256,  conv3_kernel: int = 7,
    lstm_units:    int   = 64,
    dense1_units:  int   = 128,
    dense2_units:  int   = 64,
    dropout_conv:  float = 0.3,
    dropout_dense: float = 0.3,
    l2_reg:        float = 1e-4,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Configuration A4 from ETASR paper:
      CNN (3 blocks) + BiLSTM (no attention) → 90.21% on RAVDESS.
    """
    inp = keras.Input(shape=input_shape, name="input_features")

    # CNN blocks
    x = prelu_block(inp, conv1_filters, conv1_kernel, dropout_conv, "conv1", l2=l2_reg)
    x = prelu_block(x,   conv2_filters, conv2_kernel, dropout_conv, "conv2", l2=l2_reg)
    x = prelu_block(x,   conv3_filters, conv3_kernel, dropout_conv, "conv3", l2=l2_reg)

    # BiLSTM — returns final hidden state only (no attention)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False,
                    kernel_regularizer=regularizers.l2(l2_reg)), name="bilstm"
    )(x)

    # Dense head
    x   = layers.Dense(dense1_units, kernel_regularizer=regularizers.l2(l2_reg), name="dense1")(x)
    x   = layers.PReLU(name="dense1_prelu")(x)
    x   = layers.Dropout(dropout_dense, name="drop_dense1")(x)
    x   = layers.Dense(dense2_units, kernel_regularizer=regularizers.l2(l2_reg), name="dense2")(x)
    x   = layers.PReLU(name="dense2_prelu")(x)
    x   = layers.Dropout(dropout_dense, name="drop_dense2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="CNN_BiLSTM_A4")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Attention model: A5 (proposed, CNN + BiLSTM + SoftAttention) ─────────────

def build_attention_model(
    input_shape: tuple,
    n_classes: int,
    conv1_filters: int   = 64,   conv1_kernel: int = 3,
    conv2_filters: int   = 128,  conv2_kernel: int = 5,
    conv3_filters: int   = 256,  conv3_kernel: int = 7,
    lstm_units:    int   = 64,
    dense1_units:  int   = 128,
    dense2_units:  int   = 64,
    dropout_conv:  float = 0.3,
    dropout_dense: float = 0.3,
    l2_reg:        float = 1e-4,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Configuration A5 — proposed model from ETASR paper:
      CNN (3 blocks) + BiLSTM + Soft Attention → 91.74% on RAVDESS.
    """
    inp = keras.Input(shape=input_shape, name="input_features")

    # CNN blocks (3 × PReLU + MaxPool + Dropout)
    x = prelu_block(inp, conv1_filters, conv1_kernel, dropout_conv, "conv1", l2=l2_reg)
    x = prelu_block(x,   conv2_filters, conv2_kernel, dropout_conv, "conv2", l2=l2_reg)
    x = prelu_block(x,   conv3_filters, conv3_kernel, dropout_conv, "conv3", l2=l2_reg)

    # BiLSTM — returns sequences for attention
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True,
                    kernel_regularizer=regularizers.l2(l2_reg)), name="bilstm"
    )(x)

    # Soft attention
    context, _ = SoftAttention(name="attention")(x)

    # Dense head
    x   = layers.Dense(dense1_units, kernel_regularizer=regularizers.l2(l2_reg), name="dense1")(context)
    x   = layers.PReLU(name="dense1_prelu")(x)
    x   = layers.Dropout(dropout_dense, name="drop_dense1")(x)
    x   = layers.Dense(dense2_units, kernel_regularizer=regularizers.l2(l2_reg), name="dense2")(x)
    x   = layers.PReLU(name="dense2_prelu")(x)
    x   = layers.Dropout(dropout_dense, name="drop_dense2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="CNN_BiLSTM_Attention_A5")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Backward-compatible alias ─────────────────────────────────────────────────
# Keep old name so existing notebooks still work
BahdanauAttention = SoftAttention


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT_SHAPE = (400, 40)   # 400 frames × 40 features (ETASR config)
    N_CLASSES   = 8

    print("=== A4 — Baseline (no attention) ===")
    baseline = build_baseline_model(INPUT_SHAPE, N_CLASSES)
    baseline.summary(line_length=100)

    print("\n=== A5 — Proposed (with attention) ===")
    attention = build_attention_model(INPUT_SHAPE, N_CLASSES)
    attention.summary(line_length=100)
