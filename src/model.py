from __future__ import annotations

import json
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import plots
from utils import FS, RESULTS_ROOT, WINDOW_SAMPLES, load_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

STEM = (
    (16, 15, 2, 1),
    (24, 7, 1, 2),
    (40, 7, 1, 2),
    (64, 5, 1, 2),
)

TRUNK_FILTERS = 32
TRUNK_KERNEL = 3
DILATIONS = (1, 2, 4, 8, 16, 32, 64)
SE_RATIO = 8
DROPOUT_TRUNK = 0.10
DROPOUT_HEAD = 0.30

EPOCHS = 70
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 12
REDUCE_LR_PATIENCE = 6

MODEL_PATH = RESULTS_ROOT / "model.keras"
HISTORY_PATH = RESULTS_ROOT / "history.json"


def receptive_field() -> tuple[int, int]:
    span, stride = 1, 1
    for _filters, kernel, conv_stride, pool in STEM:
        span += (kernel - 1) * stride
        stride *= conv_stride
        if pool > 1:
            span += (pool - 1) * stride
            stride *= pool
    for dilation in DILATIONS:
        effective_kernel = 1 + (TRUNK_KERNEL - 1) * dilation
        for _ in range(2):
            span += (effective_kernel - 1) * stride
    return span, stride


def _squeeze_excite(x, ratio: int, name: str):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling1D(name=f"{name}_se_squeeze")(x)
    se = layers.Dense(max(filters // ratio, 4), activation="relu",
                      name=f"{name}_se_reduce")(se)
    se = layers.Dense(filters, activation="sigmoid", name=f"{name}_se_expand")(se)
    se = layers.Reshape((1, filters), name=f"{name}_se_reshape")(se)
    return layers.Multiply(name=f"{name}_se_scale")([x, se])


def _residual_block(x, filters: int, kernel: int, dilation: int,
                    dropout: float, name: str):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same", use_bias=False,
                                 name=f"{name}_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    y = layers.Conv1D(filters, kernel, padding="same", dilation_rate=dilation,
                      use_bias=False, name=f"{name}_conv1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.Activation("relu", name=f"{name}_relu1")(y)
    y = layers.SpatialDropout1D(dropout, name=f"{name}_drop")(y)

    y = layers.Conv1D(filters, kernel, padding="same", dilation_rate=dilation,
                      use_bias=False, name=f"{name}_conv2")(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)
    y = _squeeze_excite(y, SE_RATIO, name)

    y = layers.Add(name=f"{name}_add")([shortcut, y])
    return layers.Activation("relu", name=f"{name}_out")(y)


def _attention_pool(x, name: str = "attn"):
    scores = layers.Conv1D(1, 1, name=f"{name}_score")(x)
    weights = layers.Softmax(axis=1, name=f"{name}_softmax")(scores)
    pooled = layers.Dot(axes=1, name=f"{name}_pool")([x, weights])
    return layers.Flatten(name=f"{name}_flat")(pooled)


def build_model(input_length: int = WINDOW_SAMPLES,
                n_channels: int = 1,
                trunk_filters: int = TRUNK_FILTERS,
                dropout_trunk: float = DROPOUT_TRUNK,
                name: str = "apnea_dilated_cnn") -> Model:
    span, _stride = receptive_field()
    assert span >= input_length, (
        f"receptive field {span} samples ({span / FS:.2f} s) does not cover "
        f"the {input_length}-sample window")

    inputs = layers.Input(shape=(input_length, n_channels), name="ecg")

    x = inputs
    for i, (filters, kernel, conv_stride, pool) in enumerate(STEM, start=1):
        if i == len(STEM):
            filters = trunk_filters
        x = layers.Conv1D(filters, kernel, strides=conv_stride, padding="same",
                          use_bias=False, name=f"stem{i}_conv")(x)
        x = layers.BatchNormalization(name=f"stem{i}_bn")(x)
        x = layers.Activation("relu", name=f"stem{i}_relu")(x)
        if pool > 1:
            x = layers.MaxPooling1D(pool, name=f"stem{i}_pool")(x)

    for i, dilation in enumerate(DILATIONS, start=1):
        x = _residual_block(x, trunk_filters, TRUNK_KERNEL, dilation,
                            dropout_trunk, name=f"block{i}_d{dilation}")

    pooled = layers.Concatenate(name="pool_concat")([
        _attention_pool(x),
        layers.GlobalAveragePooling1D(name="avg_pool")(x),
        layers.GlobalMaxPooling1D(name="max_pool")(x),
    ])

    y = layers.Dense(64, activation="relu", name="head_dense")(pooled)
    y = layers.Dropout(DROPOUT_HEAD, name="head_drop")(y)
    outputs = layers.Dense(1, activation="sigmoid", name="apnea_prob")(y)

    return Model(inputs, outputs, name=name)


def compile_model(model: Model, learning_rate: float = LEARNING_RATE) -> Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.Precision(name="precision")],
    )
    return model


def prepare() -> dict:
    train, val = load_split("train"), load_split("val")
    return {
        "X_tr": train["X"][..., np.newaxis], "y_tr": train["y"],
        "X_val": val["X"][..., np.newaxis], "y_val": val["y"],
    }


def train_dataset(X, y, seed: int):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


class LearningRateLog(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["lr"] = float(tf.keras.backend.get_value(
                self.model.optimizer.learning_rate))


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    data = prepare()
    model = compile_model(build_model())

    # model.summary()

    history = model.fit(
        train_dataset(data["X_tr"], data["y_tr"], SEED),
        validation_data=(data["X_val"], data["y_val"]),
        epochs=EPOCHS,
        callbacks=[
            LearningRateLog(),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=EARLY_STOP_PATIENCE,
                restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", mode="max", factor=0.5,
                patience=REDUCE_LR_PATIENCE, min_lr=1e-6, verbose=1),
        ],
        verbose=2,
    )

    model.save(str(MODEL_PATH))

    record = {k: [float(v) for v in vals]
              for k, vals in history.history.items()}
    HISTORY_PATH.write_text(json.dumps(record, indent=1))

    prob_val = model.predict(data["X_val"], verbose=0).ravel()

    print(f"  saved {MODEL_PATH}")
    print(f"  saved {HISTORY_PATH}")
    print(f"  saved {plots.plot_history(record, (data['y_val'], prob_val))}")


if __name__ == "__main__":
    main()
