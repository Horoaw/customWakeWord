#!/usr/bin/env python3
"""Train a microWakeWord model on the TFRecord splits produced by build_features.py.

Generic over projects: `--project <slug>` reads `data/<slug>/clean/{train,val,test}.tfrecord`
and writes the float Keras model to `outputs/<slug>/`. INT8 export to `models/<slug>-wakeword-v0.tflite`
happens via `scripts/export_tflite.py` (called automatically when --export is set).

Designed to run on a single 24-48 GB GPU. Trains in 1-2 h on an A40 for the
default ~60k positive features + ~12k hard-neg features + ~500k bulk windows/epoch.

Usage:
    python scripts/train_microwakeword.py \\
        --project tofu \\
        --config configs/train.yaml \\
        --out outputs/tofu/ \\
        --export models/tofu-wakeword-v0.tflite
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def parse_tfrecord(record_bytes, n_mels: int, n_frames: int):
    import tensorflow as tf
    feat_desc = {
        "features": tf.io.FixedLenFeature([], tf.string),
        "shape": tf.io.FixedLenFeature([2], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    ex = tf.io.parse_single_example(record_bytes, feat_desc)
    feats = tf.io.decode_raw(ex["features"], tf.float32)
    feats = tf.reshape(feats, [n_mels, n_frames])
    return feats, ex["label"]


def build_dataset(tfrecord_path: Path, n_mels: int, n_frames: int,
                  batch_size: int, shuffle_buffer: int, training: bool):
    import tensorflow as tf
    ds = tf.data.TFRecordDataset(str(tfrecord_path))
    ds = ds.map(lambda x: parse_tfrecord(x, n_mels, n_frames),
                num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(shuffle_buffer)
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_inception_model(n_mels: int, n_frames: int, inc_cfg: dict):
    """microWakeWord-style streaming Inception. Lean implementation in Keras 3.

    We use a simplified streaming-friendly variant: 1D convolutions over the
    time axis with the mel axis treated as channels. For maximum compatibility
    with the upstream microWakeWord runtime, we recommend cloning their repo
    and using their reference model code; this is a faithful reimpl for
    development. See https://github.com/kahrendt/microWakeWord.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=(n_mels, n_frames), name="features")
    x = layers.Permute((2, 1))(inputs)  # (time, mels) so Conv1D walks time
    for i, (filters, k) in enumerate(zip(inc_cfg["filters"], inc_cfg["kernel_strides"])):
        # Inception block: parallel 1x1 + kx1 + 2k x1 → concat
        b1 = layers.Conv1D(filters, 1, padding="same", activation="relu")(x)
        b3 = layers.Conv1D(filters, k, padding="same", activation="relu")(x)
        b5 = layers.Conv1D(filters, 2 * k - 1, padding="same", activation="relu")(x)
        x = layers.Concatenate(axis=-1)([b1, b3, b5])
        x = layers.BatchNormalization()(x)
        if inc_cfg.get("dropout", 0) > 0:
            x = layers.Dropout(inc_cfg["dropout"])(x)
        x = layers.MaxPool1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1, activation="sigmoid", name="prob")(x)
    return Model(inputs, out, name="microwakeword")


def far_at_recall(y_true, y_prob, target_recall: float = 0.99) -> float:
    """Compute the false-accept rate when the threshold is tuned to hit
    `target_recall` on the positives."""
    import numpy as np
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos_mask = y_true == 1
    if not pos_mask.any():
        return 1.0
    pos_probs = y_prob[pos_mask]
    sorted_probs = np.sort(pos_probs)
    # threshold = quantile that admits `target_recall` of positives
    k = int(np.floor((1 - target_recall) * len(sorted_probs)))
    k = max(0, min(k, len(sorted_probs) - 1))
    thr = sorted_probs[k]
    neg_mask = y_true == 0
    if not neg_mask.any():
        return 0.0
    far = float((y_prob[neg_mask] >= thr).mean())
    return far


class FARatRecallCallback:
    """Custom Keras callback computing val_far_at_99_recall after each epoch."""

    def __init__(self, val_ds, target_recall: float = 0.99, max_batches: int = 50):
        import tensorflow as tf
        self.tf = tf
        self.val_ds = val_ds
        self.target_recall = target_recall
        self.max_batches = max_batches
        self.history: list[float] = []

    def on_epoch_end(self, epoch, logs=None):
        import numpy as np
        y_true, y_prob = [], []
        for i, (x, y) in enumerate(self.val_ds):
            if i >= self.max_batches:
                break
            p = self._model.predict_on_batch(x).reshape(-1)
            y_true.append(y.numpy())
            y_prob.append(p)
        if not y_true:
            return
        far = far_at_recall(np.concatenate(y_true), np.concatenate(y_prob),
                            self.target_recall)
        self.history.append(far)
        if logs is not None:
            logs["val_far_at_99_recall"] = far
        print(f"  val_far_at_99_recall = {far:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--data-dir", default=None,
                    help="Default: data/<project>/clean")
    ap.add_argument("--out", default=None,
                    help="Default: outputs/<project>")
    ap.add_argument("--export", default=None,
                    help="If set, run scripts/export_tflite.py to produce an INT8 .tflite.")
    ap.add_argument("--max-steps-per-epoch", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=50)
    ap.add_argument("--skip-train", action="store_true",
                    help="Load --out as a trained Keras model and skip to export.")
    args = ap.parse_args()

    import tensorflow as tf
    cfg = yaml.safe_load(Path(args.config).read_text())
    project = args.project
    data_dir = Path(args.data_dir) if args.data_dir else Path(f"data/{project}/clean")
    out_dir = Path(args.out) if args.out else Path(f"outputs/{project}")
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = cfg["model"]["features"]
    n_mels = feat["n_mels"]
    n_frames = feat["n_frames"]

    print(f"=== train microWakeWord — project '{project}' ===")
    print(f"  data:    {data_dir}")
    print(f"  out:     {out_dir}")
    print(f"  config:  {args.config}")
    print()

    if cfg["training"].get("mixed_precision"):
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

    if args.skip_train and (out_dir / "model.keras").exists():
        model = tf.keras.models.load_model(out_dir / "model.keras")
        print("loaded pre-trained model")
    else:
        train_ds = build_dataset(data_dir / "train.tfrecord", n_mels, n_frames,
                                 batch_size=cfg["training"]["batch_size"],
                                 shuffle_buffer=cfg["training"]["shuffle_buffer"],
                                 training=True)
        val_ds = build_dataset(data_dir / "val.tfrecord", n_mels, n_frames,
                               batch_size=cfg["training"]["batch_size"],
                               shuffle_buffer=10,
                               training=False)

        model = build_inception_model(n_mels, n_frames, cfg["model"]["inception"])
        model.summary()

        lr = cfg["optimizer"]["learning_rate"]
        if cfg["optimizer"]["schedule"] == "cosine":
            steps = args.max_steps_per_epoch * cfg["training"]["epochs"]
            lr_sched = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr, decay_steps=steps,
            )
        else:
            lr_sched = lr

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_sched,
                                               weight_decay=cfg["optimizer"].get("weight_decay", 0)),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        far_cb = FARatRecallCallback(val_ds, target_recall=cfg["eval"]["target_recall"],
                                     max_batches=args.val_batches)
        far_cb._model = model

        class _CallbackShim(tf.keras.callbacks.Callback):
            def on_epoch_end(_self, epoch, logs=None):
                far_cb.on_epoch_end(epoch, logs)

        es_cfg = cfg["training"]["early_stop"]
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(out_dir / "model.keras"), save_best_only=True,
                monitor="val_loss", mode="min",
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor=es_cfg["metric"], patience=es_cfg["patience"],
                mode=es_cfg["mode"], min_delta=es_cfg.get("min_delta", 0),
                restore_best_weights=True,
            ),
            _CallbackShim(),
        ]

        # Class weights — positives weighted heavier
        cw = cfg["class_weights"]
        class_weight = {0: 1.0, 1: cw["positive"]}

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg["training"]["epochs"],
            steps_per_epoch=args.max_steps_per_epoch,
            validation_steps=args.val_batches,
            class_weight=class_weight,
            callbacks=callbacks,
        )

        # Final metrics dump
        (out_dir / "history.json").write_text(json.dumps(
            {k: [float(x) for x in v] for k, v in history.history.items()}, indent=2
        ))
        model.save(out_dir / "model.keras")
        print(f"saved model to {out_dir}/model.keras")

    if args.export:
        import subprocess
        cmd = ["python", "scripts/export_tflite.py",
               "--project", project,
               "--keras", str(out_dir / "model.keras"),
               "--out", args.export,
               "--data", str(data_dir / "train.tfrecord"),
               "--config", args.config]
        print(f"running: {' '.join(cmd)}")
        subprocess.check_call(cmd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
