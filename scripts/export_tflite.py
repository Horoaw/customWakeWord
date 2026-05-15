#!/usr/bin/env python3
"""INT8 post-training quantize a trained Keras model → TFLite Micro .tflite.

Designed for the microWakeWord ESP32-S3 runtime: 8-bit weights, 8-bit
activations, representative dataset for activation range calibration.

Usage:
    python scripts/export_tflite.py \\
        --project tofu \\
        --keras outputs/tofu/model.keras \\
        --data data/tofu/clean/train.tfrecord \\
        --config configs/train.yaml \\
        --out models/tofu-wakeword-v0.tflite \\
        --emit-esphome configs/esphome_tofu.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


def build_rep_dataset(tfrecord_path: Path, n_mels: int, n_frames: int, n_samples: int):
    import tensorflow as tf
    feat_desc = {
        "features": tf.io.FixedLenFeature([], tf.string),
        "shape": tf.io.FixedLenFeature([2], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse(record):
        ex = tf.io.parse_single_example(record, feat_desc)
        feats = tf.io.decode_raw(ex["features"], tf.float32)
        return tf.reshape(feats, [n_mels, n_frames])

    ds = tf.data.TFRecordDataset(str(tfrecord_path)).map(_parse).take(n_samples)

    def generator():
        for feat in ds:
            yield [tf.expand_dims(feat, 0)]

    return generator


def render_esphome(template_path: Path, model_url: str, probability_cutoff: float) -> str:
    text = template_path.read_text()
    text = text.replace("{{model_url}}", model_url)
    text = text.replace("{{probability_cutoff}}", f"{probability_cutoff:.3f}")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--keras", required=True, help="Path to the float Keras model.")
    ap.add_argument("--data", required=True, help="TFRecord for representative dataset.")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--out", required=True, help="Output .tflite path.")
    ap.add_argument("--emit-esphome", default=None,
                    help="Render the ESPHome YAML template to this path.")
    ap.add_argument("--esphome-template", default="configs/templates/esphome_template.yaml")
    ap.add_argument("--probability-cutoff", type=float, default=0.85)
    ap.add_argument("--hf-repo-id", default=None,
                    help="If set, embed the HF Hub model URL in the ESPHome YAML.")
    args = ap.parse_args()

    import tensorflow as tf

    cfg = yaml.safe_load(Path(args.config).read_text())
    n_mels = cfg["model"]["features"]["n_mels"]
    n_frames = cfg["model"]["features"]["n_frames"]
    rep_n = cfg["quantization"]["representative_n"]

    print(f"=== INT8 quantize {args.keras} ===")
    model = tf.keras.models.load_model(args.keras)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = build_rep_dataset(
        Path(args.data), n_mels, n_frames, rep_n,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_bytes = converter.convert()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_bytes)
    size_kb = len(tflite_bytes) / 1024
    print(f"  wrote {out_path} ({size_kb:.1f} kB)")

    if args.emit_esphome:
        if args.hf_repo_id:
            model_url = f"https://huggingface.co/{args.hf_repo_id}/resolve/main/{out_path.name}"
        else:
            model_url = f"./{out_path.name}"
        rendered = render_esphome(Path(args.esphome_template), model_url,
                                  args.probability_cutoff)
        Path(args.emit_esphome).write_text(rendered)
        print(f"  wrote ESPHome YAML to {args.emit_esphome}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
