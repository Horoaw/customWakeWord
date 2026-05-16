#!/usr/bin/env python3
"""Train + INT8-export a wake-word model via upstream OHF-Voice/micro-wake-word.

Thin wrapper around the upstream `microwakeword.model_train_eval` CLI.
Translates our project's YAML config into the argparse Namespace the
upstream code expects (architecture flags + training_parameters.yaml
path), runs the training loop, then runs the streaming INT8 quantize
pass to produce a `tflite-micro`-compatible artefact.

Output:
    trained_models/<project>/
        ├── best_weights.weights.h5
        ├── tflite_stream_state_internal_quant/
        │     └── stream_state_internal_quant.tflite     ← THE DEPLOYABLE ARTEFACT
        └── training_config.yaml                          (snapshot)

After training, the script copies `stream_state_internal_quant.tflite`
to `models/<project>-wakeword-v0.tflite` and prints the operating-point
table (cutoff vs recall vs FA/hr) so you can pick `probability_cutoff`
for the manifest.

Usage:
    python scripts/train_microwakeword.py --project tofu \\
        --training-config configs/examples/tofu/training_parameters.yaml
"""
from __future__ import annotations

import argparse
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import yaml


# Default MixedNet architecture flags (canonical recipe from the upstream notebook).
# These match the model that ships as Okay Nabu / Hey Jarvis / Alexa in ESPHome.
# Type rules (from reading upstream `microwakeword/mixednet.py`):
#   - String, parsed via `parse(...)`: pointwise_filters, repeat_in_block,
#     mixconv_kernel_sizes, residual_connection. These hold comma-separated
#     lists / bracketed sublists; the parser splits the string.
#   - Int, used directly in arithmetic: first_conv_filters (compared to 0,
#     used as conv filter count), first_conv_kernel_size (subtracted from
#     int, used in kernel shape tuple), stride (used in integer division
#     by load_config, used in strides tuple).
# Mixing the two breaks at TypeError boundaries that are not always reached
# during quick sanity tests.
DEFAULT_MIXEDNET_FLAGS = {
    "pointwise_filters": "64,64,64,64",
    "repeat_in_block": "1,1,1,1",
    "mixconv_kernel_sizes": "[5],[7,11],[9,15],[23]",
    "residual_connection": "0,0,0,0",
    "first_conv_filters": 32,
    "first_conv_kernel_size": 5,
    "stride": 3,
}


def build_flags(training_config: str, train_dir: str,
                arch_flags: dict | None = None) -> Namespace:
    """Build the argparse Namespace `microwakeword.model_train_eval` expects.

    `microwakeword.model_train_eval.load_config(flags, model_module)` reads:
      - `flags.training_config`: path to YAML
      - everything in `flags.__dict__` that doesn't start with `_` ends up
        in `config["flags"]` (i.e., the architecture hyperparams)
    """
    arch = {**DEFAULT_MIXEDNET_FLAGS, **(arch_flags or {})}
    ns = Namespace(
        training_config=training_config,
        train=1,
        restore_checkpoint=0,
        test_tf_nonstreaming=0,
        test_tflite_nonstreaming=0,
        test_tflite_nonstreaming_quantized=0,
        test_tflite_streaming=0,
        test_tflite_streaming_quantized=1,         # ← the artefact we ship
        use_weights="best_weights",
    )
    # Add architecture flags. Strings (not parsed) — upstream's parser uses
    # `ast.literal_eval` so commas/brackets must look like Python literals.
    for k, v in arch.items():
        setattr(ns, k, v)
    return ns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--training-config", default=None,
                    help="Default: configs/examples/<project>/training_parameters.yaml")
    ap.add_argument("--train-dir", default=None,
                    help="Default: trained_models/<project>")
    ap.add_argument("--arch-overrides", default=None,
                    help="JSON of MixedNet flag overrides. Leave None for canonical recipe.")
    ap.add_argument("--copy-to", default=None,
                    help="After training, copy the .tflite here. "
                         "Default: models/<project>-wakeword-v0.tflite")
    args = ap.parse_args()

    project = args.project
    tc_path = Path(args.training_config) if args.training_config else Path(
        f"configs/examples/{project}/training_parameters.yaml")
    train_dir = Path(args.train_dir) if args.train_dir else Path(
        f"trained_models/{project}")
    copy_to = Path(args.copy_to) if args.copy_to else Path(
        f"models/{project}-wakeword-v0.tflite")

    if not tc_path.exists():
        print(f"ERROR: {tc_path} not found.", file=sys.stderr)
        return 1

    # Patch the YAML's train_dir to our convention before passing to upstream.
    cfg = yaml.safe_load(tc_path.read_text())
    cfg["train_dir"] = str(train_dir)
    patched = tc_path.parent / "_train_run.yaml"
    patched.write_text(yaml.safe_dump(cfg, sort_keys=False))

    arch_overrides = None
    if args.arch_overrides:
        import json
        arch_overrides = json.loads(args.arch_overrides)

    flags = build_flags(str(patched), str(train_dir), arch_overrides)
    print(f"=== microWakeWord train: project '{project}' ===", flush=True)
    print(f"  training_config: {patched}")
    print(f"  train_dir:       {train_dir}")
    print(f"  arch:            mixednet {flags.pointwise_filters} "
          f"k={flags.mixconv_kernel_sizes}", flush=True)
    print(flush=True)

    # Defer the import — these pull TF/Keras, which is expensive.
    import microwakeword.data as input_data
    import microwakeword.mixednet as mixednet
    from microwakeword.model_train_eval import (
        load_config, train_model, evaluate_model,
    )

    config = load_config(flags, mixednet)
    data_processor = input_data.FeatureHandler(config)
    model = mixednet.model(
        flags,
        shape=config["training_input_shape"],
        batch_size=config["batch_size"],
    )

    print("\n=== model summary ===", flush=True)
    model.summary()

    print("\n=== training ===", flush=True)
    train_model(config, model, data_processor, restore_checkpoint=False)

    print("\n=== evaluating + exporting INT8 streaming .tflite ===", flush=True)
    # evaluate_model emits the .tflite when test_tflite_streaming_quantized=1.
    evaluate_model(config, model, data_processor,
                   test_tf_nonstreaming=False,
                   test_tflite_nonstreaming=False,
                   test_tflite_nonstreaming_quantized=False,
                   test_tflite_streaming=False,
                   test_tflite_streaming_quantized=True)

    # Copy the deployable artefact to our conventional location.
    src = train_dir / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"
    if src.exists():
        copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, copy_to)
        size_kb = copy_to.stat().st_size / 1024
        print(f"\n=== artefact: {copy_to} ({size_kb:.1f} kB) ===", flush=True)
    else:
        print(f"WARN: expected {src} not produced", file=sys.stderr)
        return 2

    print(f"\nNext: pick `probability_cutoff` from the table above, then:")
    print(f"  python scripts/emit_manifest.py --project {project} --threshold <cutoff>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
