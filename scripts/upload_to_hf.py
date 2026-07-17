#!/usr/bin/env python3
"""Upload a trained wake-word .tflite to HuggingFace Hub, with a generated model card.

Usage:
    python scripts/upload_to_hf.py --project tofu \\
        --model models/tofu-wakeword-v0.tflite \\
        --repo-id <you>/tofu-wakeword-v0 \\
        --eval-json eval/results/tofu-v0__1715800000.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


MODEL_CARD_TEMPLATE = """---
license: other
license_name: training-data-dependent
tags:
  - wake-word
  - keyword-spotting
  - microwakeword
  - tflite
  - esp32
  - on-device
library_name: tflite
---

# {project_title}

> Custom INT8 TFLite Micro wake-word detector for **ESP32-S3**. Fires on:
{trigger_block}

## Metrics

| Metric | Value |
|---|---|
| FRR (positives) | {frr:.2%} |
| FAR (bulk negatives, /hour) | {far_per_hour:.2f} |
| Detection threshold | {threshold:.3f} |
| Model size | {size_kb:.1f} kB |
| Audio features | {n_mels} MicroFrontend channels, {hop_ms} ms step |

Per-bucket FAR:

{per_bucket_table}

## Use with ESPHome

```yaml
micro_wake_word:
  microphone: tofu_mic
  vad:
  models:
    - model: https://huggingface.co/{repo_id}/resolve/main/manifest.json
```

The manifest references `{filename}` and contains the measured threshold.

## Training recipe

Trained with the [customWakeWord](https://github.com/temm1e-labs/customWakeWord) pipeline:

1. ~{n_positives} synthetic positives across {n_phrases} trigger phrases ({tts_engines}).
2. ~{n_hard_negs} hand- and LLM-curated hard-negatives covering {n_buckets} collision categories.
3. ~300 hours of bulk negative audio sampled from MUSAN, DEMAND, Common Voice, AudioSet.
4. audiomentations chain: RIR convolution + additive noise + codec degradation + pitch/speed jitter.
5. microWakeWord MixedNet streaming architecture, INT8 quantized for tflite-micro on ESP32-S3.

See the repo README for replicability.

## License

Code is Apache 2.0. Model redistribution terms also depend on every TTS voice
and negative dataset used for this particular training run. The default
precomputed microWakeWord negative dataset is non-commercial; verify or replace
training corpora before commercial use.

## Citation

```bibtex
@misc{{{project}-wakeword-v0,
  title  = {{ {project} Wake Word v0: custom keyword spotter for ESP32-S3 }},
  year   = {{ 2026 }},
  url    = {{ https://huggingface.co/{repo_id} }},
  note   = {{ Apache 2.0. Built with the customWakeWord toolkit. }}
}}
```
"""


def render_model_card(args, eval_data: dict, manifest_data: dict) -> str:
    triggers = manifest_data.get("phrases", [])
    trigger_block = "\n".join(f"- `{p}`" for p in triggers) or "- (see config)"

    summary = eval_data.get("summary", {})
    threshold = summary.get("threshold", 0.85)
    frr = summary.get("frr", 0.0)
    far_per_hour = summary.get("far_per_hour", 0.0)
    per_bucket = summary.get("per_bucket_far", {})

    per_bucket_table = "| Bucket | FAR |\n|---|---|\n"
    if per_bucket:
        for bid, val in sorted(per_bucket.items()):
            per_bucket_table += f"| {bid} | {val:.2%} |\n"
    else:
        per_bucket_table += "| (no buckets) | — |\n"

    size_bytes = Path(args.model).stat().st_size
    return MODEL_CARD_TEMPLATE.format(
        project=args.project,
        project_title=args.project.title() + " Wake Word",
        repo_id=args.repo_id,
        filename=Path(args.model).name,
        trigger_block=trigger_block,
        frr=frr,
        far_per_hour=far_per_hour,
        threshold=threshold,
        size_kb=size_bytes / 1024,
        n_mels=40, hop_ms=10,
        n_positives=manifest_data.get("n_positives", "~10000"),
        n_hard_negs=manifest_data.get("n_hard_negatives", "~2500"),
        n_phrases=len(triggers) or "several",
        n_buckets=manifest_data.get("n_hard_negative_buckets", 5),
        tts_engines=", ".join(manifest_data.get("tts_engines", ["Piper"])),
        per_bucket_table=per_bucket_table,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--model", required=True, help="Path to the .tflite artefact.")
    ap.add_argument("--repo-id", required=True, help="HF repo id, e.g. you/tofu-wakeword-v0")
    ap.add_argument("--eval-json", required=True,
                    help="eval/results/<project>-v0__*.json — pulls metrics for the model card.")
    ap.add_argument("--esphome", required=True,
                    help="ESPHome manifest JSON or YAML to upload.")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import create_repo, upload_file

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set — source scripts/load_creds.sh first", file=sys.stderr)
        return 1

    if not Path(args.model).is_file():
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        return 1
    if not Path(args.eval_json).is_file():
        print(f"ERROR: eval result not found: {args.eval_json}", file=sys.stderr)
        return 1
    if not Path(args.esphome).is_file():
        print(f"ERROR: ESPHome metadata not found: {args.esphome}", file=sys.stderr)
        return 1
    eval_data = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    summary = eval_data.get("summary", {})
    if summary.get("n_positives", 0) <= 0 or summary.get("bulk_stream_minutes", 0) <= 0:
        print("ERROR: eval must include positive and measured bulk-negative tasks",
              file=sys.stderr)
        return 1

    # Pull phrase manifest from the project config
    cfg_path = Path(f"configs/examples/{args.project}/wake_phrases.yaml")
    manifest_data = {}
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        manifest_data["phrases"] = [p["text"] for p in cfg.get("phrases", [])]
        manifest_data["n_positives"] = sum(p.get("count", 0) for p in cfg.get("phrases", []))
        manifest_data["tts_engines"] = [
            e.get("name", "unknown") for e in cfg.get("engines", [])
        ]

    print(f"=== uploading to https://huggingface.co/{args.repo_id} ===")

    try:
        create_repo(args.repo_id, repo_type="model", token=token,
                    private=args.private, exist_ok=True)
    except Exception as e:
        print(f"WARN: create_repo: {e}")

    # Upload model
    upload_file(path_or_fileobj=args.model, path_in_repo=Path(args.model).name,
                repo_id=args.repo_id, token=token)
    print(f"  uploaded {args.model}")

    # Upload eval JSON for reproducibility
    if args.eval_json and Path(args.eval_json).exists():
        upload_file(path_or_fileobj=args.eval_json, path_in_repo="eval_results.json",
                    repo_id=args.repo_id, token=token)
        print(f"  uploaded {args.eval_json} as eval_results.json")

    # Upload the ESPHome model metadata under the name consumers expect.
    if args.esphome and Path(args.esphome).exists():
        metadata_name = "manifest.json" if Path(args.esphome).suffix == ".json" else "esphome.yaml"
        upload_file(path_or_fileobj=args.esphome, path_in_repo=metadata_name,
                    repo_id=args.repo_id, token=token)
        print(f"  uploaded {args.esphome} as {metadata_name}")

    # Render and upload model card
    card = render_model_card(args, eval_data, manifest_data)
    card_path = Path(args.model).with_suffix(".README.md")
    card_path.write_text(card, encoding="utf-8")
    upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md",
                repo_id=args.repo_id, token=token)
    print("  uploaded model card")

    print(f"\ndone: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
