# xingxing wake-word project

Bootstrapped by `scripts/init_wake.py`.

## Trigger phrases

- `星星`

## Wake word: `星星`

## Next steps

```bash
# 1. Optionally ask an LLM for harder collisions.
python scripts/suggest_hard_negatives.py --name xingxing

# 2. Synthesize project audio.
python scripts/synth_positives.py --project xingxing
python scripts/synth_hard_negatives.py --project xingxing

# 3. Download shared negatives and build device-compatible features.
python scripts/download_hf_negatives.py --out data/negative_datasets
python scripts/build_features.py --project xingxing --download-aug-corpora

# 4. Train on an existing GPU host (or use runpod_train.py).
python scripts/train_microwakeword.py --project xingxing

# 5. Seed held-out tasks and evaluate.
python scripts/seed_eval_tasks.py --project xingxing --bulk-audio-dir data/raw/negatives
python -m eval.runner --project xingxing --model models/xingxing-wakeword-v0.tflite

# 6. Emit an ESPHome manifest from a measured operating point.
python scripts/emit_manifest.py --project xingxing --eval-json eval/results/xingxing-v0__latest.json
```
