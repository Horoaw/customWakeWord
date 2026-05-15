# READY_TO_TRAIN — pre-flight checklist

Walk this top to bottom. When every box is ticked, run **one command** to
start training.

---

## Phase A — credentials (do once)

```bash
mkdir -p ~/.config/tofu-wake && chmod 700 ~/.config/tofu-wake

cat > ~/.config/tofu-wake/hf.env << 'EOF'
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
EOF

cat > ~/.config/tofu-wake/runpod.env << 'EOF'
export RUNPOD_API_KEY=rpa_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF

cat > ~/.config/tofu-wake/gh.env << 'EOF'    # optional, only if GH releases
export GH_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF

cat > ~/.config/tofu-wake/together.env << 'EOF'   # optional, suggest_hard_negatives.py
export TOGETHER_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF

chmod 600 ~/.config/tofu-wake/*.env
```

- [ ] `~/.config/tofu-wake/hf.env` contains a valid HF token with **write** permission
- [ ] `~/.config/tofu-wake/runpod.env` contains a valid RunPod key
- [ ] **RunPod account has at least $5 credit** (this is the blocker noted in the user message)
- [ ] HuggingFace account exists; you have a username to publish under

Test:
```bash
source scripts/load_creds.sh
echo "HF set: ${HF_TOKEN:+yes}"        # must say "HF set: yes"
echo "RP set: ${RUNPOD_API_KEY:+yes}"  # must say "RP set: yes"
```

---

## Phase B — local repo + Python env

```bash
cd ~/Documents/Github/tofuWakeWord
python3.10 -m venv .venv && source .venv/bin/activate    # microwakeword needs 3.10
pip install --upgrade pip
pip install -r requirements.txt
```

- [ ] `.venv` activated
- [ ] `pip install` finishes without errors
  - Common gotcha: `pymicro-features` needs a C compiler. `xcode-select --install` on Mac fixes it.
  - Common gotcha: `microwakeword` install may fail if Python version isn't 3.10. Verify with `python --version`.
- [ ] `python -c "import microwakeword; print(microwakeword.__file__)"` runs cleanly

---

## Phase C — confirm Tofu project is initialized

```bash
ls configs/examples/tofu/
# should show: wake_phrases.yaml, hard_negatives.yaml, training_parameters.yaml, README.md
```

- [ ] `configs/examples/tofu/wake_phrases.yaml` exists, lists hey/hi/hello/okay tofu
- [ ] `configs/examples/tofu/hard_negatives.yaml` exists, has at least 5 buckets
- [ ] `configs/examples/tofu/training_parameters.yaml` exists (this is the upstream-schema YAML the trainer consumes)

(Optional) extend the hard-negatives with LLM suggestions:
```bash
source scripts/load_creds.sh         # needs TOGETHER_API_KEY
python scripts/suggest_hard_negatives.py --name tofu
```

---

## Phase D — local smoke test (optional but recommended)

Validates the data pipeline end-to-end on tiny inputs before paying for GPU.

```bash
# 1. Generate 50 sample positives across 4 phrases (~5 min on Mac MPS)
python scripts/synth_positives.py --project tofu --count 200

# 2. Generate 50 sample hard negatives
python scripts/synth_hard_negatives.py --project tofu --count 200

# 3. Peek at the manifests
wc -l data/tofu/synth/positives/manifest.jsonl
wc -l data/tofu/synth/hard_negatives/manifest.jsonl

# 4. Sanity-listen to one WAV
afplay $(find data/tofu/synth/positives -name '*.wav' | head -1)
```

- [ ] Manifest row counts roughly match expectations (~200 each)
- [ ] At least one positive WAV sounds like "hey tofu" (or your phrase) when played

If you skip this step, the failure-discovery point shifts to the RunPod pod
— still OK, just costs a few cents more if something's broken upstream.

---

## Phase E — go / no-go

When all above is ticked, **one command** launches the full training run:

```bash
source scripts/load_creds.sh
python scripts/runpod_train.py --project tofu \
    --hf-repo-id <YOUR_USERNAME>/tofu-wakeword-v0
```

Or via Claude Code: `/wake-train tofu --hf-repo-id <YOUR_USERNAME>/tofu-wakeword-v0`.

Expect:
- **Wall time**: 30-45 min (mostly waiting on TTS + HF download + train)
- **Cost**: ~$0.55 on 4090 SECURE; max $2.76 if the pipeline runs to the 4 h cap.
- **Artefacts produced**:
  - `models/tofu-wakeword-v0.tflite` (~50-100 kB)
  - `configs/examples/tofu/manifest.json` (ESPHome JSON sidecar)
  - `eval/results/tofu-v0__<timestamp>.json` (operating-point table)
  - HuggingFace repo at `https://huggingface.co/<user>/tofu-wakeword-v0`

---

## Phase F — after training

Look at the operating-point table the trainer printed:
```
cutoff | recall | far/hr
0.50   | 0.99   | 1.20
0.70   | 0.98   | 0.60
0.85   | 0.96   | 0.30  ← typical pick
0.95   | 0.91   | 0.10
```

Pick the highest cutoff where `recall ≥ 0.95` and `far/hr ≤ 0.5`. Re-run
the manifest emitter if you want to change it:
```bash
python scripts/emit_manifest.py --project tofu --threshold 0.85
```

Then flash an ESP32-S3 with the ESPHome YAML at
`configs/examples/tofu/esphome.yaml`. See [`REPLICATE.md`](REPLICATE.md) §11.

---

## What to do if it goes wrong

| Symptom | Most likely cause | Fix |
|---|---|---|
| Pod stuck at "image pull" | RunPod queue | Wait 5 min; if still stuck, change GPU to A40 |
| `microwakeword` install fails | Python version drift | Pin to `python3.10` Docker image (already done in `runpod_train.py`) |
| `piper-sample-generator: file not found` | Silent clone failure | Re-run; or pre-clone before launching |
| Eval shows `far_per_hour > 50` | INT8 quantization collapse (rare) | Don't touch architecture; reduce `negative_class_weight` to 10 and retrain |
| Eval shows FRR > 20% | Insufficient positive diversity | Re-synth with `--count 30000` |
| Model triggers on "I love tofu" | Hard-neg bucket too weak | Bump `tofu_no_greeting` bucket count to 1000 in `hard_negatives.yaml` and retrain |

Detailed iteration playbook: [`TRAINING_PLAN.md`](TRAINING_PLAN.md) §7.

---

## $ hard cap

Before running, decide: **$10** budget cap is the recommended ceiling for v0.
If a single training run exceeds $3, kill the pod and review the log
before continuing.
