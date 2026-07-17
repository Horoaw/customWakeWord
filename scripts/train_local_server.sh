#!/usr/bin/env bash
# One-command training launcher for a personal Linux NVIDIA server.
#
# Edit the "User configuration" section, then run from the repository root:
#
#   bash scripts/train_local_server.sh
#
# Public Hugging Face downloads do not require HF_TOKEN. The script uses the
# pre-built dependency image, so the host only needs Docker, an NVIDIA driver,
# and NVIDIA Container Toolkit (`docker run --gpus ...`).

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

# ASCII project slug. Use a NEW name when changing the wake phrase so old
# audio/features cannot accidentally be reused.
PROJECT_NAME="${PROJECT_NAME:-xingxing}"

# Comma-separated trigger phrases. For a single Mandarin wake word:
#   WAKE_PHRASES="星星"
# For multiple English variants:
#   WAKE_PHRASES="hey sunny,hi sunny,hello sunny"
WAKE_PHRASES="${WAKE_PHRASES:-星星}"
LANGUAGE="${LANGUAGE:-zh}"                       # zh or en

# Optional comma-separated per-phrase counts. Empty uses init_wake.py defaults
# (5000 for the first phrase, then 2500, 1500, 1000, ...).
POSITIVE_COUNTS="${POSITIVE_COUNTS:-}"

# Optional synthesis total overrides. Empty means use the generated YAML.
POSITIVE_TOTAL="${POSITIVE_TOTAL:-}"
HARD_NEGATIVE_TOTAL="${HARD_NEGATIVE_TOTAL:-}"

# Use a new version after a failed/finished training attempt instead of
# deleting checkpoints, for example MODEL_VERSION=v1.
MODEL_VERSION="${MODEL_VERSION:-v0}"

# Leave empty until a held-out evaluation justifies a cutoff. If set, a
# manifest is emitted even when no held-out eval tasks are present.
MANIFEST_THRESHOLD="${MANIFEST_THRESHOLD:-}"

# Dependency image built from this repository's Dockerfile.
TRAIN_IMAGE="${TRAIN_IMAGE:-ghcr.io/temm1e-labs/customwake-deps:v0.5}"

# Physical GPU index shown by `nvidia-smi`, or a full GPU UUID. GPU numbering
# starts at 0, so the eighth card is 7. Exactly one GPU is exposed to Docker.
GPU_DEVICE="${GPU_DEVICE:-0}"

# ---------------------------------------------------------------------------

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

stage() {
    printf '\n\033[1;34m==> %s\033[0m\n' "$*"
}

die() {
    printf '\nERROR: %s\n' "$*" >&2
    exit 1
}

if [[ ! "${PROJECT_NAME}" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
    die "PROJECT_NAME must be an ASCII slug using a-z, 0-9, _ or -."
fi
if [[ "${LANGUAGE}" != "zh" && "${LANGUAGE}" != "en" ]]; then
    die "LANGUAGE must be 'zh' or 'en'."
fi
if [[ -z "${WAKE_PHRASES//[[:space:]]/}" ]]; then
    die "WAKE_PHRASES cannot be empty."
fi
if [[ ! "${MODEL_VERSION}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    die "MODEL_VERSION may only contain letters, numbers, _, . or -."
fi
if [[ ! "${GPU_DEVICE}" =~ ^[0-9]+$ \
   && ! "${GPU_DEVICE}" =~ ^GPU-[a-fA-F0-9-]+$ ]]; then
    die "GPU_DEVICE must be a non-negative nvidia-smi index or a full GPU UUID."
fi

# Host-side entry: launch the same script inside the reproducible GPU image.
if [[ "${CUSTOMWAKE_IN_CONTAINER:-0}" != "1" ]]; then
    command -v docker >/dev/null 2>&1 || die "Docker is not installed."
    docker info >/dev/null 2>&1 || die "Docker daemon is not running or is not accessible."
    command -v nvidia-smi >/dev/null 2>&1 || die "NVIDIA driver/nvidia-smi is not available."

    selected_gpu_info="$(
        nvidia-smi -i "${GPU_DEVICE}" \
            --query-gpu=index,uuid,name,memory.total,driver_version \
            --format=csv,noheader 2>&1
    )" || die "GPU_DEVICE=${GPU_DEVICE} is unavailable: ${selected_gpu_info}"

    stage "Selected host GPU"
    printf '%s\n' "${selected_gpu_info}"
    GPU_REQUEST="device=${GPU_DEVICE}"

    docker_args=(
        run --rm --pull=missing --gpus "${GPU_REQUEST}" --shm-size=8g
        --user "$(id -u):$(id -g)"
        --env CUSTOMWAKE_IN_CONTAINER=1
        --env "GPU_DEVICE=${GPU_DEVICE}"
        --env EXPECTED_GPU_COUNT=1
        --env "PROJECT_NAME=${PROJECT_NAME}"
        --env "WAKE_PHRASES=${WAKE_PHRASES}"
        --env "LANGUAGE=${LANGUAGE}"
        --env "POSITIVE_COUNTS=${POSITIVE_COUNTS}"
        --env "POSITIVE_TOTAL=${POSITIVE_TOTAL}"
        --env "HARD_NEGATIVE_TOTAL=${HARD_NEGATIVE_TOTAL}"
        --env "MODEL_VERSION=${MODEL_VERSION}"
        --env "MANIFEST_THRESHOLD=${MANIFEST_THRESHOLD}"
        --env HOME=/tmp/customwake-home
        --volume "${REPO_DIR}:/workspace/customWakeWord"
        --workdir /workspace/customWakeWord
    )
    if [[ -n "${HF_TOKEN:-}" ]]; then
        docker_args+=(--env HF_TOKEN)
    fi

    stage "Starting training container: ${TRAIN_IMAGE}"
    exec docker "${docker_args[@]}" "${TRAIN_IMAGE}" \
        bash scripts/train_local_server.sh
fi

# Everything below runs inside the training container.
mkdir -p "${HOME}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PSG_DIR="${PIPER_SAMPLE_GENERATOR_DIR:-/opt/piper-sample-generator}"
CONFIG_DIR="configs/examples/${PROJECT_NAME}"
WAKE_CONFIG="${CONFIG_DIR}/wake_phrases.yaml"
TRAINING_CONFIG="${CONFIG_DIR}/training_parameters.yaml"
POSITIVE_MANIFEST="data/${PROJECT_NAME}/synth/positives/manifest.jsonl"
HARD_NEGATIVE_MANIFEST="data/${PROJECT_NAME}/synth/hard_negatives/manifest.jsonl"
TRAIN_DIR="trained_models/${PROJECT_NAME}-${MODEL_VERSION}"
MODEL_PATH="models/${PROJECT_NAME}-wakeword-${MODEL_VERSION}.tflite"

stage "Container preflight"
printf 'Requested host GPU: %s\n' "${GPU_DEVICE}"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version \
    --format=csv,noheader
"${PYTHON_BIN}" - <<'PY'
import os
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Python 3.10 required, got {sys.version.split()[0]}")

import tensorflow as tf
import microwakeword
import piper
import pymicro_features

gpus = tf.config.list_physical_devices("GPU")
expected_gpu_count = int(os.environ.get("EXPECTED_GPU_COUNT", "1"))
print(f"Python: {sys.version.split()[0]}")
print(f"TensorFlow: {tf.__version__}")
print(f"TensorFlow GPUs: {gpus}")
if len(gpus) != expected_gpu_count:
    raise SystemExit(
        f"TensorFlow expected {expected_gpu_count} GPU, but sees {len(gpus)}. "
        "Check GPU_DEVICE and NVIDIA Container Toolkit."
    )
print(f"Selected GPU details: {tf.config.experimental.get_device_details(gpus[0])}")
PY

stage "Wake-word configuration"
if [[ ! -f "${WAKE_CONFIG}" ]]; then
    init_args=(
        "${PYTHON_BIN}" scripts/init_wake.py
        --name "${PROJECT_NAME}"
        --phrases "${WAKE_PHRASES}"
        --language "${LANGUAGE}"
    )
    if [[ -n "${POSITIVE_COUNTS}" ]]; then
        init_args+=(--counts "${POSITIVE_COUNTS}")
    fi
    "${init_args[@]}"
else
    # Do not force-overwrite manually reviewed hard negatives or mix an old
    # project's generated data with a newly edited wake phrase.
    "${PYTHON_BIN}" - "${WAKE_CONFIG}" "${WAKE_PHRASES}" "${LANGUAGE}" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
expected_phrases = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
expected_language = sys.argv[3]
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
actual_phrases = [item["text"] for item in config.get("phrases", [])]
actual_language = config.get("language")

if actual_phrases != expected_phrases or actual_language != expected_language:
    raise SystemExit(
        f"Existing {config_path} does not match this launcher.\n"
        f"  launcher phrases/language: {expected_phrases!r} / {expected_language!r}\n"
        f"  existing phrases/language: {actual_phrases!r} / {actual_language!r}\n"
        "Choose a new PROJECT_NAME when changing the wake phrase."
    )
print(f"Using existing {config_path}: {actual_phrases!r} ({actual_language})")
PY
fi

[[ -f "${TRAINING_CONFIG}" ]] || die "Missing ${TRAINING_CONFIG}."
printf 'Project: %s\nWake phrase(s): %s\nLanguage: %s\n' \
    "${PROJECT_NAME}" "${WAKE_PHRASES}" "${LANGUAGE}"

stage "Synthesize positive samples"
positive_args=(
    "${PYTHON_BIN}" scripts/synth_positives.py
    --project "${PROJECT_NAME}"
    --psg-dir "${PSG_DIR}"
)
if [[ -n "${POSITIVE_TOTAL}" ]]; then
    positive_args+=(--count "${POSITIVE_TOTAL}")
fi
"${positive_args[@]}"
[[ -s "${POSITIVE_MANIFEST}" ]] || die "Positive manifest was not produced."

stage "Synthesize hard-negative samples"
hard_negative_args=(
    "${PYTHON_BIN}" scripts/synth_hard_negatives.py
    --project "${PROJECT_NAME}"
    --psg-dir "${PSG_DIR}"
)
if [[ -n "${HARD_NEGATIVE_TOTAL}" ]]; then
    hard_negative_args+=(--count "${HARD_NEGATIVE_TOTAL}")
fi
"${hard_negative_args[@]}"
[[ -s "${HARD_NEGATIVE_MANIFEST}" ]] || die "Hard-negative manifest was not produced."

stage "Download public bulk-negative features"
if [[ ! -d data/negative_datasets/speech \
   || ! -d data/negative_datasets/dinner_party \
   || ! -d data/negative_datasets/dinner_party_eval ]]; then
    "${PYTHON_BIN}" scripts/download_hf_negatives.py \
        --out data/negative_datasets
fi
for required_dataset in speech dinner_party dinner_party_eval; do
    [[ -d "data/negative_datasets/${required_dataset}" ]] \
        || die "Required negative dataset is missing: ${required_dataset}"
done

stage "Build microWakeWord features"
if [[ ! -d "data/${PROJECT_NAME}/features/training/wakeword_mmap" \
   || ! -d "data/${PROJECT_NAME}/hard_negatives_features/training/wakeword_mmap" ]]; then
    "${PYTHON_BIN}" scripts/build_features.py \
        --project "${PROJECT_NAME}" \
        --download-aug-corpora
else
    printf 'Feature stores already exist; feature build skipped.\n'
fi
[[ -d "data/${PROJECT_NAME}/features/training/wakeword_mmap" ]] \
    || die "Positive training features were not produced."
[[ -d "data/${PROJECT_NAME}/hard_negatives_features/training/wakeword_mmap" ]] \
    || die "Hard-negative training features were not produced."

stage "Train and export INT8 TFLite"
if [[ -s "${MODEL_PATH}" ]]; then
    printf 'Model already exists; training skipped: %s\n' "${MODEL_PATH}"
else
    if [[ -e "${TRAIN_DIR}" ]]; then
        die "${TRAIN_DIR} already exists but ${MODEL_PATH} does not. Set MODEL_VERSION to a new value (for example v1) and rerun."
    fi
    "${PYTHON_BIN}" scripts/train_microwakeword.py \
        --project "${PROJECT_NAME}" \
        --training-config "${TRAINING_CONFIG}" \
        --train-dir "${TRAIN_DIR}" \
        --copy-to "${MODEL_PATH}"
fi
[[ -s "${MODEL_PATH}" ]] || die "Training completed without producing ${MODEL_PATH}."

eval_dir="eval/tasks/${PROJECT_NAME}"
if find "${eval_dir}" -name '*.json' -print -quit 2>/dev/null | grep -q .; then
    stage "Held-out evaluation"
    eval_json="eval/results/${PROJECT_NAME}-${MODEL_VERSION}__$(date +%s).json"
    "${PYTHON_BIN}" -m eval.runner \
        --project "${PROJECT_NAME}" \
        --model "${MODEL_PATH}" \
        --out "${eval_json}"
    cp "${eval_json}" "eval/results/${PROJECT_NAME}-${MODEL_VERSION}__latest.json"

    stage "Emit evaluated ESPHome manifest"
    "${PYTHON_BIN}" scripts/emit_manifest.py \
        --project "${PROJECT_NAME}" \
        --tflite "${MODEL_PATH}" \
        --eval-json "${eval_json}"
elif [[ -n "${MANIFEST_THRESHOLD}" ]]; then
    stage "Emit manually-thresholded ESPHome manifest"
    "${PYTHON_BIN}" scripts/emit_manifest.py \
        --project "${PROJECT_NAME}" \
        --tflite "${MODEL_PATH}" \
        --threshold "${MANIFEST_THRESHOLD}"
else
    printf '\nNo held-out eval tasks found. Manifest generation was skipped.\n'
    printf 'After evaluation, rerun with MANIFEST_THRESHOLD=<measured cutoff>.\n'
fi

stage "Complete"
printf 'Model: %s\n' "${MODEL_PATH}"
printf 'Training directory: %s\n' "${TRAIN_DIR}"
