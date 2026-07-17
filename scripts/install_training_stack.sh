#!/usr/bin/env bash
set -euo pipefail

# Install the two source-only training dependencies at known-good revisions.
#
# Environment overrides:
#   INSTALL_ROOT       clone destination root (default: /opt)
#   PYTHON_BIN         Python interpreter inside the target venv
#   MICRO_WAKE_WORD_REF / PIPER_SAMPLE_GENERATOR_REF
#
# Both Dockerfile and the Lambda launcher call this script so their training
# environments cannot silently drift apart.

INSTALL_ROOT="${INSTALL_ROOT:-/opt}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MICRO_WAKE_WORD_REPO="https://github.com/OHF-Voice/micro-wake-word.git"
MICRO_WAKE_WORD_REF="${MICRO_WAKE_WORD_REF:-4665173cd35f1cff9a61e06fc427f124766c488e}"
MICRO_WAKE_WORD_DIR="${MICRO_WAKE_WORD_DIR:-${INSTALL_ROOT}/microwakeword}"

PIPER_SAMPLE_GENERATOR_REPO="https://github.com/rhasspy/piper-sample-generator.git"
PIPER_SAMPLE_GENERATOR_REF="${PIPER_SAMPLE_GENERATOR_REF:-v3.0.0}"
PIPER_SAMPLE_GENERATOR_DIR="${PIPER_SAMPLE_GENERATOR_DIR:-${INSTALL_ROOT}/piper-sample-generator}"
PIPER_GENERATOR_MODEL="${PIPER_GENERATOR_MODEL:-${PIPER_SAMPLE_GENERATOR_DIR}/models/en_US-libritts_r-medium.pt}"
PIPER_GENERATOR_MODEL_URL="https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"

checkout_ref() {
    local repo="$1"
    local ref="$2"
    local destination="$3"

    mkdir -p "$(dirname "${destination}")"
    if [ -e "${destination}" ] && [ ! -d "${destination}/.git" ]; then
        echo "ERROR: ${destination} exists but is not a git checkout" >&2
        return 1
    fi

    if [ ! -d "${destination}/.git" ]; then
        git init -q "${destination}"
        git -C "${destination}" remote add origin "${repo}"
    fi

    git -C "${destination}" fetch -q --depth 1 origin "${ref}"
    local requested
    requested="$(git -C "${destination}" rev-parse FETCH_HEAD)"
    if git -C "${destination}" rev-parse --verify HEAD >/dev/null 2>&1; then
        local current
        current="$(git -C "${destination}" rev-parse HEAD)"
        if [ "${current}" != "${requested}" ]; then
            echo "ERROR: ${destination} is at ${current}, expected ${requested}." >&2
            echo "       Use an empty INSTALL_ROOT or remove that dependency checkout." >&2
            return 1
        fi
    else
        git -C "${destination}" checkout -q --detach FETCH_HEAD
    fi
}

echo "Installing microWakeWord ${MICRO_WAKE_WORD_REF}"
checkout_ref "${MICRO_WAKE_WORD_REPO}" "${MICRO_WAKE_WORD_REF}" "${MICRO_WAKE_WORD_DIR}"

# Upstream uses setuptools.find_packages(), while these namespace directories
# do not contain __init__.py. Add them before the editable install so feature
# generation and streaming layers are importable.
touch "${MICRO_WAKE_WORD_DIR}/microwakeword/audio/__init__.py"
touch "${MICRO_WAKE_WORD_DIR}/microwakeword/layers/__init__.py"
"${PYTHON_BIN}" -m pip install --no-cache-dir -e "${MICRO_WAKE_WORD_DIR}"
"${PYTHON_BIN}" -c \
    "from microwakeword.audio.augmentation import Augmentation; \
from microwakeword.audio.clips import Clips; \
from microwakeword.audio.spectrograms import SpectrogramGeneration; \
print('microWakeWord source install OK')"

echo "Installing Piper Sample Generator ${PIPER_SAMPLE_GENERATOR_REF}"
checkout_ref \
    "${PIPER_SAMPLE_GENERATOR_REPO}" \
    "${PIPER_SAMPLE_GENERATOR_REF}" \
    "${PIPER_SAMPLE_GENERATOR_DIR}"
"${PYTHON_BIN}" -m pip install --no-cache-dir -e "${PIPER_SAMPLE_GENERATOR_DIR}"

mkdir -p "$(dirname "${PIPER_GENERATOR_MODEL}")"
if [ ! -s "${PIPER_GENERATOR_MODEL}" ]; then
    wget -q -O "${PIPER_GENERATOR_MODEL}" "${PIPER_GENERATOR_MODEL_URL}"
fi
"${PYTHON_BIN}" "${PIPER_SAMPLE_GENERATOR_DIR}/generate_samples.py" --help >/dev/null

echo "Training stack installed under ${INSTALL_ROOT}"
