#!/bin/bash
# Remote-side setup script — runs on the Lambda Labs instance via SSH.
#
# Invoked by scripts/lambda_train.py over SSH. The launcher pipes the
# necessary env vars (HF_TOKEN, GH_TOKEN, project, repo_url, branch,
# hf_repo_id) before sourcing this script.
#
# Logs all output to /workspace/setup.log AND streams to stdout (which
# the SSH session captures locally, eliminating the need for an in-pod
# http log server).
#
# Same shape as the RunPod entrypoint, minus:
# - Self-restarting log server (not needed; SSH is the log path)
# - apt-installing software-properties-common (Lambda images come with it)
# - dockerStartCmd wrapping
#
# Writes /workspace/STAGE (single-line marker) + /workspace/_done at end.

set -exo pipefail

# Required env vars (passed in by launcher)
: "${PROJECT:?PROJECT required}"
: "${REPO_URL:?REPO_URL required}"
: "${BRANCH:=main}"
: "${HF_TOKEN:?HF_TOKEN required}"
: "${GH_TOKEN:?GH_TOKEN required}"
HF_REPO_ID="${HF_REPO_ID:-}"

write_stage() {
    echo "$1" > /workspace/STAGE
    date +"%H:%M:%S" > /workspace/STAGE_TIME
    echo "[$(date +%H:%M:%S)] STAGE: $1"
}

sudo mkdir -p /workspace
sudo chown "$(whoami):$(whoami)" /workspace
cd /workspace

write_stage "boot"
echo "[$(date +%H:%M:%S)] host info: $(hostname) $(uname -r)"
echo "[$(date +%H:%M:%S)] base python: $(python3 --version 2>&1)"

write_stage "apt_install"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg sox libsox-fmt-mp3 unzip wget git \
    build-essential

# Lambda images typically ship python3.10. Verify; if not, install via deadsnakes.
if ! python3.10 --version >/dev/null 2>&1; then
    write_stage "python310_install"
    sudo apt-get install -y -qq software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.10 python3.10-venv python3.10-dev python3.10-distutils
fi
echo "[$(date +%H:%M:%S)] python3.10: $(python3.10 --version)"

write_stage "git_clone"
# Private repo — auth via GH_TOKEN. set +x around clone so token doesn't echo.
repo_path=$(echo "$REPO_URL" | sed -E 's|^https://github.com/||; s|\.git$||')
{ set +x; } 2>/dev/null
git clone --branch "$BRANCH" \
    "https://x-access-token:${GH_TOKEN}@github.com/${repo_path}.git" \
    /workspace/customwake 2>&1 | sed -E 's|x-access-token:[^@]*@|x-access-token:***@|g'
{ set -x; } 2>/dev/null
cd /workspace/customwake

write_stage "venv_create"
python3.10 -m venv /workspace/.venv
source /workspace/.venv/bin/activate
pip install --upgrade pip wheel

write_stage "pip_install_requirements"
pip install --no-cache-dir -r requirements.txt
echo "[$(date +%H:%M:%S)] deps installed: $(python --version)"

write_stage "synth_positives"
if [ ! -s "data/${PROJECT}/synth/positives/manifest.jsonl" ]; then
    python scripts/synth_positives.py --project "$PROJECT" \
        --psg-dir /workspace/piper-sample-generator
fi

write_stage "synth_hard_negatives"
if [ ! -s "data/${PROJECT}/synth/hard_negatives/manifest.jsonl" ]; then
    python scripts/synth_hard_negatives.py --project "$PROJECT" \
        --psg-dir /workspace/piper-sample-generator
fi

write_stage "download_hf_negatives"
if [ ! -d "data/negative_datasets/speech" ]; then
    python scripts/download_hf_negatives.py --out data/negative_datasets
fi

write_stage "build_features"
if [ ! -d "data/${PROJECT}/features/training/wakeword_mmap" ]; then
    python scripts/build_features.py --project "$PROJECT" \
        --download-aug-corpora
fi

write_stage "train"
python scripts/train_microwakeword.py --project "$PROJECT" \
    --training-config "configs/examples/${PROJECT}/training_parameters.yaml"

write_stage "eval"
if [ -d "eval/tasks/${PROJECT}" ]; then
    python -m eval.runner --project "$PROJECT" \
        --model "models/${PROJECT}-wakeword-v0.tflite" \
        --out "eval/results/${PROJECT}-v0__$(date +%s).json"
    cp $(ls -t eval/results/${PROJECT}-v0__*.json | head -1) \
       eval/results/${PROJECT}-v0__latest.json
fi

write_stage "emit_manifest"
python scripts/emit_manifest.py --project "$PROJECT"

if [ -n "$HF_REPO_ID" ]; then
    write_stage "upload_to_hf"
    echo "[$(date +%H:%M:%S)] uploading to HF: ${HF_REPO_ID}"
    python scripts/upload_to_hf.py --project "$PROJECT" \
        --model "models/${PROJECT}-wakeword-v0.tflite" \
        --repo-id "$HF_REPO_ID" \
        --eval-json "eval/results/${PROJECT}-v0__latest.json" \
        --esphome "configs/examples/${PROJECT}/manifest.json"
fi

write_stage "done"
echo "[$(date +%H:%M:%S)] DONE"
touch /workspace/_done
