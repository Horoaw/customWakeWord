#!/bin/bash
# Retry-until-capacity launcher.
#
# RunPod has periodic capacity droughts that fail pod creation with
# "no instances currently available". This loops through preferred GPU
# types every minute and fires runpod_train.py the moment one accepts.
#
# On success, this script exits 0 with the full pipeline output.
# On unrecoverable error, exits 1.
#
# Tunables: GPUS, MAX_ATTEMPTS, SLEEP_BETWEEN.

set -uo pipefail

cd "$(dirname "$0")/.."
source scripts/load_creds.sh

# Preferred GPU pools in priority order. Format: "GPU_ID:CLOUD:HOURLY_RATE".
# Premium-first ordering — higher-tier hosts have meaningfully better network
# egress (observed 2026-05-15: 4090 SECURE choked on 9 GB HF Hub download for
# >2h; A100 SXM completed the same set in minutes per LESSONS_v0.md).
GPUS=(
    "NVIDIA A100-SXM4-80GB:SECURE:1.39"
    "NVIDIA L40S:SECURE:0.79"
    "NVIDIA L40:SECURE:0.69"
    "NVIDIA A40:SECURE:0.44"
    "NVIDIA GeForce RTX 4090:SECURE:0.69"
    "NVIDIA GeForce RTX 3090:SECURE:0.22"
    "NVIDIA GeForce RTX 4090:COMMUNITY:0.34"
)

MAX_ATTEMPTS=${MAX_ATTEMPTS:-30}
SLEEP_BETWEEN=${SLEEP_BETWEEN:-60}

PROJECT=${PROJECT:-tofu}
HF_REPO_ID=${HF_REPO_ID:-nagisanzeninz/tofu-wakeword-v0}

ATTEMPT=0
while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo "=========================================="
    echo "[$(date +%H:%M:%S)] RETRY ATTEMPT $ATTEMPT/$MAX_ATTEMPTS"
    echo "=========================================="

    for opt in "${GPUS[@]}"; do
        IFS=':' read -r gpu cloud rate <<< "$opt"
        echo "  → try $gpu $cloud \$$rate/hr"
        out=$(python3 scripts/runpod_train.py \
            --project "$PROJECT" \
            --hf-repo-id "$HF_REPO_ID" \
            --gpu "$gpu" \
            --cloud-type "$cloud" \
            --hourly-rate "$rate" 2>&1)
        rc=$?
        # Show last few lines for context
        echo "$out" | tail -6 | sed 's/^/    /'

        if [ "$rc" -eq 0 ]; then
            echo "[$(date +%H:%M:%S)] PIPELINE COMPLETE on $gpu $cloud"
            echo "$out"
            exit 0
        fi

        # Classify the error
        if echo "$out" | grep -qE "no instances currently available|no longer any instances|create pod:"; then
            echo "    [capacity miss, trying next pool]"
            continue
        fi

        if echo "$out" | grep -qE "STUCK: pod runtime null"; then
            echo "    [pod got stuck on host, trying next pool]"
            continue
        fi

        # Other failure → report and bail (don't burn $)
        echo "[$(date +%H:%M:%S)] UNEXPECTED FAILURE on $gpu $cloud:"
        echo "$out" | tail -30
        exit 1
    done

    echo "[$(date +%H:%M:%S)] all pools dry; sleeping ${SLEEP_BETWEEN}s before next round"
    sleep "$SLEEP_BETWEEN"
done

echo "[$(date +%H:%M:%S)] max attempts (${MAX_ATTEMPTS}) exhausted"
exit 1
