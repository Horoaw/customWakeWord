#!/bin/bash
# Source this from any script that needs credentials.
# All env files live at ~/.config/tofu-wake/ with chmod 600.
# Never commit these files; never echo their values.

CREDS_DIR="${HOME}/.config/tofu-wake"

if [ ! -d "$CREDS_DIR" ]; then
    # Fallback: TemLLM uses the same env-var names (HF_TOKEN, RUNPOD_API_KEY,
    # GH_TOKEN, TOGETHER_API_KEY) so its credential directory is drop-in
    # compatible. Lets a sibling project reuse existing tokens.
    FALLBACK="${HOME}/.config/temllm"
    if [ -d "$FALLBACK" ]; then
        echo "load_creds: using fallback ${FALLBACK}" >&2
        CREDS_DIR="$FALLBACK"
    else
        echo "ERROR: ${CREDS_DIR} and ${FALLBACK} both missing. See READY_TO_TRAIN.md Phase A." >&2
        return 1 2>/dev/null || exit 1
    fi
fi

# hf.env and runpod.env are required; gh.env + together.env are optional.
for f in hf.env runpod.env; do
    if [ ! -f "${CREDS_DIR}/${f}" ]; then
        echo "ERROR: missing ${CREDS_DIR}/${f}" >&2
        return 1 2>/dev/null || exit 1
    fi
    set -a
    source "${CREDS_DIR}/${f}"
    set +a
done

for f in gh.env together.env lambda.env; do
    if [ -f "${CREDS_DIR}/${f}" ]; then
        set -a
        source "${CREDS_DIR}/${f}"
        set +a
    fi
done

# Sanity-check the required env vars without echoing values.
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set" >&2
    return 1 2>/dev/null || exit 1
fi
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "ERROR: RUNPOD_API_KEY not set" >&2
    return 1 2>/dev/null || exit 1
fi
return 0 2>/dev/null || true
