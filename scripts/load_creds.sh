#!/bin/bash
# Source this from any script that needs credentials.
# All env files live at ~/.config/tofu-wake/ with chmod 600.
# Never commit these files; never echo their values.

CREDS_DIR="${HOME}/.config/tofu-wake"

if [ ! -d "$CREDS_DIR" ]; then
    echo "ERROR: ${CREDS_DIR} does not exist. See REPLICATE.md §1." >&2
    return 1 2>/dev/null || exit 1
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

for f in gh.env together.env; do
    if [ -f "${CREDS_DIR}/${f}" ]; then
        set -a
        source "${CREDS_DIR}/${f}"
        set +a
    fi
done

# Sanity-check the required env vars without echoing values.
[ -z "$HF_TOKEN" ]       && { echo "ERROR: HF_TOKEN not set" >&2; return 1 2>/dev/null || exit 1; }
[ -z "$RUNPOD_API_KEY" ] && { echo "ERROR: RUNPOD_API_KEY not set" >&2; return 1 2>/dev/null || exit 1; }
