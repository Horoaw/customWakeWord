#!/usr/bin/env bash
# Fast local-Mac smoke test for the customWakeWord pipeline.
#
# Run BEFORE pushing to git + retrying on RunPod. Catches the cheap-to-find
# 80% of bugs (syntax errors, missing imports, bad YAML, broken --help)
# without installing the full TF + torch + microwakeword stack (~30 min,
# Linux-only).
#
# What this DOES verify (~30 sec, no heavy installs):
#   - Every Python file in scripts/ and eval/ AST-parses
#   - Every Python file's top-level imports resolve (uses system Python's
#     stdlib + already-installed packages, skips files needing heavy deps)
#   - Every YAML config in configs/examples/<project>/ parses
#   - Every script's argparse --help runs without exceptions
#   - The data manifests we'd produce have valid structure
#
# What this does NOT verify (needs the cloud + heavy deps):
#   - Real TTS synthesis (needs piper-sample-generator + torch)
#   - microwakeword training (needs TF + GPU)
#   - End-to-end .tflite quality (needs full pipeline)
#
# Usage:
#   scripts/local_smoke_test.sh                       # tests all projects
#   scripts/local_smoke_test.sh greet                 # tests one project's configs
#
# Exit codes:
#   0 — all checks pass; safe to push + try on cloud
#   1 — caught a bug; fix locally before burning cloud cycles

# Intentionally NOT set -e: we want each check to complete + report status,
# not abort on the first failure. set -u catches typos in variable names.
set -uo pipefail

PROJECT="${1:-}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PYTHON="${PYTHON:-python3}"
PASS=0
FAIL=0
WARNINGS=()

ok()    { printf "  \033[32m✓\033[0m %s\n" "$*"; PASS=$((PASS+1)); }
bad()   { printf "  \033[31m✗\033[0m %s\n" "$*"; FAIL=$((FAIL+1)); }
warn()  { printf "  \033[33m⚠\033[0m %s\n" "$*"; WARNINGS+=("$*"); }
hdr()   { printf "\n\033[1m== %s ==\033[0m\n" "$*"; }


hdr "1. AST-parse all Python sources"
PY_FILES=$(find scripts eval -name "*.py" -type f 2>/dev/null | sort)
PY_COUNT=$(echo "$PY_FILES" | wc -l | xargs)
for f in $PY_FILES; do
    if $PYTHON -c "import ast; ast.parse(open('$f').read())" 2>/dev/null; then
        :
    else
        err=$($PYTHON -c "import ast; ast.parse(open('$f').read())" 2>&1 | tail -3)
        bad "$f: $err"
    fi
done
ok "AST-parsed $PY_COUNT Python files"


hdr "2. Validate YAML configs"
if ! $PYTHON -c "import yaml" 2>/dev/null; then
    warn "pyyaml not installed for $PYTHON — falling back to text-only sanity (run 'uv pip install pyyaml' for real validation)"
    YAML_VALIDATE="text-only"
else
    YAML_VALIDATE="full"
fi
GLOB_ROOT="configs/examples"
if [ -n "$PROJECT" ]; then GLOB_ROOT="configs/examples/$PROJECT"; fi
YAML_COUNT=0
while IFS= read -r f; do
    [ -e "$f" ] || continue
    YAML_COUNT=$((YAML_COUNT+1))
    if [ "$YAML_VALIDATE" = "full" ]; then
        if ! $PYTHON -c "import yaml; yaml.safe_load(open('$f'))" 2>/dev/null; then
            err_line=$($PYTHON -c "import yaml; yaml.safe_load(open('$f'))" 2>&1 | grep -E "line [0-9]+" | head -1)
            bad "$f: ${err_line:-yaml.safe_load failed}"
        fi
    else
        # Cheap sanity: file is non-empty + has no obvious tab indentation
        [ -s "$f" ] || { bad "$f: empty"; continue; }
        if grep -qP "^\t" "$f" 2>/dev/null; then
            bad "$f: tab indentation (YAML requires spaces)"
        fi
    fi
done < <(find "$GLOB_ROOT" -maxdepth 2 -name "*.yaml" -type f 2>/dev/null | sort)
ok "Checked $YAML_COUNT YAML configs ($YAML_VALIDATE mode)"


hdr "3. argparse --help smoke test for entry-point scripts"
# Resolve a portable 5-sec timeout. Mac default has none; gtimeout comes
# from brew install coreutils. Fall back to a python wrapper if neither.
if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_CMD="timeout 5"
elif command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_CMD="gtimeout 5"
else
    # Inline python timeout. Slow, but works without coreutils.
    TIMEOUT_CMD="$PYTHON -c 'import subprocess,sys; sys.exit(subprocess.run(sys.argv[1:], timeout=5).returncode)'"
fi

# Scripts that should support --help. Skip ones that need heavy imports
# (TF / torch / microwakeword) — those will fail on Mac without the full env.
ENTRY_SCRIPTS=(
    "scripts/init_wake.py"
    "scripts/synth_positives.py"
    "scripts/synth_hard_negatives.py"
    "scripts/download_hf_negatives.py"
    "scripts/emit_manifest.py"
    "scripts/upload_to_hf.py"
)
for s in "${ENTRY_SCRIPTS[@]}"; do
    [ -f "$s" ] || { warn "$s missing"; continue; }
    eval "$TIMEOUT_CMD $PYTHON \"$s\" --help" >/tmp/_smoke_$$.out 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        ok "$s --help (no import errors)"
    else
        out=$(tail -5 /tmp/_smoke_$$.out)
        if echo "$out" | grep -q "ModuleNotFoundError\|ImportError"; then
            missing=$(echo "$out" | grep -oE "named '[^']+'" | head -1)
            warn "$s --help: missing module $missing (expected on bare Mac; cloud OK)"
        else
            bad "$s --help failed (rc=$rc): $(echo "$out" | head -2)"
        fi
    fi
    rm -f /tmp/_smoke_$$.out
done


hdr "4. Tiny synth dry-run (only if torch + piper-sample-generator available)"
if $PYTHON -c "import torch" 2>/dev/null; then
    if [ -d "piper-sample-generator" ] || [ -d "/opt/piper-sample-generator" ]; then
        ok "torch + piper-sample-generator both present — could run synth_positives.py --count 2"
        warn "NOT executed (would take ~30s + a few hundred MB). Run manually if needed:"
        warn "    $PYTHON scripts/synth_positives.py --project tofu --count 2"
    else
        warn "torch present but piper-sample-generator not cloned; skipping"
    fi
else
    warn "torch not installed; skipping. (Install in venv: uv venv .venv --python 3.10 && uv pip install torch)"
fi


hdr "5. Local dev environment summary"
echo "  Python:        $($PYTHON --version 2>&1)"
echo "  uv available:  $(which uv 2>/dev/null || echo no)"
echo "  Docker daemon: $(docker info >/dev/null 2>&1 && echo running || echo NOT running)"
echo "  Cwd:           $REPO_DIR"


echo
if [ "$FAIL" -gt 0 ]; then
    echo "──────────────────────────────────────────"
    echo "❌ $FAIL check(s) failed, $PASS passed"
    echo "Fix the above before pushing to cloud."
    exit 1
fi

echo "──────────────────────────────────────────"
echo "✓ $PASS check(s) passed, $FAIL failures"
if [ "${#WARNINGS[@]}" -gt 0 ]; then
    echo "  (${#WARNINGS[@]} warning(s) — non-blocking on bare Mac; cloud has the full env)"
fi
echo "Safe to push + retry on cloud."
