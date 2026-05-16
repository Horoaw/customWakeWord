#!/usr/bin/env python3
"""RunPod launcher for the upstream microWakeWord training pipeline.

Mirrors TemLLM's H100 launcher pattern: provisions a SECURE pod, runs a
log server on :8001, drives the end-to-end pipeline, stops the pod on
completion.

The pod boots with our pre-baked image (ghcr.io/temm1e-labs/customwake-deps),
so all heavy installs (apt, python3.10, venv, pip install -r requirements.txt,
piper-sample-generator clone + libritts model) are already done at image-build
time on GitHub Actions. The pod entrypoint only does work that depends on
runtime state: cloning our private repo, running the data + training pipeline.

This bypasses the recurring Errno 5 disk I/O failure that hit RunPod A100 SXM
pods during pip's "Installing collected packages" phase (see LESSONS_v0.md #15).

Pipeline (all run on the pod after image boot):
  1. Clone our repo (private, via x-access-token)
  2. Symlink /opt/piper-sample-generator -> /workspace/piper-sample-generator
  3. scripts/synth_positives.py (~3-10 min on a 4090)
  4. scripts/synth_hard_negatives.py
  5. scripts/download_hf_negatives.py (~9 GB, ~3 min)
  6. scripts/build_features.py --download-aug-corpora
  7. scripts/train_microwakeword.py
  8. scripts/emit_manifest.py
  9. (Optional) scripts/upload_to_hf.py

Usage:
    source scripts/load_creds.sh
    python scripts/runpod_train.py --project tofu \\
        --hf-repo-id <user>/tofu-wakeword-v0
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import requests


REST = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"

# Default GPU is RTX 4090 SECURE (~$0.69/hr). A40 SECURE (~$0.44/hr) is the
# A-spec fallback if 4090 is unavailable.
GPU_TYPE_ID = "NVIDIA GeForce RTX 4090"
CLOUD_TYPE = "SECURE"
# Pre-baked deps image built on GitHub Actions (see Dockerfile +
# .github/workflows/build-deps-image.yml). The image is public on GHCR so
# RunPod pulls it without auth. Built with python3.10 + CUDA 12.4 + tf 2.16+
# + torch + microwakeword + piper-sample-generator pre-cloned.
# Update the tag here when requirements.txt changes (rebuild via:
#   gh workflow run build-deps-image -f tag=v0.2).
IMAGE = "ghcr.io/temm1e-labs/customwake-deps:v0.1"
HOURLY_RATE = 0.69

MAX_WAIT_S = 4 * 60 * 60   # 4 h hard cap → ~$2.76 worst case


def rest(method: str, path: str, body: dict | None = None) -> dict:
    token = os.environ["RUNPOD_API_KEY"]
    r = requests.request(
        method,
        f"{REST}{path}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body, timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"RunPod {method} {path} HTTP {r.status_code}: {r.text[:300]}")
    if r.status_code == 204:
        return {}
    return r.json()


def gql_pod(pod_id: str) -> dict:
    token = os.environ["RUNPOD_API_KEY"]
    r = requests.post(
        GQL,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"query": """
            query Pod($id: String!) {
                pod(input: { podId: $id }) {
                    id desiredStatus
                    runtime { uptimeInSeconds ports { privatePort publicPort type } }
                }
            }
        """, "variables": {"id": pod_id}},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("data", {}).get("pod") or {}


def fetch_log(pod_id: str, max_lines: int = 30) -> str:
    url = f"https://{pod_id}-8001.proxy.runpod.net/setup.log"
    try:
        r = requests.get(url, timeout=15)
        if r.ok:
            return "\n".join(r.text.splitlines()[-max_lines:])
    except Exception as e:
        return f"<log fetch err: {e}>"
    return f"<log fetch HTTP {r.status_code}>"


def build_train_script(project: str, repo_url: str, branch: str,
                       hf_repo_id: str | None) -> str:
    """Returns the bash entrypoint executed on the pod.

    Layered into a single subshell so all stdout/stderr lands in /workspace/setup.log,
    while a separate :8001 http.server keeps the log readable even if the
    training subshell crashes silently.
    """
    upload_block = ""
    if hf_repo_id:
        upload_block = (
            f"\n    write_stage \"upload_to_hf\"\n"
            f"    echo \"[$(date +%H:%M:%S)] uploading to HF\"\n"
            f"    python scripts/upload_to_hf.py --project {project} \\\n"
            f"        --model models/{project}-wakeword-v0.tflite \\\n"
            f"        --repo-id {hf_repo_id} \\\n"
            f"        --eval-json eval/results/{project}-v0__latest.json \\\n"
            f"        --esphome configs/examples/{project}/manifest.json\n"
        )

    return rf"""
mkdir -p /workspace
cd /workspace
touch /workspace/setup.log

# Stage marker file — updated at every pipeline step. Tiny (<100 bytes),
# served via :8001/STAGE. Survives even if setup.log gets clobbered.
echo "boot" > /workspace/STAGE
echo "$(date +%H:%M:%S)" > /workspace/STAGE_TIME

# Self-restarting log server with low OOM priority — was killed during
# pip's heavy install in v0 (LESSONS_v0.md #8). The `while true` loop
# restarts it if it ever dies; oom_score_adj=-1000 makes it the LAST
# process the kernel picks during memory pressure.
(
    while true; do
        cd /workspace
        # Set very low OOM priority so the http.server is last to be killed.
        echo -1000 > /proc/self/oom_score_adj 2>/dev/null || true
        nice -n 19 python3 -m http.server 8001 2>/dev/null
        echo "[$(date +%H:%M:%S)] log server died, restarting in 2s" >> /workspace/logserver.log
        sleep 2
    done
) >> /workspace/logserver.log 2>&1 &
disown

(
    # -e: abort on error; -x: trace; -o pipefail: abort if any pipe stage fails
    set -exo pipefail

    # Helper: write a 1-line stage marker. Read externally via :8001/STAGE.
    write_stage() {{
        echo "$1" > /workspace/STAGE
        date +"%H:%M:%S" > /workspace/STAGE_TIME
        echo "[$(date +%H:%M:%S)] STAGE: $1"
    }}

    write_stage "image_check"
    # All deps live in /opt/venv (see Dockerfile). Activate up-front so every
    # subsequent python/pip resolves there.
    source /opt/venv/bin/activate
    echo "[$(date +%H:%M:%S)] python: $(python --version) at $(which python)"
    python -c "import tensorflow as tf, torch, microwakeword; \
        print(f'  tf={{tf.__version__}} torch={{torch.__version__}}')"

    write_stage "git_clone"
    # Private repo — auth via GH_TOKEN env var injected into the pod payload.
    # Temporarily disable `set -x` so the token-bearing URL isn't echoed,
    # and pipe through sed to redact any token that does slip into git's own
    # progress output. set -o pipefail propagates failures out of the pipe.
    repo_path=$(echo "{repo_url}" | sed -E 's|^https://github.com/||; s|\.git$||')
    {{ set +x; }} 2>/dev/null
    git clone --branch {branch} \
        "https://x-access-token:${{GH_TOKEN}}@github.com/${{repo_path}}.git" \
        /workspace/customwake 2>&1 | sed -E 's|x-access-token:[^@]*@|x-access-token:***@|g'
    {{ set -x; }} 2>/dev/null
    cd /workspace/customwake

    write_stage "piper_link"
    # piper-sample-generator was pre-cloned in /opt at image-build time; some
    # scripts default to looking for it at /workspace/piper-sample-generator,
    # so symlink rather than re-clone to keep the entrypoint compatible.
    if [ ! -e /workspace/piper-sample-generator ]; then
        ln -s /opt/piper-sample-generator /workspace/piper-sample-generator
    fi

    # 1. Positives
    write_stage "synth_positives"
    if [ ! -s "data/{project}/synth/positives/manifest.jsonl" ]; then
        python scripts/synth_positives.py --project {project} \
            --psg-dir /workspace/piper-sample-generator
    fi

    # 2. Hard negatives
    write_stage "synth_hard_negatives"
    if [ ! -s "data/{project}/synth/hard_negatives/manifest.jsonl" ]; then
        python scripts/synth_hard_negatives.py --project {project} \
            --psg-dir /workspace/piper-sample-generator
    fi

    # 3. Bulk negatives (HF mmap zips)
    write_stage "download_hf_negatives"
    if [ ! -d "data/negative_datasets/speech" ]; then
        python scripts/download_hf_negatives.py --out data/negative_datasets
    fi

    # 4. Features
    write_stage "build_features"
    if [ ! -d "data/{project}/features/training/wakeword_mmap" ]; then
        python scripts/build_features.py --project {project} \
            --download-aug-corpora
    fi

    # 5. Train + INT8 export
    write_stage "train"
    python scripts/train_microwakeword.py --project {project} \
        --training-config configs/examples/{project}/training_parameters.yaml

    # 6. Eval against held-out tasks (if seeded)
    write_stage "eval"
    if [ -d "eval/tasks/{project}" ]; then
        python -m eval.runner --project {project} \
            --model models/{project}-wakeword-v0.tflite \
            --out eval/results/{project}-v0__$(date +%s).json
        cp $(ls -t eval/results/{project}-v0__*.json | head -1) \
           eval/results/{project}-v0__latest.json
    fi

    # 7. Emit manifest
    write_stage "emit_manifest"
    python scripts/emit_manifest.py --project {project}
    {upload_block}
    write_stage "done"
    echo "[$(date +%H:%M:%S)] DONE"
    touch /workspace/_done
) > /workspace/setup.log 2>&1 &

exec tail -f /dev/null
"""


def build_payload(project: str, repo_url: str, branch: str, hf_repo_id: str | None,
                  gpu_type_id: str = GPU_TYPE_ID,
                  cloud_type: str = CLOUD_TYPE) -> dict:
    return {
        "name": f"customwake-{project}",
        "imageName": IMAGE,
        "gpuTypeIds": [gpu_type_id],
        "gpuCount": 1,
        "vcpuCount": 8,
        "containerDiskInGb": 50,
        "volumeInGb": 50,
        "ports": ["8001/http"],
        "interruptible": False,
        "cloudType": cloud_type,
        "supportPublicIp": True,
        "env": {
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "HUGGING_FACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
            "GH_TOKEN": os.environ.get("GH_TOKEN", ""),
            "WAKE_PROJECT": project,
        },
        "dockerStartCmd": ["bash", "-lc", build_train_script(project, repo_url, branch, hf_repo_id)],
    }


def wait_for_done(pod_id: str) -> bool:
    """Poll until /workspace/_done exists or MAX_WAIT_S hits."""
    t0 = time.time()
    last_log_dump = 0
    while time.time() - t0 < MAX_WAIT_S:
        elapsed = int(time.time() - t0)
        try:
            pod = gql_pod(pod_id)
        except Exception as e:
            print(f"  [{elapsed}s] poll err: {e}", file=sys.stderr, flush=True)
            time.sleep(30)
            continue
        runtime_obj = pod.get("runtime")
        status = pod.get("desiredStatus", "?")
        uptime = runtime_obj.get("uptimeInSeconds", 0) if runtime_obj else 0
        print(f"  [{elapsed}s] status={status} uptime={uptime}s", flush=True)

        try:
            r = requests.get(f"https://{pod_id}-8001.proxy.runpod.net/_done", timeout=10)
            if r.status_code == 200:
                print("  marker /_done present — pipeline completed.", flush=True)
                return True
        except Exception:
            pass

        if elapsed - last_log_dump >= 180 and runtime_obj is not None:
            print("  --- setup.log tail ---", flush=True)
            print(fetch_log(pod_id, max_lines=15), flush=True)
            print("  --- end tail ---", flush=True)
            last_log_dump = elapsed

        # Observed: a healthy host boots within 3-4 min on a cached image. Lower
        # hosts (≤41GB mem on 4090) repeatedly hang at uptime=0. Trip the STUCK
        # detector at 7 min so retry_launch.sh can cycle to the next pool faster.
        # ~$0.08 wasted per stuck attempt at 4090 rates.
        if elapsed >= 7 * 60 and runtime_obj is None:
            raise RuntimeError("STUCK: pod runtime null after 7 min — likely bad host; retry")

        time.sleep(60)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--repo-url", default="https://github.com/temm1e-labs/customWakeWord.git")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--hf-repo-id", default=None,
                    help="If set, upload artefact + manifest to HF Hub.")
    ap.add_argument("--gpu", default=GPU_TYPE_ID,
                    help="RunPod GPU type ID. Default: %(default)s")
    ap.add_argument("--cloud-type", default=CLOUD_TYPE, choices=["SECURE", "COMMUNITY"],
                    help="RunPod cloud type. COMMUNITY is cheaper but spot-interruptible.")
    ap.add_argument("--hourly-rate", type=float, default=HOURLY_RATE,
                    help="Hourly rate of the chosen GPU (for cost cap display).")
    ap.add_argument("--keep-pod", action="store_true",
                    help="Do not stop the pod on completion (for manual inspection).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the would-be payload + entrypoint, don't launch.")
    args = ap.parse_args()

    payload = build_payload(args.project, args.repo_url, args.branch,
                            args.hf_repo_id, gpu_type_id=args.gpu,
                            cloud_type=args.cloud_type)

    print(f"GPU:    {args.gpu} {args.cloud_type}", flush=True)
    print(f"Image:  {IMAGE}", flush=True)
    print(f"Rate:   ${args.hourly_rate}/hr (cap ~${args.hourly_rate * MAX_WAIT_S/3600:.2f})", flush=True)
    print(f"Project: {args.project}", flush=True)
    if args.hf_repo_id:
        print(f"HF repo: {args.hf_repo_id}", flush=True)
    print()

    if args.dry_run:
        print("=== DRY RUN — entrypoint script ===")
        print(payload["dockerStartCmd"][2])
        print("=== END DRY RUN ===")
        return 0

    print("=== launching pod ===", flush=True)
    pod = rest("POST", "/pods", payload)
    pod_id = pod.get("id")
    if not pod_id:
        raise RuntimeError(f"no pod id in response: {pod}")
    print(f"  pod id:     {pod_id}", flush=True)
    print(f"  log server: https://{pod_id}-8001.proxy.runpod.net/setup.log", flush=True)

    try:
        ok = wait_for_done(pod_id)
        if not ok:
            print("\n!! pod did not complete within MAX_WAIT_S; last log:", flush=True)
            print(fetch_log(pod_id, max_lines=50), file=sys.stderr)
            return 2
        print("\n=== PIPELINE COMPLETE ===", flush=True)
        return 0
    finally:
        if not args.keep_pod:
            try:
                rest("POST", f"/pods/{pod_id}/stop")
                print(f"  pod {pod_id} stopped", flush=True)
            except Exception as e:
                print(f"  stop failed: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
