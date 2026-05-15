#!/usr/bin/env python3
"""RunPod launcher for the upstream microWakeWord training pipeline.

Mirrors TemLLM's H100 launcher pattern: provisions a SECURE pod, runs a
log server on :8001, drives the end-to-end pipeline, stops the pod on
completion. The pipeline is:

  1. Install Python 3.10 (pinned to avoid OHF-Voice/micro-wake-word#62)
  2. Clone our repo + install deps
  3. Clone piper-sample-generator + download libritts_r generator
  4. Run scripts/synth_positives.py (~3-10 min on a 4090)
  5. Run scripts/synth_hard_negatives.py
  6. Run scripts/download_hf_negatives.py (~9 GB, ~3 min)
  7. Run scripts/build_features.py --download-aug-corpora
  8. Run scripts/train_microwakeword.py
  9. Run scripts/emit_manifest.py
  10. (Optional) scripts/upload_to_hf.py

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
# Real RunPod pytorch tag (the "2.4.0-py3.10-..." format from 2024 docs is
# obsolete — current catalog uses `1.0.3-cu1290-torch271-ubuntu2204`).
# Ubuntu 22.04 ships system Python 3.10; we install python3.10 explicitly via
# deadsnakes anyway so the base image's Python version doesn't matter.
IMAGE = "runpod/pytorch:1.0.3-cu1290-torch271-ubuntu2204"
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
            f"\necho \"[$(date +%H:%M:%S)] uploading to HF\"\n"
            f"python scripts/upload_to_hf.py --project {project} "
            f"--model models/{project}-wakeword-v0.tflite "
            f"--repo-id {hf_repo_id} "
            f"--eval-json eval/results/{project}-v0__latest.json "
            f"--esphome configs/examples/{project}/manifest.json\n"
        )

    return rf"""
mkdir -p /workspace
cd /workspace
touch /workspace/setup.log

# Log server on :8001 — survives any subshell crash.
(cd /workspace && python3 -m http.server 8001) >> /workspace/logserver.log 2>&1 &

(
    set -ex

    echo "[$(date +%H:%M:%S)] starting pod setup"
    echo "[$(date +%H:%M:%S)] base python: $(python3 --version 2>&1)"

    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq ffmpeg sox libsox-fmt-mp3 unzip wget git \
        software-properties-common build-essential

    # Install Python 3.10 from deadsnakes — microwakeword pins >=3.10,<3.11
    # (see OHF-Voice/micro-wake-word issue #62 for the Python 3.11 break).
    add-apt-repository ppa:deadsnakes/ppa -y
    apt-get update -qq
    apt-get install -y -qq python3.10 python3.10-venv python3.10-dev python3.10-distutils
    echo "[$(date +%H:%M:%S)] python3.10 installed: $(python3.10 --version)"

    git clone --branch {branch} "{repo_url}" /workspace/customwake
    cd /workspace/customwake

    # Fresh venv on Python 3.10 — base image's torch is unused.
    python3.10 -m venv /workspace/.venv
    source /workspace/.venv/bin/activate
    pip install --upgrade pip wheel

    pip install --no-cache-dir -r requirements.txt
    echo "[$(date +%H:%M:%S)] deps installed: $(python --version)"

    # 1. Positives
    if [ ! -s "data/{project}/synth/positives/manifest.jsonl" ]; then
        python scripts/synth_positives.py --project {project} \
            --psg-dir /workspace/piper-sample-generator
    fi

    # 2. Hard negatives
    if [ ! -s "data/{project}/synth/hard_negatives/manifest.jsonl" ]; then
        python scripts/synth_hard_negatives.py --project {project} \
            --psg-dir /workspace/piper-sample-generator
    fi

    # 3. Bulk negatives (HF mmap zips)
    if [ ! -d "data/negative_datasets/speech" ]; then
        python scripts/download_hf_negatives.py --out data/negative_datasets
    fi

    # 4. Features
    if [ ! -d "data/{project}/features/training/wakeword_mmap" ]; then
        python scripts/build_features.py --project {project} \
            --download-aug-corpora
    fi

    # 5. Train + INT8 export
    python scripts/train_microwakeword.py --project {project} \
        --training-config configs/examples/{project}/training_parameters.yaml

    # 6. Eval against held-out tasks (if seeded)
    if [ -d "eval/tasks/{project}" ]; then
        python -m eval.runner --project {project} \
            --model models/{project}-wakeword-v0.tflite \
            --out eval/results/{project}-v0__$(date +%s).json
        cp $(ls -t eval/results/{project}-v0__*.json | head -1) \
           eval/results/{project}-v0__latest.json
    fi

    # 7. Emit manifest
    python scripts/emit_manifest.py --project {project}
    {upload_block}
    echo "[$(date +%H:%M:%S)] DONE"
    touch /workspace/_done
) > /workspace/setup.log 2>&1 &

exec tail -f /dev/null
"""


def build_payload(project: str, repo_url: str, branch: str, hf_repo_id: str | None,
                  gpu_type_id: str = GPU_TYPE_ID) -> dict:
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
        "cloudType": CLOUD_TYPE,
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

        # Pull + boot can take 8-12 min on a fresh GPU. Be generous.
        if elapsed >= 15 * 60 and runtime_obj is None:
            raise RuntimeError("STUCK: pod runtime null after 15 min — likely image-pull failure or no GPU capacity")

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
    ap.add_argument("--keep-pod", action="store_true",
                    help="Do not stop the pod on completion (for manual inspection).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the would-be payload + entrypoint, don't launch.")
    args = ap.parse_args()

    payload = build_payload(args.project, args.repo_url, args.branch,
                            args.hf_repo_id, gpu_type_id=args.gpu)

    print(f"GPU:    {args.gpu} {CLOUD_TYPE}", flush=True)
    print(f"Image:  {IMAGE}", flush=True)
    print(f"Rate:   ${HOURLY_RATE}/hr (cap ~${HOURLY_RATE * MAX_WAIT_S/3600:.2f})", flush=True)
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
