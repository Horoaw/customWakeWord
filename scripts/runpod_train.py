#!/usr/bin/env python3
"""RunPod A40/4090 launcher for microWakeWord training.

Mirrors TemLLM's H100 launcher pattern: provisions a SECURE pod, runs a
tiny http.server on :8001 so /workspace/setup.log can be tailed via the
RunPod proxy, runs build_features → train → export → upload, and stops
the pod on success.

Usage:
    source scripts/load_creds.sh
    python scripts/runpod_train.py --project tofu \\
        --repo-url https://github.com/temm1e-labs/customWakeWord.git \\
        --hf-repo-id <you>/tofu-wakeword-v0
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import requests


REST = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"

GPU_TYPE_ID = "NVIDIA RTX A40"
CLOUD_TYPE = "SECURE"
IMAGE = "runpod/tensorflow:2.15.0-py3.11-cuda12.2.0-devel-ubuntu22.04"
HOURLY_RATE = 0.39  # ~A40 SECURE rate

MAX_WAIT_S = 4 * 60 * 60  # cap at 4 h


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
    upload_step = ""
    if hf_repo_id:
        upload_step = (
            f"\npython scripts/upload_to_hf.py --project {project} "
            f"--model models/{project}-wakeword-v0.tflite "
            f"--repo-id {hf_repo_id} "
            f"--eval-json eval/results/{project}-v0__latest.json\n"
        )

    return rf"""
mkdir -p /workspace
cd /workspace
touch /workspace/setup.log

# Log server on :8001 — survives any training crash, served via runpod proxy.
(cd /workspace && python3 -m http.server 8001) >> /workspace/logserver.log 2>&1 &

(
    set -x
    echo "[$(date +%H:%M:%S)] starting pod setup"
    apt-get update -qq && apt-get install -y -qq ffmpeg sox libsox-fmt-mp3 unzip
    git clone --branch {branch} "{repo_url}" /workspace/customwake
    cd /workspace/customwake

    pip install --no-cache-dir -r requirements.txt
    echo "[$(date +%H:%M:%S)] deps installed"

    # 1. Build features (assumes synth + bulk audio already prepared on a
    #    separate CPU pod and rsynced to this volume; if not, do it here.)
    if [ ! -f "data/{project}/clean/train.tfrecord" ]; then
        echo "[$(date +%H:%M:%S)] building features"
        python scripts/build_features.py --project {project}
    fi

    # 2. Train + export
    echo "[$(date +%H:%M:%S)] starting training"
    python scripts/train_microwakeword.py \
        --project {project} \
        --config configs/train.yaml \
        --out outputs/{project}/ \
        --export models/{project}-wakeword-v0.tflite

    # 3. Eval on held-out test set
    echo "[$(date +%H:%M:%S)] eval"
    python -m eval.runner --project {project} \
        --model models/{project}-wakeword-v0.tflite \
        --out eval/results/{project}-v0__$(date +%s).json
    cp $(ls -t eval/results/{project}-v0__*.json | head -1) \
       eval/results/{project}-v0__latest.json
    {upload_step}
    echo "[$(date +%H:%M:%S)] DONE"
    touch /workspace/_done
) > /workspace/setup.log 2>&1 &

exec tail -f /dev/null
"""


def build_payload(project: str, repo_url: str, branch: str, hf_repo_id: str | None) -> dict:
    return {
        "name": f"customwake-{project}",
        "imageName": IMAGE,
        "gpuTypeIds": [GPU_TYPE_ID],
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
    """Poll until /workspace/_done exists or we time out / hit a stuck state."""
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
        print(f"  [{elapsed}s] status={status} uptime={runtime_obj.get('uptimeInSeconds', 0) if runtime_obj else 0}", flush=True)

        # Probe for completion marker.
        try:
            r = requests.get(f"https://{pod_id}-8001.proxy.runpod.net/_done", timeout=10)
            if r.status_code == 200:
                print("  marker /_done present — training completed.", flush=True)
                return True
        except Exception:
            pass

        if elapsed - last_log_dump >= 180 and runtime_obj is not None:
            print("  --- setup.log tail ---", flush=True)
            print(fetch_log(pod_id, max_lines=15), flush=True)
            print("  --- end tail ---", flush=True)
            last_log_dump = elapsed

        if elapsed >= 10 * 60 and runtime_obj is None:
            raise RuntimeError("STUCK: pod runtime null after 10 min")

        time.sleep(60)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--repo-url", default="https://github.com/temm1e-labs/customWakeWord.git")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--hf-repo-id", default=None,
                    help="If set, upload the trained .tflite to HuggingFace Hub.")
    ap.add_argument("--keep-pod", action="store_true",
                    help="Do not stop the pod on completion (for manual inspection).")
    args = ap.parse_args()

    print(f"GPU:    {GPU_TYPE_ID} {CLOUD_TYPE}", flush=True)
    print(f"Image:  {IMAGE}", flush=True)
    print(f"Rate:   ${HOURLY_RATE}/hr (cap ~${HOURLY_RATE * MAX_WAIT_S/3600:.2f})", flush=True)
    print(f"Project: {args.project}", flush=True)
    print()

    payload = build_payload(args.project, args.repo_url, args.branch, args.hf_repo_id)
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
        print("\n=== TRAINING COMPLETE ===", flush=True)
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
