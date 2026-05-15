#!/usr/bin/env python3
"""Lambda Labs launcher for the wake-word training pipeline.

Analog to scripts/runpod_train.py but for Lambda Cloud
(https://cloud.lambdalabs.com/api/v1/docs). Differences from the RunPod
launcher captured in LAMBDA_SETUP.md "Differences from the RunPod path".

Flow:
    1. Auth check (LAMBDA_API_KEY).
    2. Upload SSH public key to Lambda (idempotent — skipped if already there).
    3. Launch an instance (default: gpu_1x_a100 in us-east-1).
    4. Poll instance status until "active" + SSH reachable.
    5. SCP scripts/_lambda_setup.sh to the instance.
    6. SSH in, run setup with env vars piped in. Stream stdout locally.
    7. Poll for /workspace/_done file via SSH on a separate channel.
    8. On done OR fatal error: terminate the instance.

Usage:
    source scripts/load_creds.sh
    python scripts/lambda_train.py --project tofu \\
        --hf-repo-id nagisanzeninz/tofu-wakeword-v0

    # cheaper instance:
    python scripts/lambda_train.py --project tofu --instance-type gpu_1x_a10 \\
        --hf-repo-id nagisanzeninz/tofu-wakeword-v0
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests


API_BASE = "https://cloud.lambdalabs.com/api/v1"
DEFAULT_INSTANCE_TYPE = "gpu_1x_a100"
DEFAULT_REGION = "us-east-1"
DEFAULT_SSH_KEY_NAME = "tofu-wake"
DEFAULT_SSH_KEY_PATH = "~/.ssh/id_ed25519"
INSTANCE_PRICING_USD_PER_HR = {
    "gpu_1x_a10": 0.75,
    "gpu_1x_a6000": 0.80,
    "gpu_1x_a100": 1.29,
    "gpu_1x_a100_sxm4": 1.79,
    "gpu_1x_h100_pcie": 2.49,
    "gpu_1x_h100": 3.29,
    "gpu_1x_v100": 0.55,
    "gpu_8x_a100": 14.32,
}
MAX_WAIT_S = 4 * 60 * 60   # 4 h hard cap


# -----------------------------------------------------------------------------
# Lambda Cloud REST helpers (Basic auth: API_KEY as username, no password)
# -----------------------------------------------------------------------------

def _auth_header() -> dict:
    key = os.environ["LAMBDA_API_KEY"]
    token = base64.b64encode(f"{key}:".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def api(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{API_BASE}{path}"
    headers = {**_auth_header(), "Content-Type": "application/json"}
    r = requests.request(method, url, headers=headers, json=body, timeout=60)
    if not r.ok:
        raise RuntimeError(f"Lambda {method} {path} HTTP {r.status_code}: {r.text[:400]}")
    return r.json() if r.text else {}


# -----------------------------------------------------------------------------
# SSH key management
# -----------------------------------------------------------------------------

def ensure_ssh_key_uploaded(name: str, public_key_path: Path) -> str:
    """Idempotently upload our public key to Lambda. Returns the SSH key name."""
    pub = public_key_path.expanduser().read_text().strip()
    existing = api("GET", "/ssh-keys").get("data", [])
    for k in existing:
        if k.get("name") == name:
            existing_key = (k.get("public_key") or "").strip().split()[:2]
            new_key = pub.split()[:2]
            if existing_key == new_key:
                return name
            print(f"  ! SSH key name '{name}' exists but differs locally — using existing", file=sys.stderr)
            return name
    print(f"  uploading SSH key '{name}' to Lambda")
    api("POST", "/ssh-keys", {"name": name, "public_key": pub})
    return name


# -----------------------------------------------------------------------------
# Instance launch + termination
# -----------------------------------------------------------------------------

def launch_instance(instance_type: str, region: str, ssh_key_name: str,
                    name: str) -> str:
    body = {
        "region_name": region,
        "instance_type_name": instance_type,
        "ssh_key_names": [ssh_key_name],
        "name": name,
    }
    resp = api("POST", "/instance-operations/launch", body)
    ids = resp.get("data", {}).get("instance_ids") or []
    if not ids:
        raise RuntimeError(f"launch returned no instance_ids: {resp}")
    return ids[0]


def get_instance(instance_id: str) -> dict:
    return api("GET", f"/instances/{instance_id}").get("data", {})


def terminate_instance(instance_id: str) -> None:
    api("POST", "/instance-operations/terminate",
        {"instance_ids": [instance_id]})


def wait_for_active(instance_id: str, max_wait_s: int = 600) -> dict:
    """Poll until instance status == 'active' (i.e. SSH may now be reachable)."""
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        info = get_instance(instance_id)
        status = info.get("status", "?")
        ip = info.get("ip", "")
        elapsed = int(time.time() - t0)
        print(f"  [{elapsed}s] status={status} ip={ip}")
        if status == "active" and ip:
            return info
        if status in ("terminated", "failed"):
            raise RuntimeError(f"instance entered terminal state: {status}")
        time.sleep(10)
    raise RuntimeError(f"instance not active after {max_wait_s}s")


# -----------------------------------------------------------------------------
# SSH command runner
# -----------------------------------------------------------------------------

def ssh_run(ip: str, ssh_key: Path, remote_cmd: str, env: dict | None = None,
            capture: bool = False, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a remote bash command, optionally with env piped in."""
    env_prelude = ""
    if env:
        # Inject env safely — single-quote-escape values
        parts = []
        for k, v in env.items():
            esc = str(v).replace("'", r"'\''")
            parts.append(f"export {k}='{esc}'")
        env_prelude = "\n".join(parts) + "\n"

    full = env_prelude + remote_cmd
    cmd = [
        "ssh",
        "-i", str(ssh_key.expanduser()),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30",
        f"ubuntu@{ip}",
        "bash -s",
    ]
    return subprocess.run(
        cmd, input=full, text=True,
        capture_output=capture, timeout=timeout,
    )


def wait_for_ssh(ip: str, ssh_key: Path, max_wait_s: int = 240) -> None:
    """Poll until SSH accepts connections + we can run `true`."""
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        r = ssh_run(ip, ssh_key, "true", capture=True, timeout=20)
        if r.returncode == 0:
            return
        elapsed = int(time.time() - t0)
        print(f"  [{elapsed}s] ssh not ready yet (rc={r.returncode})")
        time.sleep(10)
    raise RuntimeError(f"SSH never reachable within {max_wait_s}s on {ip}")


def scp_file(ip: str, ssh_key: Path, local: Path, remote: str) -> None:
    cmd = [
        "scp",
        "-i", str(ssh_key.expanduser()),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        str(local.expanduser()),
        f"ubuntu@{ip}:{remote}",
    ]
    subprocess.run(cmd, check=True)


# -----------------------------------------------------------------------------
# Setup script orchestration
# -----------------------------------------------------------------------------

def run_setup_remote(ip: str, ssh_key: Path, project: str, repo_url: str,
                     branch: str, hf_repo_id: str | None,
                     setup_script_local: Path) -> None:
    """Push the setup script over and run it with env vars piped in."""
    print(f"  scp {setup_script_local.name} → ubuntu@{ip}:/tmp/_lambda_setup.sh")
    # First ensure /tmp/_lambda_setup.sh exists; scp it
    scp_file(ip, ssh_key, setup_script_local, "/tmp/_lambda_setup.sh")

    env = {
        "PROJECT": project,
        "REPO_URL": repo_url,
        "BRANCH": branch,
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "GH_TOKEN": os.environ.get("GH_TOKEN", ""),
    }
    if hf_repo_id:
        env["HF_REPO_ID"] = hf_repo_id

    # Run setup in background, stream stdout to /workspace/setup.log on remote;
    # local SSH captures the same stream.
    remote_cmd = (
        "sudo mkdir -p /workspace && "
        "sudo chown ubuntu:ubuntu /workspace && "
        "chmod +x /tmp/_lambda_setup.sh && "
        "bash /tmp/_lambda_setup.sh 2>&1 | tee /workspace/setup.log"
    )

    print(f"  starting setup remotely (live stream below)\n")
    print("=" * 60)
    # Note: this blocks until the remote script exits. SSH connection keeps
    # the channel open via ServerAliveInterval=30.
    r = ssh_run(ip, ssh_key, remote_cmd, env=env)
    print("=" * 60)
    if r.returncode != 0:
        raise RuntimeError(f"remote setup exited rc={r.returncode}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--repo-url", default="https://github.com/temm1e-labs/customWakeWord.git")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--hf-repo-id", default=None,
                    help="If set, the remote pipeline uploads to HF Hub.")
    ap.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE,
                    choices=sorted(INSTANCE_PRICING_USD_PER_HR.keys()))
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--ssh-key", default=DEFAULT_SSH_KEY_PATH,
                    help="Local path to private SSH key. Default %(default)s")
    ap.add_argument("--ssh-key-name", default=DEFAULT_SSH_KEY_NAME,
                    help="Name to register the public key under on Lambda. "
                         "Idempotent — re-uploads only if name not present.")
    ap.add_argument("--upload-ssh-key", action="store_true",
                    help="Force re-upload of public key even if name exists.")
    ap.add_argument("--keep-instance", action="store_true",
                    help="Do not terminate on completion (manual inspection).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned launch payload + remote script; don't execute.")
    args = ap.parse_args()

    rate = INSTANCE_PRICING_USD_PER_HR.get(args.instance_type, "?")
    print(f"Instance: {args.instance_type} @ ${rate}/hr  region={args.region}")
    print(f"Project:  {args.project}")
    if args.hf_repo_id:
        print(f"HF repo:  {args.hf_repo_id}")

    ssh_key = Path(args.ssh_key)
    pub_key = Path(str(ssh_key) + ".pub")
    if not pub_key.expanduser().exists():
        print(f"ERROR: SSH public key not found at {pub_key}", file=sys.stderr)
        print("       Generate with: ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ''", file=sys.stderr)
        return 2

    setup_script = Path(__file__).parent / "_lambda_setup.sh"
    if not setup_script.exists():
        print(f"ERROR: {setup_script} missing", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"\n=== DRY RUN ===")
        print(f"would launch: instance_type={args.instance_type} region={args.region}")
        print(f"would scp {setup_script}")
        print(f"would SSH-run the script with PROJECT={args.project} HF_REPO_ID={args.hf_repo_id}")
        print(f"\n--- _lambda_setup.sh preview (first 40 lines) ---")
        for line in setup_script.read_text().splitlines()[:40]:
            print(line)
        print("...")
        return 0

    if "LAMBDA_API_KEY" not in os.environ:
        print("ERROR: LAMBDA_API_KEY not set. See LAMBDA_SETUP.md Phase B.", file=sys.stderr)
        return 2

    # 1. SSH key
    if args.upload_ssh_key:
        print(f"  force-uploading SSH key '{args.ssh_key_name}'")
        api("POST", "/ssh-keys",
            {"name": args.ssh_key_name, "public_key": pub_key.expanduser().read_text().strip()})
    else:
        ensure_ssh_key_uploaded(args.ssh_key_name, pub_key)

    # 2. Launch
    print(f"\n=== launching instance ===")
    name = f"customwake-{args.project}-{int(time.time())}"
    instance_id = launch_instance(args.instance_type, args.region,
                                  args.ssh_key_name, name)
    print(f"  instance id: {instance_id}")

    try:
        # 3. Wait for active
        print(f"\n=== waiting for active state ===")
        info = wait_for_active(instance_id)
        ip = info["ip"]
        print(f"  IP: {ip}")

        # 4. Wait for SSH
        print(f"\n=== waiting for SSH ===")
        wait_for_ssh(ip, ssh_key)

        # 5. Push + run setup
        print(f"\n=== running pipeline on instance ===")
        run_setup_remote(ip, ssh_key, args.project, args.repo_url,
                         args.branch, args.hf_repo_id, setup_script)

        print(f"\n=== PIPELINE COMPLETE ===")
        return 0
    finally:
        if not args.keep_instance:
            try:
                print(f"\n  terminating instance {instance_id}")
                terminate_instance(instance_id)
            except Exception as e:
                print(f"  terminate failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
