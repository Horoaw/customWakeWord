# LESSONS — v0 training session (2026-05-15)

The actual landmines hit during the first end-to-end run of `tofu` on RunPod. Each entry: **symptom → root cause → fix to apply for v0.1**.

This is the "knowhow" companion to [`TRAINING_PLAN.md`](TRAINING_PLAN.md). The plan describes the happy path; this doc enumerates everything that went sideways.

---

## 1. Docker image tag — `2.4.0-py3.10-...` is dead

**Symptom**: pod created with `desiredStatus=RUNNING`, `uptime=0s` for 10+ minutes. Container never starts. Log server unreachable. Pod silently bills.

**Root cause**: the `runpod/pytorch:2.4.0-py3.10-cuda12.4.1-devel-ubuntu22.04` tag from older RunPod docs/blogs doesn't exist in the current registry. RunPod's image catalog naming changed to `1.0.3-cuXXXX-torchXXX-ubuntu22XX` format (no Python in the tag — Ubuntu 22.04 ships with Python 3.10 by default; Ubuntu 24.04 ships 3.12).

**Fix applied** (commit `74bb070`): use `runpod/pytorch:1.0.3-cu1290-torch271-ubuntu2204`. Verify any new tag against Docker Hub before pinning:
```bash
curl -s 'https://hub.docker.com/v2/repositories/runpod/pytorch/tags/?page_size=20' \
    | jq -r '.results[].name' | head
```

**v0.1 patch**: pin image via a SemVer-aliased tag if RunPod ever publishes one, or sanity-check tag existence at launcher startup before sending the create-pod request.

---

## 2. Private repo clone fails without explicit auth

**Symptom**: container boots, deps install, then `fatal: could not read Username for 'https://github.com': No such device or address`. Subshell aborts, pod idles eating $0.69/hr until manually stopped.

**Root cause**: `git clone https://github.com/temm1e-labs/customWakeWord.git` defaults to unauthenticated HTTPS. Public repos work; private repos prompt for credentials and bail in non-interactive mode.

**Fix applied** (commit `2095af8`): use `x-access-token` pattern:
```bash
git clone --branch main \
    "https://x-access-token:${GH_TOKEN}@github.com/temm1e-labs/customWakeWord.git" \
    /workspace/customwake
```

**Security**: the token in the URL would be echoed by `set -x` into `/workspace/setup.log`, which is publicly served on `:8001`. Disable tracing around the clone and scrub via sed:
```bash
{ set +x; } 2>/dev/null
git clone ... 2>&1 | sed -E 's|x-access-token:[^@]*@|x-access-token:***@|g'
{ set -x; } 2>/dev/null
```

Also need `set -o pipefail` (commit `2095af8` did this) so a failed clone piped through sed still aborts the script.

**v0.1 patch**: also redact GH_TOKEN from `runtime/logserver.log` (currently unredacted). Better: switch to deploy-key SSH auth so no token sits in any log.

---

## 3. Token leak via RunPod pod listing API

**Symptom**: `GET /v1/pods` returns the full pod payload including the literal env values for `HF_TOKEN`, `GH_TOKEN`, `HUGGING_FACE_HUB_TOKEN`. They appear verbatim in the API response (and therefore in any monitoring transcript that captured it).

**Root cause**: RunPod treats pod env vars as inspectable metadata, not secrets. Anyone with the API key (which is single-tenant — not a real risk) can read them, but they leak into logs/transcripts.

**Fix to apply**: rotate tokens after each session — particularly `HF_TOKEN` and `GH_TOKEN`. For v0.1, consider:
- **Short-lived tokens**: GitHub fine-grained PATs scoped to one repo, 7-day expiry
- **Pod-scope HF token**: HF allows creating a token per project — minimum privilege per pod
- **Volume-mounted secrets**: instead of env vars, mount a secret file via RunPod's network volume system (if supported)

---

## 4. Pod runtime stuck at `uptime=0s` — bad host assignment

**Symptom**: Same image, same payload — sometimes the container boots in 3-4 min (uptime increments cleanly), sometimes uptime stays 0 for the full launcher timeout (15 min in the original config). Logged 3 separate occurrences of this in one afternoon.

**Root cause**: RunPod's scheduler assigns pods to physical hosts that may be in a degraded state. Observed correlation: **low-memory hosts (≤41 GB system RAM on 4090 SECURE) consistently fail to boot, high-memory hosts (~100 GB) boot cleanly**. Possibly the image pull stalls on disk-constrained hosts. No way to predict from outside.

**Fix applied** (commit `26cce49`): trip the STUCK detector at 7 min (was 15) so retry_launch cycles to a different host faster. Each bad-host attempt now costs ~$0.08 instead of ~$0.17.

**v0.1 patch**: query the pod's machine specs via GraphQL right after creation — if `machine.memoryInGb < 60`, eagerly terminate and retry. No way to do this *before* pod creation (RunPod doesn't expose target host specs in the create-pod request).

---

## 5. Capacity droughts — every GPU type denied

**Symptom**: `POST /v1/pods` returns HTTP 500 with `"error":"create pod: There are no instances currently available"`. Reproduced across 4090 SECURE, 4090 COMMUNITY, A40 SECURE, L40, L40S, 3090 — all denied within 30 seconds of each other.

**Root cause**: RunPod is provider-of-last-resort for many users; capacity flexes with demand. Mid-day workday hours see the worst droughts.

**Fix applied** (commit `a0f1b32`): `scripts/retry_launch.sh` rotates through 6 GPU/cloud combinations every 60s. As soon as one accepts, fires the full pipeline.

**Observation**: **A100 SXM 80GB** ($1.39/hr) had capacity when all consumer GPUs were dry. The reverse hierarchy — go *up* the price tier during droughts; fewer takers means more availability.

**v0.1 patch**: add a dedicated `--capacity-probe` mode that checks GPU availability via cheap throwaway pod creations before committing to the full pipeline. Currently retry_launch.sh blocks on the actual training, which means cycle time is bounded by STUCK timeout (7 min) per bad attempt.

---

## 6. Orphan pods — TaskStop on launcher leaks the pod

**Symptom**: After `TaskStop` on a stuck `runpod_train.py`, the pod it created keeps running, silently billing. Found one orphan that had been alive 24 min at $0.69/hr (~$0.28 wasted).

**Root cause**: `runpod_train.py` has a `try/finally` that stops its created pod when the script exits normally or via `RuntimeError`. But `TaskStop` (SIGTERM/SIGKILL) bypasses the `finally` block — the pod is never told to stop.

**Fix applied** (this session, manual): periodic `runpod list-pods` + manual termination of orphans.

**v0.1 patch**: register a `signal.SIGTERM` handler in `runpod_train.py` that stops + deletes the pod before exiting. Also write the pod_id to a known file (e.g., `/tmp/runpod_train_<project>.pid`) so a separate `runpod_cleanup.py` script can audit at session start.

---

## 7. Stop ≠ Terminate — stopped pods still cost storage

**Symptom**: After 5+ failed boots, the RunPod balance dropped faster than expected. Listed pods showed 7 EXITED entries each with 50GB containerDisk + 50GB volumeInGb.

**Root cause**: `POST /pods/{id}/stop` pauses the pod but retains its disk and volume — RunPod bills for that at ~$0.05/GB/month, but more importantly the per-hour cost field continues to display as if running, possibly inflating the `currentSpendPerHr` metric. Only `DELETE /pods/{id}` (HTTP 204) fully releases the storage.

**Fix applied** (this session): `for pid in $ORPHANS; do curl -X DELETE /v1/pods/$pid; done`.

**v0.1 patch**: in `runpod_train.py`'s cleanup path, switch from `/stop` to `DELETE /pods/{id}`. Stop only makes sense if you're going to resume the same pod — for training pipelines that always start fresh, terminate is correct.

---

## 8. Log server is a single point of failure

**Symptom**: ~45 minutes into a successful pipeline run, the log server proxy returns 502 across all paths. Pod is still RUNNING (uptime increasing), but we're blind. HF poller is the only completion signal left.

**Root cause**: `python3 -m http.server 8001 &` is launched as a one-shot in the entrypoint. No supervisor, no restart logic. When the heavy pip install (PyTorch + TF + CUDA libs, ~5 GB peak RAM) triggered the kernel OOM killer, the http.server was picked (low-priority, easy to kill) and stayed dead for the rest of the run.

**Fix to apply for v0.1**:

```bash
# Self-restarting log server with low OOM priority
( while true; do
    cd /workspace
    # Make http.server the LAST candidate the OOM killer picks
    echo -1000 > /proc/self/oom_score_adj 2>/dev/null || true
    nice -n 19 python3 -m http.server 8001 2>/dev/null
    sleep 2
  done ) >> /workspace/logserver.log 2>&1 &
disown
```

Plus: write a **stage marker file** (`/workspace/STAGE`) at each pipeline step that's served via the same log server. Even if the log gets cluttered, the user can `curl .../STAGE` and see one-line progress.

**v0.1 patch**: also add a sidecar HF Hub uploader running in parallel — every 5 min, upload the current state of `/workspace/setup.log` to the HF repo as `progress.log`. That gives us an out-of-band view of the log even if the in-pod server dies.

---

## 9. pip install is 5× slower than benchmarks suggested

**Symptom**: requirements.txt install took 40 minutes on A100, not the 5-8 min the research estimated.

**Root cause**: the dep tree is enormous — 144 packages including TensorFlow (1.5 GB), PyTorch (2 GB), microwakeword's transitive deps (audiomentations, librosa, scipy, scikit-learn, all NVIDIA CUDA libs separately, etc.). Total download was ~6 GB. RunPod egress bandwidth is shared.

**Fix to apply for v0.1**: build a custom Docker image with all deps pre-installed, push it to GitHub Container Registry (free for public images), and use that as the pod's `imageName`. Should drop pip install from 40 min → ~30 sec (just `pip install -e .` for the repo itself).

Rough recipe:
```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev \
    ffmpeg sox libsox-fmt-mp3 unzip wget git
WORKDIR /workspace
COPY requirements.txt .
RUN python3.10 -m venv /workspace/.venv && \
    . /workspace/.venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt
```

Build + push to `ghcr.io/temm1e-labs/customwake-deps:v0.1`, swap the pod image. Saves ~$1 per training run (40 min × $1.49/hr = $0.99 of pure pip install on A100).

---

## 10. Pricing field is misleading

**Symptom**: `currentSpendPerHr=$2.27` while only one $1.49/hr pod was running. After deleting 7 EXITED pods, the rate dropped to $1.50.

**Root cause**: RunPod's `currentSpendPerHr` GraphQL field includes accrued storage charges from EXITED pods, not just the live billing rate. Misleading metric to gauge "am I burning money fast".

**Fix to apply for v0.1**: compute spend rate yourself by summing `costPerHr` across pods with `desiredStatus == 'RUNNING'`. That's the real burn.

---

## 11. SSH access for debugging requires PUBLIC_KEY env var upfront

**Symptom**: When log server died mid-pipeline, no way to SSH into the pod to investigate. RunPod exposed port 19123 → 60009 for SSH, but pod was created with `PUBLIC_KEY=""`.

**Fix to apply for v0.1**: always set `PUBLIC_KEY` in the pod env from `~/.ssh/id_ed25519.pub` (or similar) so SSH-based recovery is available. Cheap insurance.

```python
env["PUBLIC_KEY"] = Path("~/.ssh/id_ed25519.pub").expanduser().read_text().strip()
```

---

## 12. Retry script doesn't track or surface pod creations

**Symptom**: When `retry_launch.sh` succeeded on attempt 1 (after 3 hours of capacity droughts), the launched pod's pod_id was hidden inside `$(python3 ... 2>&1)` — output not flushed until process exits. I had to query the RunPod API to find the new pod.

**Fix to apply for v0.1**: don't capture launcher output with `$(...)` — let it stream to stdout. Use `tee` to dual-write to a logfile for retrospective inspection:
```bash
python3 scripts/runpod_train.py ... 2>&1 | tee -a /tmp/launcher.log
```
That way `pod id:` lines appear in real time, and retry_launch can pattern-match them.

---

## 13. GPU host class matters more than GPU compute for this workload

**Observation (added 2026-05-15, second session)**: same image, same entrypoint, same pip install — wildly different speed across hosts.

| GPU host | pip install wall | HF 9GB download wall | Notes |
|---|---|---|---|
| 4090 SECURE (consumer DC, 41-62 GB host mem) | ~40 min | >2 h (didn't complete) | Choked on HF Hub egress |
| A100 SXM 80GB (server class, datacenter mem) | ~40 min | partial — couldn't measure due to log-OOM | Faster network than 4090, killed by separate issue |
| A40 SECURE 48GB (server class, 50 GB host mem) | **~15 min** (2.5× faster) | TBD this run | Server-class network |

**Root cause**: RunPod's consumer-GPU pods (4090, 3090) live in budget datacenters with shared 1-2 Gbps NICs and no peering to HF Hub's CDN. Server-class GPUs (A40, A100, L40, H100) live in tier-1 datacenters with 10-25 Gbps NICs and direct or near-direct peering to major content networks. For this pipeline, **network bandwidth is the dominant cost** (40 min pip + 60+ min HF download), so paying $0.44/hr for an A40 beats $0.69/hr for a 4090 on total cost.

**Fix applied** (commit `679bf7c`): updated `scripts/retry_launch.sh` GPU ordering to prefer A100 → L40S → L40 → A40 ahead of 4090/3090. Consumer GPUs are now fallbacks, not defaults.

**v0.1 patch**: bake this learning into `configs/preferred_provider.txt` + `runpod_train.py --gpu-tier server` flag so default behavior is server-class.

---

## Compounding cost across this session

| Failure mode | Wasted $ | Count |
|---|---|---|
| Bad image tag → image pull stuck (10 min × $0.69/hr) | $0.12 | 1 |
| GH auth fail → idle until manual stop (12 min × $0.69/hr) | $0.14 | 1 |
| Bad-host boot stuck (5-12 min × $0.69-$1.49/hr) | $0.42 | 3 |
| Orphan pod after TaskStop (24 min × $0.69/hr) | $0.28 | 1 |
| Capacity probes via throwaway pods (~10s each, capacity-denied = free) | $0.00 | ~12 |
| Successful pod, blind via 502 (in progress, billed for actual training) | $1.50 | 1 |
| **Total wasted on failures** | **~$0.96** | |
| **Total budget burned** | **~$2.00** | of $20.87 starting balance |

Wasted $ is small in absolute terms but each failure cost ~10-30 minutes of wall time. The cumulative wall-time waste was ~3 hours of attempts vs the 30-45 min the happy path should have taken.

---

## Priority for v0.1 hardening

In rough cost-saving + reliability order:

1. **Custom Docker image with pre-installed deps** (#9) — saves ~$1/run + 40 min wall
2. **Self-restarting log server** (#8) — restores visibility under memory pressure
3. **Stage marker file system** (#8b) — completion signal independent of log server
4. **DELETE not STOP on cleanup** (#7) — releases storage immediately
5. **SIGTERM handler for runpod_train.py** (#6) — kill the pod when launcher is killed
6. **PUBLIC_KEY auto-set** (#11) — SSH recovery available
7. **Capacity probe mode** (#5) — bound cycle time during droughts
8. **Stream launcher output through `tee`** (#12) — observability under retry loop
9. **Memory-spec gating** (#4) — reject low-mem hosts before paying

None of these are blockers for v0 finishing this run — they're for the next training cycle.

---

# Round 2 (2026-05-16/17) — the pre-baked image session

The v0 patches above (custom image, log server, stage markers, DELETE-not-STOP) all landed and worked. New failures emerged from finally exercising the rest of the pipeline.

## 14. Pre-baked image stops the pod-disk Errno 5 — but only for pip

**Symptom (v0)**: `pip install` 5 GB of wheels on RunPod A100 SXM crashes 3/3 times with `OSError [Errno 5] Input/output error` during "Installing collected packages".

**Symptom (v0.1+)**: With deps pre-baked into `ghcr.io/temm1e-labs/customwake-deps`, the install never runs on the pod and never fails. BUT — same `Errno 5` reappears during `synth_positives` when piper writes 10 000 WAVs in a tight batch loop. Same disk, different write pattern. Some WAVs are pre-allocated to their full size on disk but never have their data flushed; `Path.stat().st_size` reports the expected ~50 KB, but reading the bytes fails. We're seeing ~40% positive loss on bad-disk hosts vs ~0% on healthy ones.

**Hypothesis (still untested)**: RunPod's container disk on some hosts is on a flaky NFS / loopback / sparse-file backing that handles "create + size announce" fine but loses actual writes under sustained pressure. Volume disks (mounted at `/workspace`) are more reliable.

**Fix applied** (commit `baf533f`): in `scripts/build_features.py`, validate each WAV by opening with `wave.open()` before adding to the symlink farm, so genuinely broken files don't get fed to the audio decoder. Logs the drop count to stderr.

**v0.2 patch idea**: in `synth_positives.py`, post-batch verify each WAV header before adding to the manifest. Don't rely on the consumer to filter.

---

## 15. GitHub Actions `ubuntu-latest` runs out of disk at the pip extract step

**Symptom**: GHA Docker build crashes ~60% through the pip install with `OSError: [Errno 28] No space left on device`. All wheels download cleanly, then the install phase fails as files are extracted into `/opt/venv`.

**Root cause**: `ubuntu-latest` ships with ~14 GB free post-toolchain. Our deps tree (torch + tf + nvidia-* + microwakeword) needs ~15–20 GB extracted plus BuildKit overlay.

**Fix applied** (commit `8d19abf`): add `jlumbroso/free-disk-space@main` step before Buildx. Frees ~25 GB by removing Android SDK, .NET, Haskell, preinstalled toolchains. ~30 s overhead, gets us to ~40 GB free.

---

## 16. `pip cache purge` errors when chained with `--no-cache-dir`

**Symptom**: `pip install --no-cache-dir ... && pip cache purge` exits 1 with `ERROR: pip cache commands can not function since cache is disabled.` After the **install successfully completed**. The chained command kills the entire Docker layer.

**Fix applied** (commit `ac60875`): just drop the chained `pip cache purge`. `--no-cache-dir` already discards cached wheels per-package. Redundant.

---

## 17. GHCR private packages — they don't inherit from a public source repo

**Symptom**: source repo `temm1e-labs/customWakeWord` is public, GHA pushes the image, image manifest is **still 401-private**. RunPod can't pull anonymously.

**Root cause**: GHCR packages are independent of their source-repo visibility. By default, every new package in an org is **private** until explicitly made public.

**Three-step unlock** (had to be done by the org admin in the UI; not scriptable with a `repo+workflow+admin:org` token):
1. Organization → Settings → Packages → "Container types" → check **Public**. (Default is unchecked; UI says *"Setting is disabled by organization administrators"* on the per-package page until this is flipped.)
2. Package settings → Danger Zone → **Change visibility** → Public → type package name to confirm.
3. (Optional, but recommended:) toggle "Inherit access from source repository" on the package so future re-pushes don't private-revert.

**Detection**: see RUNPOD_RECIPE.md §2 for the `curl` one-liner that confirms anonymous manifest fetch.

**Fallback we built but didn't use**: `POST /v1/containerregistryauth` (RunPod REST) registers GHCR creds server-side, then pod create passes `containerRegistryAuthId`. Works in theory but requires a `read:packages`-scoped PAT which we didn't have. Public is simpler.

---

## 18. PyTorch 2.6+ defaults `torch.load(weights_only=True)` — breaks old pickles

**Symptom**: piper-sample-generator v2.0.0's `generate_samples.py:74` calls `torch.load(model_path)` without args. The libritts checkpoint pickle contains `piper_train.vits.models.SynthesizerTrn`, which is not in PyTorch's safe-globals allowlist for `weights_only=True`. Errors:
```
_pickle.UnpicklingError: Weights only load failed. ...
WeightsUnpickler error: Unsupported global: GLOBAL piper_train.vits.models.SynthesizerTrn
```

**Fix applied** (commit `4dbdd3a`): inline sed-patch in the Dockerfile to replace the `torch.load(model_path)` line with `torch.load(model_path, weights_only=False)`. Builds verify the patch landed via `grep -q weights_only=False`.

**Why we trust the checkpoint**: it's a versioned release artefact from a known-good upstream (rhasspy/piper-sample-generator v2.0.0). We download it from GitHub Releases over HTTPS at image-build time.

---

## 19. Default PyPI torch is built with CUDA 13 — RunPod A100 driver is 12.7

**Symptom**: `synth_positives.py` runs but logs:
```
UserWarning: CUDA initialization: The NVIDIA driver on your system
is too old (found version 12070). Please update your GPU driver...
```
torch silently falls back to CPU. piper-sample-generator runs at ~5% of expected throughput. 10k WAVs would burn 2–3 hours of A100 time.

**Root cause**: `pip install torch` (unversioned) currently pulls `torch==2.12.0+cu130`. CUDA 13 needs driver 13.0+. RunPod A100 SXM hosts ship driver 12.7.

**Fix applied** (commit `89aa224`): pin via the PyTorch cu124 index:
```
--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.5.1+cu124
torchaudio==2.5.1+cu124
```
Bonus: torch 2.5 was the last version where `weights_only` defaulted to `False`, making lesson #18 redundant (but we keep the sed patch as defense in depth).

**v0.2 patch idea**: add a startup check in `runpod_train.py` that queries `torch.cuda.is_available()` and abort early if False. Avoids hours of CPU-only work.

---

## 20. `microwakeword` submodules silently dropped by `pip install`

**Symptom**: `from microwakeword.audio.augmentation import Augmentation` → `ModuleNotFoundError: No module named 'microwakeword.audio'`. Top-level `microwakeword` imports fine.

**Root cause**: upstream's `setup.py` uses `packages=setuptools.find_packages()` which **requires `__init__.py` in each subdir**. The `microwakeword/audio/` and `microwakeword/layers/` directories exist in the repo but have no `__init__.py`. setuptools silently skips them at install time.

**Fix applied** (commit `d15b4ca`): in the Dockerfile, clone microwakeword, `touch` the missing `__init__.py` files, then `pip install -e /opt/microwakeword`. Sanity-check the import in the same RUN layer.

**Upstream PR opportunity**: send a one-line `touch __init__.py` fix to OHF-Voice/micro-wake-word. The `inference.py`, `data.py`, and `model_train_eval.py` modules import from `microwakeword.audio.*`, so this is broken for everyone, not just us.

---

## 21. `datasets==4.x` requires `torchcodec` for audio streaming

**Symptom**: `RIR download failed: To support encoding audio data, please install 'torchcodec'.` Caused by `load_dataset("davidscripka/MIT_environmental_impulse_responses", streaming=True)` in `build_features.py`.

**Fix applied** (commit `d15b4ca`): pin `datasets>=3.0,<4.0` in `requirements.txt`. 3.x uses `soundfile` for audio decode which is already installed.

**v0.2 patch idea**: when 4.x adoption is forced (e.g., for a streaming dataset that requires it), add `torchcodec` to the image. Currently it's not a runtime dep we need elsewhere.

---

## 22. Upstream `Clips` API dropped `filepath_text_files=[...]` for glob-only

**Symptom**: `TypeError: Clips.__init__() got an unexpected keyword argument 'filepath_text_files'` in `build_features.py`.

**Root cause**: Previous microwakeword versions accepted a text file containing one path per line via `filepath_text_files=[...]`. The current API only accepts `input_directory: str + file_pattern: str` and uses `Path(input_directory).glob(file_pattern)` internally.

**Fix applied** (commit `57c1379`): replace the manifest-file mechanism with a flat **symlink farm** per split. For each split (training/validation/testing), create `out_root/<split>/_clips/{NNNNNNN}.wav` symlinks pointing at the actual phrase-subdir WAVs. Then call `Clips(input_directory=str(symlink_dir), file_pattern="*.wav")`.

**v0.2 patch idea**: pin microwakeword to a specific commit (`-e git+https://github.com/.../micro-wake-word.git@SHA`) for stability. Currently we install HEAD.

---

## 23. Relative-path symlinks are dangling — Unix resolves targets from the SYMLINK'S directory, not CWD

**Symptom**: After the symlink farm in #22 was built, `build_features` iteration died with `FileNotFoundError: [Errno 2] No such file or directory: 'data/tofu/features/training/_clips/0007908.wav'`. The symlink existed (visible in directory listing), the target file existed (visible on disk), but `open(symlink)` errored.

**Root cause**: this surprised us. The manifest stores `wav_path` as `"data/tofu/synth/positives/hey_tofu/0.wav"` — relative to /workspace/customwake (the script's CWD). We did `link.symlink_to(p)` where `p` is that relative string. The kernel stores the relative target verbatim in the inode. When something opens the symlink, the kernel resolves `"data/tofu/synth/..."` **relative to the symlink's own directory** (`/workspace/customwake/data/tofu/features/training/_clips/`), giving `/workspace/customwake/data/tofu/features/training/_clips/data/tofu/synth/positives/hey_tofu/0.wav` — which doesn't exist. Every symlink in the farm was silently dangling.

**Fix applied** (commit `1116b68`): `link.symlink_to(Path(p).resolve())` so the symlink target is locked to its absolute path.

**Standard Unix behavior, easy trap**: this is documented (`man symlink`), but the failure message — FileNotFoundError on the symlink path, not the target — makes it look like the symlink was never created. Spent a pod cycle ($1.50, ~30 min) on this one.

---

## 24. `clip_duration_ms` must be divisible by `window_step_ms × stride`

**Symptom**: training runs successfully through 10k steps, then crashes at INT8 quantization export with:

```python
File "/opt/microwakeword/microwakeword/utils.py", line 321, in representative_dataset_gen
    assert spectrogram.shape[0] % stride == 0
AssertionError
```

The trained Keras model is on disk but the `.tflite` is never produced — pipeline dies halfway through `evaluate_model`. Hit this on greet's first run after I dropped `clip_duration_ms` from tofu's 1500 to 1000 to suppress context noise on bare greetings.

**Root cause**: microwakeword's calibration generator (`utils.py:318-324`) requires the spectrogram time-dim to divide evenly by the model's streaming stride. With the defaults from `scripts/train_microwakeword.py`:
- `window_step_ms = 10` (so spectrogram has `clip_duration_ms / 10` slices)
- `stride = 3` (MixedNet default in `MIXEDNET_DEFAULTS`)

The constraint is `clip_duration_ms % 30 == 0`. Tofu's 1500 ms satisfies (150 % 3 = 0). 1000 ms doesn't (100 % 3 = 1). Bumping to 1020 fixes it (102 % 3 = 0).

**Fix**: when setting `clip_duration_ms` for any new project, round up to the nearest multiple of 30:

```python
clip_duration_ms = math.ceil(target_ms / 30) * 30
```

Or for `stride != 3`, multiple of `window_step_ms × stride`.

**Cost**: ~$0.20 — the failure happens at the END of training, so you lose the entire train phase (~20 min on RTX 3090) plus the 25 min synth + features prior. Mitigation: the wrapper can be rerun on the SAME pod with `git pull` instead of `git clone` to preserve synth + features, restoring just the 20 min training cost (~$0.15).

**TODO for the pipeline**: `train_microwakeword.py` should validate this constraint at startup before paying for training. One-line `assert clip_duration_ms % (window_step_ms * stride) == 0` saves the full train cycle.

---

## Final cost ledger (post-v0.1 work)

| Failure mode | Wasted $ | Count |
|---|---|---|
| Errno 5 disk during pip install (now solved by image) | $0.46 × 3 | 3 |
| Failed GHA builds (Errno 28, pip cache purge) | $0 GHA-free | 2 |
| Bad-host stuck (now 14 min vs 7 min) | $0.32 | 2 |
| CUDA wheel mismatch (CPU fallback caught fast) | $0.07 | 1 |
| microwakeword.audio missing | $0.20 | 1 |
| Clips API drift | $0.18 | 1 |
| Relative symlink dangling | $0.41 | 2 |
| **Total this session** | **~$3.20** | of $17 starting balance |

The cost is mostly **iteration**, not raw failure cost. Each "fix → rebuild → relaunch" cycle is ~15 min of wall and $0.30–$0.50 of GPU time. Bundling fixes into a single rebuild whenever possible compresses this.
