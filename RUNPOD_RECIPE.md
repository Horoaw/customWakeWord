# RUNPOD recipe — how to run customWakeWord pipelines reliably

Bottled know-how from a multi-day debug session running the Tofu wake-word
pipeline on RunPod A100 SXM. Most of these failures aren't documented
anywhere in RunPod's official material — they came from hard iteration.
Use this as the **operating manual** for any future training run.

The order below matches the lifecycle of a single run.

---

## 0. Before you launch — must-haves

| Check | Why |
|---|---|
| `~/.config/tofu-wake/{hf,runpod,gh}.env` exists, chmod 600 | `scripts/load_creds.sh` looks for them; missing → cryptic auth failure mid-pipeline. Fallback path is `~/.config/temllm/` for shared tokens. |
| GHCR package `ghcr.io/temm1e-labs/customwake-deps:vN` is **public** | RunPod pulls without auth. If private, pod gets stuck with `runtime=null` for 10+ min (looks identical to a bad-host). See §4. |
| RunPod balance >= **$5** | One A100 SXM cycle is ~$1–$2 if nothing goes wrong, ~$3–$5 with one iteration. |
| `git push` is clean | The pod git-clones the **remote** main branch, not your local working copy. Uncommitted fixes don't reach the pod. |

---

## 1. The Docker image is your stability budget

**Original v0 approach** (don't): install all deps on the pod each run.
**Result**: 40 min of pip install × every retry, intermittent `OSError [Errno 5]` mid-install, ~$1 wasted per attempt, 60% success rate.

**v0.1+ approach** (do): bake all deps into `ghcr.io/temm1e-labs/customwake-deps:vN`, built on GitHub Actions.

```bash
# Bump tag, then trigger:
gh workflow run build-deps-image -f tag=vN -f push_latest=true
```

The GHA workflow at `.github/workflows/build-deps-image.yml` does:
1. `jlumbroso/free-disk-space` — frees ~25 GB on the runner (default 14 GB free can't fit the install).
2. Build `Dockerfile` → push to GHCR with the version tag + `latest`.
3. ~20 min wall time. Cached layers cut subsequent builds to ~10 min.

### Don't forget the Dockerfile patches

The image isn't "just pip install": it carries **5 inline patches** that took a day each to discover. See `Dockerfile` comments for details, but in summary:

1. **`pip cache purge` after `--no-cache-dir`** — errors (`cache disabled`). Don't chain them.
2. **`piper-sample-generator` pinned to `v2.0.0`** — HEAD moved to a packaged layout (`python -m piper_sample_generator`) on 2026-03-12 with required `--model` arg that breaks our launcher.
3. **Sed-patch `generate_samples.py:74`** — torch 2.6 flipped `torch.load`'s `weights_only` default to True; the v2.0.0 libritts checkpoint contains `piper_train.vits.models.SynthesizerTrn` not in the safe-globals allowlist. Add `weights_only=False`.
4. **`microwakeword/audio/__init__.py` + `layers/__init__.py`** — upstream's `setup.py` uses `find_packages()` but those subdirs ship without `__init__.py`. `pip install` drops them entirely → `ModuleNotFoundError: microwakeword.audio`. Clone, touch the `__init__.py` files, install `-e`, then verify the import at build time.
5. **CUDA wheel pin** (`torch==2.5.1+cu124` from PyTorch's cu124 index) — RunPod A100 SXM hosts ship CUDA driver 12.7. The default PyPI torch (>=2.7 currently) is built with CUDA 13 and refuses driver 12.7, silently falling back to CPU. `torchaudio` must match (`torchaudio==2.5.1+cu124`).

### Light constraint: `datasets<4`

`datasets==4.x` switched audio decoding from `soundfile` to `torchcodec`. We hit:
> `RIR download failed: To support encoding audio data, please install 'torchcodec'.`
Pinning `datasets>=3.0,<4.0` keeps us on the older soundfile-based path.

---

## 2. Public GHCR packages — the org-policy dance

Hugging Face Spaces and RunPod both want to pull images anonymously. By default, **GHCR packages pushed by an org workflow are PRIVATE**, even when the source repo is public. The visibility flip requires THREE conditions to align:

1. **Org-level setting**: GitHub.com → Organizations → your-org → Settings → **Packages** → check **Public** under "Container types". This is *off* by default and produces a deeply unhelpful "*Setting is disabled by organization administrators*" message on the per-package page.
2. **Personal Access Token** with `read:packages` + `write:packages` scopes if you want to flip via API. (`admin:org` alone isn't enough.) Most users just click the UI.
3. **Per-package flip**: the package's Settings page → Danger Zone → **Change visibility** → Public → type the package name to confirm.

Verify with:
```bash
TOK=$(curl -s "https://ghcr.io/token?service=ghcr.io&scope=repository:ORG/PACKAGE:pull" | jq -r .token)
curl -sI -H "Authorization: Bearer $TOK" \
  -H "Accept: application/vnd.oci.image.index.v1+json" \
  "https://ghcr.io/v2/ORG/PACKAGE/manifests/TAG" | head -1
# Want: HTTP/2 200
```

The `Accept` header **matters** — without an explicit media type GHCR returns 404 even for public packages (confusing!).

---

## 3. Pod creation — the secure-network choice

`scripts/runpod_train.py` and `scripts/retry_launch.sh` already encode the right defaults. Key choices:

| Setting | Why |
|---|---|
| `gpuTypeIds: ["NVIDIA A100-SXM4-80GB"]` (1st in retry order) | **Network bandwidth** is the dominant cost driver — server-class hosts (A100 / A40 / L40) have 10+ Gbps NICs in tier-1 DCs. Consumer 4090/3090 hosts choke on HF Hub egress (>2h to download 9 GB vs 5 min on A100). |
| `cloudType: SECURE` | Spot pricing (COMMUNITY) costs less but interrupts mid-training. |
| `containerDiskInGb: 50` + `volumeInGb: 50` | The container disk takes the wheel install (~5 GB extracted); the volume holds checkpoints + features (~10–20 GB). |
| `dockerStartCmd` writes `/workspace/STAGE` markers | Each pipeline step writes a tiny one-line marker file, served via `:8001/STAGE`. The launcher uses this for liveness independent of `setup.log` (which can be clobbered by OOM). |
| Self-restarting `python3 -m http.server 8001` with `oom_score_adj=-1000` | The log server is the only window into a stuck pod. If killed mid-pipeline by OOM, a `while true` restarts it. |

### STUCK timeout: 14 min (not 7)

Earlier versions tripped at 7 min if `runtime=null` (pod created but container hadn't started reporting uptime). With the pre-baked image we have to **pull a 5 GB image from GHCR on a cold host**, which can take 8–12 min legitimately. 14 min absorbs that; faster failure detection isn't worth false-stuck reports.

### **DELETE pods, don't STOP**

`POST /v1/pods/{id}/stop` keeps the storage allocated and continues billing at ~$0.05/GB/month. **Always** use `DELETE /v1/pods/{id}` (HTTP 204) when the run finishes (or fails). The launcher's cleanup path was previously stop; new runs use delete.

---

## 4. When boots stay stuck — diagnostic flowchart

`runtime=null` for >14 min, `desiredStatus=RUNNING`:

| Symptom | Likely cause | Fix |
|---|---|---|
| Logged `pulled image` then nothing | Image isn't public yet OR auth missing | Verify GHCR pull anonymously (§2) |
| `desiredStatus=RUNNING` but `runtime` always `None` | Bad host (low-mem 4090, or rare A100 with bad NIC) | Delete + retry (`scripts/retry_launch.sh` does this) |
| Image cached on host (boots in 30 s) but no STAGE file | Entrypoint crash before `write_stage "boot"` line | `curl .../setup.log` — usually a typo in the bash; or pull errored silently |
| Pull seems to succeed but pod dies right after | `containerRegistryAuthId` referencing a stale or rotated PAT | Re-register: `POST /v1/containerregistryauth` |

---

## 5. The pipeline stages and their typical durations

A clean A100 SXM run with the v0.5+ image, **post-fixes**, on a host that already cached the image:

| Stage | Wall (A100 SXM) | Failure mode to watch for |
|---|---|---|
| Pod create → first STAGE | 30 s – 4 min | Image pull (cold), or stuck (bad host) |
| `image_check` | ~5 s | TF/torch import failure (CUDA wheel mismatch — see §1.5) |
| `git_clone` | ~10 s | Wrong branch, missing GH_TOKEN |
| `piper_link` | <1 s | First run only; symlinks `/opt/piper-sample-generator` → `/workspace/piper-sample-generator` |
| `synth_positives` | 2–3 min on A100 | `OSError [Errno 5]` mid-WAV-write on bad-disk hosts → some WAVs silently truncated (size says OK, content missing). Defended by `wave.open()` validation in `build_features.py`. |
| `synth_hard_negatives` | 4–6 min | Same disk-flakiness pattern as above |
| `download_hf_negatives` | 3–13 min | HF cache hits make this 3 min on a hot host; cold hosts vary. The downloaded files are mmap-ninja **features**, not raw audio. |
| `build_features` | 5–10 min | Aug corpora downloads (RIR succeeds, FMA succeeds, AudioSet 404s — that one's been removed from upstream, soft-fail is fine). Then spectrogram extraction with augmentation. |
| `train` | 12–15 min on A100 | TF data pipeline errors most often from mmap dirs not having `train.npy` / `test.npy` / `validation.npy` — symptom of build_features crashing earlier. |
| `eval` | 1–3 min | Skipped if `eval/tasks/{project}` isn't seeded |
| `emit_manifest` | <5 s | Reads eval JSON; if eval skipped, uses CLI threshold default |
| `upload_to_hf` | 1–3 min | Creates HF repo, pushes `.tflite` + `manifest.json` + model card |

**Total clean run:** ~30–45 min, ~$0.70–$1.04 on A100 SXM at $1.39/hr.

---

## 6. Relative-path symlink trap (this one cost a full pod cycle)

The single non-obvious fail in the pipeline:

> **Symlinks store their target string verbatim. The kernel resolves
> a relative target from the SYMLINK'S directory, not from the
> calling process's CWD.**

The manifest stores `wav_path` as `data/tofu/synth/positives/.../*.wav` (relative). When `build_features.py` glob-stages those WAVs as symlinks under `data/tofu/features/training/_clips/`, every symlink points at `<that dir>/data/tofu/synth/positives/...` which doesn't exist → every iteration fails with `FileNotFoundError: 'data/tofu/features/training/_clips/0007908.wav'`. The error message even looks like the symlink itself is missing (it's not — its target is).

Fix: `link.symlink_to(Path(p).resolve())` so the target is locked to its absolute path.

This is **not** specific to RunPod, but it bit us only there because (a) the synth+build_features path doesn't run locally, (b) the failure looks identical to "your filter missed something". Defense in `build_features.py`: also validate via `wave.open()` at filter time so genuinely broken WAVs are caught.

---

## 7. Cost discipline

| Mistake | Typical waste |
|---|---|
| Forgetting to DELETE a stopped pod | $0.05/GB/month — slow drip, easy to miss |
| Launcher killed via TaskStop (bypasses Python `finally`) | Pod runs orphan at $1.39/hr until you notice |
| `--no-cache-dir` pip install in entrypoint (not image) | 40 min × $1.39/hr = $0.93 per run, every run |
| Running a bad-host that doesn't boot | $0.32 per 14-min STUCK trip |
| Iterating on the launcher locally with a long-running pod | Fix takes 30 s, but the pod's been billing for 20 min |

**Rule of thumb**: any time you `TaskStop` a launcher, run `curl -X DELETE` on the pod within 1 min. The launcher does NOT clean up after itself when killed externally.

---

## 8. The retry contract

`scripts/retry_launch.sh` provides a single "do until done" mode:

```bash
MAX_ATTEMPTS=10 SLEEP_BETWEEN=45 bash scripts/retry_launch.sh
```

It rotates GPU tiers premium-first (A100 → L40S → L40 → A40 → 4090 → 3090 → 4090-COMMUNITY) and treats these as recoverable (next pool):
- HTTP 500 "no instances currently available" — capacity drought
- `RuntimeError: STUCK: pod runtime null after 14 min` — bad host

Everything else is unrecoverable; the launcher exits 1 and the pod is deleted by the same script's exit handler.

---

## 9. After the pipeline — what's left

1. **Verify HF push**: `curl -s https://huggingface.co/api/models/<repo> | jq '.id, .lastModified'`
2. **Delete the pod** even if successful: `curl -X DELETE /v1/pods/{id}` returns 204.
3. **Rotate `HF_TOKEN`, `GH_TOKEN`**: RunPod's pod-list API leaks env vars. The training run wrote them to its env block; assume they're compromised after any failure.
4. **Update this doc** when something new bites — it's the only place these are written down.

---

## See also

- `LESSONS_v0.md` — chronological failure-mode catalog (numbered #1–22, in failure order).
- `PROVIDERS.md` — cross-provider decision matrix; when to bail on RunPod entirely.
- `Dockerfile` — the image is the most concentrated form of all these lessons.
- `.github/workflows/build-deps-image.yml` — GHA build with the free-disk-space step.
