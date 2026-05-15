# PROVIDERS — compute-provider choice for wake-word training

Decision record + reference table for picking a GPU provider when launching training runs from this repo. Captures the actual pain points hit on RunPod 2026-05-15 and the cost/effort/reliability tradeoffs across alternatives.

**TL;DR ranking** (best fit first for this repo's hobby-→-commercial trajectory):

1. **Lambda Labs** — best balance; drop-in replacement for the RunPod launcher
2. **RunPod** — cheapest but flakiest; current default
3. **Modal** — easiest end-to-end but requires Python rewrite
4. **GCP** — most reliable; right when going commercial
5. **AWS** — only if existing organizational dependency
6. **Vast.ai / Together / Hyperstack** — situational only

---

## Within-RunPod tier matters

Observed 2026-05-15: **GPU tier on RunPod is a bigger lever than compute speed for this workload.**

| RunPod tier | Pod $/hr | pip install | HF 9 GB download |
|---|---|---|---|
| **Server-class** (A100/A40/L40/L40S, server DC, 10+ Gbps NIC) | $0.44-$1.39 | **~15 min** | minutes |
| **Consumer-class** (4090/3090, budget DC, 1-2 Gbps NIC) | $0.34-$0.69 | ~40 min | hours |

The 2-3× compute markup on A40 vs 4090 is fully recovered by network speed. **Pick server-class first on RunPod** unless cost-minimizing is the absolute priority — see `retry_launch.sh` for the canonical ordering (A100 → L40S → L40 → A40 → 4090 → 3090).

---

## Decision matrix

| Provider | A100 80GB rate | Setup effort for me | Reliability | Network | Lock-in |
|---|---|---|---|---|---|
| **RunPod** | $1.39/hr SECURE, $1.04 spot | Low (REST + 1 token) | ⚠️ 60% (capacity droughts, host roulette) | Slow (consumer DCs) | None |
| **Lambda Labs** | $1.49/hr | Low (REST + 1 token) | ✓ 90% | Fast (DC-grade) | None |
| **Modal** | ~$4/hr | High (rewrite as @modal.function) | ✓ 95% (serverless) | Very fast | Python API lock-in |
| **GCP** | $5.07/hr (a2-ultragpu-1g) | Moderate (gcloud + service account) | ✓ 99% | Excellent | Low |
| **AWS** | $5.12/hr (p4d) | High (IAM + VPC + EC2 vs SageMaker) | ✓ 99% | Excellent | Low |
| **Vast.ai** | $0.80-1.50/hr | Low (REST + 1 token) | ⚠️ 50% (spot-only, more variable than RunPod) | Variable | None |
| **Together AI** | n/a | n/a — hosted inference only, no custom training | — | — | — |

Rates current as of 2026-05. Network "speed" measured roughly by HF Hub egress in MB/s and small-package pip install wall time.

---

## Real costs for ONE Tofu training cycle (compute alone)

Based on observed wall times during the 2026-05-15 session.

| Stage | Wall (RunPod) | RunPod $ | Wall (Lambda) | Lambda $ | Wall (GCP) | GCP $ |
|---|---|---|---|---|---|---|
| Image pull + boot | 3-12 min | $0.07-$0.28 | <2 min | <$0.05 | <2 min | <$0.17 |
| pip install (5GB deps) | 40 min | $0.46 | 8-12 min | $0.25 | 8-12 min | $0.85 |
| TTS (20k positives) | 5 min | $0.06 | 5 min | $0.12 | 5 min | $0.42 |
| HF mmap download (9 GB) | 60-90 min | $0.69-$1.03 | 2-3 min | $0.06 | 2-3 min | $0.21 |
| Feature build | 10-15 min | $0.12-$0.17 | 10-15 min | $0.31 | 10-15 min | $1.06 |
| Train (10k steps MixedNet) | 15 min | $0.17 | 12 min | $0.30 | 8 min | $0.68 |
| Manifest + HF upload | 1 min | $0.01 | 1 min | $0.02 | 1 min | $0.08 |
| **Total typical** | **~135 min, $1.58** | | **~38 min, $1.11** | | **~30 min, $3.47** | |
| **Total worst case** | **~$3 + retries** | | **~$1.50** | | **~$5** | |

Plus GCP-specific extras: network egress (~$0.08/GB out → +$0.72 if data leaves the project), persistent disk ($0.17/GB-month).

Lambda Labs is the only provider where the FASTER NETWORK actually overcomes the higher hourly rate to produce a lower total cost than RunPod. GCP is faster but pays for the markup; the wall-time savings don't recover it.

---

## Failure modes hit on each (observed or expected)

### RunPod (observed 2026-05-15)
- Capacity drought: 4× HTTP 500 "no instances currently available" across 6 GPU types in 5 min
- Host roulette: 3× pods stuck at uptime=0 for >10 min on low-memory hosts (≤41 GB RAM 4090s)
- Orphan pod: 1× pod left running after TaskStop killed the launcher (silent $0.69/hr burn)
- Bad image tag: 1× wasted ~12 min on a non-existent `pytorch:2.4.0-py3.10-...`
- Token leak: pod listing API echoes env vars including HF_TOKEN, GH_TOKEN

### Lambda Labs (expected)
- Smaller GPU catalog — no L40S, no 4090; A100 + H100 only
- Fewer regions → potential queue when bursting
- Higher rate (~+8% vs RunPod) — small premium for predictability

### Modal (expected)
- Python-only — requires rewriting `scripts/runpod_train.py` as `@modal.function`
- Serverless billing precision means cold-start latency is your friend, but iteration becomes a rebuild loop
- Volumes/secrets API has its own learning curve

### GCP (expected)
- Initial setup: enable Compute Engine API, create service account, generate JSON key, install + configure `gcloud`
- A2 / G2 instance types have separate quotas — must request increases
- IAM scopes for GCS access from VM (workload identity preferred)
- Network egress charges for HF downloads if data path leaves the region

### AWS (expected)
- Most ceremony: VPC, subnet, security group, IAM role, EC2 launch template, EBS volume, SSH key pair, possibly SageMaker training job
- p4d/p4de instances have low default quotas — capacity reservation often needed
- Spot interruption is real

### Vast.ai (expected)
- Community hosts → variable reliability per provider
- Spot-only on best prices
- Occasional security concerns from shared hardware

---

## My migration plan if this repo grows

**Phase 1 (now, hobby/v0)**: stay on RunPod. The repo is already wired to it. v0.1 hardening (LESSONS_v0.md) fixes most of the failure modes that aren't capacity-droughts. ~$2-3/run when it works.

**Phase 2 (v0.1+, iteration)**: port `runpod_train.py` → `lambda_train.py`. Same agent contract (slash commands, manifest emission, HF upload). Drop-in. ~30 min of code work. ~$1.50/run, ~90% reliability.

**Phase 3 (v1, "Tofu becomes a real product")**: add `gcp_train.py` as a parallel path. Use it for nightly canaries + production checkpoint runs. Keep Lambda for iteration. Total cost grows but ops simplicity is a major win.

**Phase 4 (commercial scale)**: pick GCP or AWS based on existing infra dependencies. Add Modal sidecars for serverless inference if latency demands it.

---

## When to switch *during* a single session

Mid-session escape hatch — if `retry_launch.sh` cycles 5+ times with all-pools-dry, switch providers immediately rather than waiting:

| Symptom | Action |
|---|---|
| All RunPod GPUs denied 3× in a row | Try Lambda Labs (it's a different pool) |
| Lambda Labs queue is full | Try GCP us-central1 |
| Specific GPU type unavailable everywhere | Drop down in tier (L40 → A40 → 4090 → 3090) |
| One specific provider is having an outage | Switch immediately, don't wait |

---

## Auth setup (one-time, per provider)

Each provider needs a single env file in `~/.config/tofu-wake/` parallel to the existing ones:

| Provider | Env file | Key vars |
|---|---|---|
| RunPod | `runpod.env` | `RUNPOD_API_KEY` |
| Lambda Labs | `lambda.env` | `LAMBDA_API_KEY` |
| Modal | `modal.env` | `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` |
| GCP | `gcp.env` | `GCP_PROJECT_ID`, `GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/tofu-wake/gcp-sa.json` |
| AWS | `aws.env` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` |
| Vast.ai | `vast.env` | `VAST_API_KEY` |

`scripts/load_creds.sh` already iterates `~/.config/tofu-wake/*.env`; new providers slot in without code changes.

---

## How I (Claude) decide in a session

When the user says "train Tofu" without specifying provider:

1. **Check `configs/preferred_provider.txt`** (one of: `runpod` / `lambda` / `modal` / `gcp`) — defaults to `runpod` if absent.
2. If the preferred provider denies 3× in a row OR all GPUs are dry → fail over to the next-tier provider with a message.
3. If the user asks "switch to X", honor immediately.

Tracking issue: codify this fallback chain in `scripts/runpod_train.py` as a `--providers` flag instead of hardcoded RunPod. v0.2 work.

---

## Reference URLs

- **Lambda Labs**: <https://lambdalabs.com/service/gpu-cloud/pricing> · API docs: <https://cloud.lambdalabs.com/api/v1/docs>
- **Modal**: <https://modal.com/pricing> · GPU types: <https://modal.com/docs/guide/gpu>
- **GCP**: <https://cloud.google.com/compute/all-pricing#gpus> · A2 instances: <https://cloud.google.com/compute/docs/gpus>
- **AWS**: <https://aws.amazon.com/ec2/instance-types/p4/> · SageMaker training: <https://aws.amazon.com/sagemaker/pricing/>
- **Vast.ai**: <https://vast.ai/pricing> · API: <https://vast.ai/docs/api>
- **Hyperstack**: <https://www.hyperstack.cloud/gpu-pricing> · also worth checking ad-hoc
- **CoreWeave**: <https://www.coreweave.com/pricing> · enterprise-tier alternative

GPU price comparison tools:
- <https://www.shadeform.ai/> — aggregator across providers
- <https://www.deployhq.com/> — cross-provider cost calculator
