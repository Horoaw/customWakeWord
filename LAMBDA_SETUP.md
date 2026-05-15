# LAMBDA_SETUP — one-time setup for the Lambda Labs pipeline

Walks you through registering on Lambda Labs, generating an API key, and
plumbing it into this repo so `scripts/lambda_train.py` works.

Total time: ~10 min after Lambda registration is approved (their account
approval can take 0-24 h on first signup).

---

## Phase A — Register on Lambda Labs

1. Sign up at <https://lambdalabs.com/>. Click "Sign up" → "GPU Cloud".
2. Verify email.
3. **Wait for account approval.** Lambda manually reviews new accounts; this
   typically happens within an hour during US business hours, up to a day
   otherwise. You'll get an email.
4. Add payment method at <https://cloud.lambdalabs.com/billing> and put $20
   of credit (matches the RunPod budget — same expected total cost).

---

## Phase B — Generate API key

1. Go to <https://cloud.lambdalabs.com/api-keys>.
2. Click "Generate API Key" → copy the token (looks like
   `secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`).
3. Add it to your credentials directory:

```bash
cat > ~/.config/tofu-wake/lambda.env << 'EOF'
export LAMBDA_API_KEY=secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF
chmod 600 ~/.config/tofu-wake/lambda.env
```

(If `~/.config/tofu-wake/` doesn't exist, our `load_creds.sh` will fall
back to `~/.config/temllm/` — drop the file there instead. It works the
same way.)

---

## Phase C — SSH key

Lambda Labs uses SSH key auth — no `dockerStartCmd` like RunPod has.
You need an SSH keypair on your Mac and the public half uploaded to Lambda.

```bash
# 1. Use an existing key (ed25519 preferred) or generate a new one
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# 2. scripts/lambda_train.py will upload it automatically on first run.
# Or pre-upload it manually:
source scripts/load_creds.sh
curl -u "${LAMBDA_API_KEY}:" -X POST https://cloud.lambdalabs.com/api/v1/ssh-keys \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"tofu-wake\", \"public_key\": \"$(cat ~/.ssh/id_ed25519.pub)\"}"
```

`scripts/lambda_train.py` defaults to `~/.ssh/id_ed25519` — override with
`--ssh-key /path/to/key` if you use a different one.

---

## Phase D — Test the launcher (dry-run, no compute)

```bash
source scripts/load_creds.sh
python scripts/lambda_train.py --project tofu --hf-repo-id nagisanzeninz/tofu-wakeword-v0 --dry-run
```

Should print the planned launch payload + the bash script it would run
remotely. No actual instance created.

---

## Phase E — Launch (when you give the go signal)

```bash
python scripts/lambda_train.py --project tofu --hf-repo-id nagisanzeninz/tofu-wakeword-v0
```

The launcher:
1. Reserves an instance (default `gpu_1x_a100` @ $1.29/hr; cheapest is
   `gpu_1x_a10` @ $0.75/hr if you want).
2. Polls until SSH is reachable (~60-90s for Lambda — faster than RunPod's
   3-5 min).
3. SSHes in, runs `scripts/_lambda_setup.sh` on the instance.
4. Streams the remote setup log live to your local terminal (no proxy log
   server hack — just real SSH).
5. Polls for `_done` marker file via SSH (no public HTTP needed).
6. On completion: uploads to HF, terminates the instance.

Expected total wall time: **~35-50 min** (faster than RunPod because
network is consistently good — pip install will be ~10-15 min, HF
download ~5 min, etc.).

Expected cost on `gpu_1x_a100`: **~$0.80-1.10**.

---

## Differences from the RunPod path you've been using

| | RunPod | Lambda Labs |
|---|---|---|
| Entrypoint | dockerStartCmd JSON | SSH + bash |
| Log access | HTTP proxy on :8001 (flaky) | Direct SSH stream |
| Capacity model | Spot-style, denials common | Reserved, queue if dry |
| Per-pod env vars | Passed in pod payload | Set inline in SSH session |
| Stop method | DELETE /pods/<id> | terminate via API |
| Auth | `Authorization: Bearer <key>` | `Basic <key>:` |
| Tokens leaked? | Yes, pod listing API echoes env | No — env is session-local |

Everything else (synth scripts, training, eval, upload) is identical —
they run inside the Lambda instance the same way they ran on the RunPod
pod.

---

## Troubleshooting

- **`SSH connection refused`**: Lambda's instances need ~60-90s after
  status=active before SSH accepts. The launcher retries automatically.
- **`Permission denied (publickey)`**: your public key isn't registered.
  Re-run the curl in Phase C, or pass `--upload-ssh-key` to the launcher.
- **`No instances available`**: Lambda's GPU pool is dry in your region.
  Try `--region us-west-2` or a different instance type.
- **`Authentication failed`**: API key bad or revoked. Regenerate at
  <https://cloud.lambdalabs.com/api-keys>.

---

## When you're ready

Say go (or just run the command). I'll surface stages and metrics in real
time same as before. If anything fails I'll classify (transient vs config
vs Lambda-side) and either retry or escalate back to RunPod with a custom
Docker image.
