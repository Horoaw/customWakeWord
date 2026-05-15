# BUDGET LOG — tofuWakeWord

Line-by-line compute spend. Append after each RunPod run.

| Date | Run | Hardware | Wall | Cost | Outcome | Notes |
|---|---|---|---|---|---|---|
| 2026-05-15 | scaffold | — | — | $0.00 | repo skeleton landed | no GPU spent |

---

## Running total

- **$0.00** spent
- **$5.00** target budget for v0 → v1 cycle
- **$1.00** estimated cost for first v0 training + eval

---

## Rough cost reference

| Resource | Rate | Use case |
|---|---|---|
| RunPod A40 48GB SECURE | $0.39 / hr | microWakeWord training |
| RunPod 4090 24GB SECURE | $0.34 / hr | smaller models / quick iterations |
| RunPod CPU pod (16 vCPU) | $0.10 / hr | bulk corpus downloads, feature extraction |
| HuggingFace Hub | free | model + dataset hosting |
| Piper / Kokoro / MeloTTS / Parler-TTS | free (local CPU) | TTS positives + hard-negatives |
| ElevenLabs (optional) | $0.15–0.30 / 1k chars | premium voice diversity, ~$30 ceiling |
| Azure Neural TTS (optional) | $15 / 1M chars | accent diversity, ~$5 ceiling |
