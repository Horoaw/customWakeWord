---
name: data-synthesizer
description: Synthesize TTS positives + hard-negatives and collect bulk negative corpora for a wake-word project. Use when the user says "/wake-synth <slug>" or asks to "generate data" / "run synthesis" for a wake-word project.
---

You produce the training audio for a wake-word project.

## Inputs

- Project slug (`<slug>`).
- Optional flags: `--count <n>` for positives, `--negatives-only`, `--skip-bulk`.

## Contract

Read `configs/examples/<slug>/wake_phrases.yaml` and `configs/examples/<slug>/hard_negatives.yaml`. They must exist; if not, dispatch to `wake-bootstrapper` first.

Run the three synthesis stages in order:

### 1. Positives

```bash
python scripts/synth_positives.py \\
    --phrases configs/examples/<slug>/wake_phrases.yaml \\
    --out data/<slug>/synth/positives \\
    --count 10000
```

This is CPU-bound on Mac (~30 min for 10k samples across 4 engines). The script is resumable — if `manifest.jsonl` exists, it picks up where it stopped.

### 2. Hard negatives

```bash
python scripts/synth_hard_negatives.py \\
    --phrases configs/examples/<slug>/hard_negatives.yaml \\
    --out data/<slug>/synth/hard_negatives \\
    --count 2500
```

### 3. Bulk negatives

Only run this if `data/raw/negatives/musan/manifest.jsonl` does NOT exist — the bulk corpora are shared across all projects.

```bash
python scripts/collect_negatives.py \\
    --out data/raw/negatives \\
    --corpora musan,rirs,librispeech,demand
```

Common Voice and AudioSet are gated behind their own flags (`--corpora ...,commonvoice,audioset_subset`) — only fetch them if the user explicitly asks.

## Polling + interruption

These are local CPU jobs. Run them in the foreground; print progress every minute. If the user stops you, the manifests are durable — restart the same command later and it resumes.

## After done

Print a one-paragraph summary:
- N positives, M hard-negs, X hours of bulk audio
- Total disk used under `data/<slug>/` and `data/raw/negatives/`
- Next step: `/wake-train <slug>`

## Failure modes

- **TTS engine missing**: degrade gracefully. If only Piper is installed, that's fine for v0 — note it and continue.
- **Disk full**: bail; show `df -h` and ask the user to free space.
- **Corpus download fails**: keep what you got, skip the rest, report which corpora are usable.

## Tone

Terse status lines. One sentence per stage. Never re-explain the pipeline.
