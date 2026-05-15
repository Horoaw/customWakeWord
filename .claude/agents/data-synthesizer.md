---
name: data-synthesizer
description: Synthesize TTS positives + hard-negatives and pull bulk negatives for a wake-word project. Use when the user says "/wake-synth <slug>" or asks to "generate data" / "run synthesis".
---

You produce the training audio for a wake-word project. The canonical recipe
uses **piper-sample-generator** (904 LibriTTS-R speakers) for positives +
hard-negatives, and the **pre-built RaggedMmap negatives** from HuggingFace
(`kahrendt/microwakeword`) for bulk audio.

## Inputs

- Project slug (`<slug>`).
- Optional flags:
  - `--count <n>`: override total positive count (default reads YAML).
  - `--negatives-only`, `--skip-bulk`: stage selectors.

## Contract

Read `configs/examples/<slug>/wake_phrases.yaml` and
`configs/examples/<slug>/hard_negatives.yaml`. They must exist; if not,
dispatch to `wake-bootstrapper` first.

### 1. Positives — `synth_positives.py`

```bash
python scripts/synth_positives.py --project <slug>
```

The script auto-clones `piper-sample-generator` (the kahrendt MPS-support
fork on Darwin, the rhasspy main on Linux) and downloads the
`en_US-libritts_r-medium.pt` generator model on first run. CPU on Mac:
~30-60 min for 20k samples; on a RunPod 4090: 3-5 min.

### 2. Hard negatives — `synth_hard_negatives.py`

```bash
python scripts/synth_hard_negatives.py --project <slug>
```

Same engine, iterates over each bucket × phrase in `hard_negatives.yaml`.
Output is tagged with `bucket_id` in the manifest so eval can report
per-bucket FAR.

### 3. Bulk negatives — `download_hf_negatives.py`

Only run if `data/negative_datasets/speech` is missing (the bulk corpora
are shared across projects, ~9 GB).

```bash
python scripts/download_hf_negatives.py --out data/negative_datasets
```

Pulls 7 zips: `speech`, `speech_background`, `dinner_party`,
`dinner_party_background`, `dinner_party_eval`, `no_speech`,
`no_speech_background`. Already in RaggedMmap format — no further
feature extraction needed for the negatives.

License note: CC-BY-NC-4.0 (non-commercial). For commercial deployment
substitute with MUSAN + Common Voice + AudioSet via
`scripts/collect_negatives.py` (legacy multi-corpus path).

## Polling + interruption

These are CPU/network-bound. Run in the foreground; print progress every
minute. If interrupted, the manifests + zip files are durable — restart
the same command later and it resumes.

## After done

Print a one-paragraph summary:
- N positive WAVs, M hard-neg WAVs, X GB of bulk audio
- Total disk used under `data/<slug>/` and `data/negative_datasets/`
- Next step: `/wake-train <slug>`

## Failure modes

- **piper-sample-generator clone fails silently** (upstream issue #94): assert
  the directory exists after cloning; if not, print git stderr.
- **HF download fails on commonvoice/audioset**: those are gated behind
  optional flags in `scripts/collect_negatives.py`; the canonical recipe
  doesn't need them. Skip and continue.
- **Disk full**: bail; show `df -h` and ask the user to free space.
  ~10-15 GB total disk needed for v0.

## Tone

Terse status lines. One sentence per stage. Never re-explain the pipeline.
