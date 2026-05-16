# Pre-baked deps image for customWakeWord training pipeline.
#
# Built via .github/workflows/build-deps-image.yml on GitHub Actions (x86_64
# native, ~15 min). Pushed to ghcr.io/temm1e-labs/customwake-deps:<tag>.
#
# WHY THIS EXISTS: RunPod A100 SXM pods consistently fail (3/3) during pip
# install's "Installing collected packages" phase with OSError [Errno 5]
# Input/output error — the cheap pod disks can't handle extracting ~5 GB of
# wheels (PyTorch + TensorFlow + nvidia-* + microwakeword + transitive deps).
# Baking the deps into a Docker image moves the pip extract step off the pod
# entirely. The pod just clones the repo + runs scripts.
#
# See LESSONS_v0.md #15 for the failure analysis.

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages — install in one layer to minimize image size
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3.10-distutils \
        ffmpeg sox libsox-fmt-mp3 unzip wget git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Fresh venv on Python 3.10 — microwakeword pins >=3.10,<3.11 (upstream
# OHF-Voice/micro-wake-word issue #62). All deps live in /opt/venv so
# `source /opt/venv/bin/activate` on the pod picks up everything.
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

RUN /opt/venv/bin/pip install --upgrade pip wheel

# This is the step that fails on RunPod pod disks. Doing it at image-build
# time on GitHub Actions' SSD instead. --no-cache-dir means pip already
# discards cached wheels after install; no separate purge needed.
COPY requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-clone piper-sample-generator at v2.0.0 — saves another 30s on the pod.
# PIN: v2.0.0 is the last release with the old `generate_samples.py` at the
# repo root. v3.x reorganized into the `piper_sample_generator` package +
# `python -m piper_sample_generator` invocation, and dropped the implicit
# model resolution that our synth scripts rely on (they don't pass --model).
# Stay on v2.0.0 until scripts/synth_*.py migrates to the new API.
#
# Inline patch: PyTorch 2.6 flipped `torch.load`'s default `weights_only`
# from False → True. The libritts_r-medium.pt checkpoint contains
# `piper_train.vits.models.SynthesizerTrn` which is not on the safe-globals
# allowlist, so the default load fails. We trust the upstream checkpoint,
# so add `weights_only=False`. Failing to patch produces:
#   _pickle.UnpicklingError: Weights only load failed. ... WeightsUnpickler
#   error: Unsupported global: GLOBAL piper_train.vits.models.SynthesizerTrn
RUN git clone --depth 1 --branch v2.0.0 \
        https://github.com/rhasspy/piper-sample-generator.git \
        /opt/piper-sample-generator \
    && mkdir -p /opt/piper-sample-generator/models \
    && wget -q -O /opt/piper-sample-generator/models/en_US-libritts_r-medium.pt \
        https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt \
    && test -f /opt/piper-sample-generator/generate_samples.py \
    && sed -i 's/torch\.load(model_path)/torch.load(model_path, weights_only=False)/' \
        /opt/piper-sample-generator/generate_samples.py \
    && grep -q 'weights_only=False' /opt/piper-sample-generator/generate_samples.py

# Sanity check — fail the build if any critical import breaks
RUN /opt/venv/bin/python -c "import tensorflow as tf; print('tf', tf.__version__)" \
    && /opt/venv/bin/python -c "import torch; print('torch', torch.__version__)" \
    && /opt/venv/bin/python -c "import microwakeword; print('microwakeword OK')" \
    && /opt/venv/bin/python -c "import audiomentations; import librosa; import mmap_ninja; print('audio stack OK')"

LABEL org.opencontainers.image.source="https://github.com/temm1e-labs/customWakeWord"
LABEL org.opencontainers.image.description="customWakeWord deps pre-installed — Python 3.10, CUDA 12.4, TF 2.16+, torch, microwakeword"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /workspace
