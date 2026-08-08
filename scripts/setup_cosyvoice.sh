#!/usr/bin/env bash
# Clone CosyVoice, install lean inference deps, download CosyVoice2-0.5B + prompt wav.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${COSYVOICE_REPO_DIR:-$ROOT/third_party/CosyVoice}"
MODEL_DIR="${COSYVOICE_MODEL_DIR:-$ROOT/pretrained_models/CosyVoice2-0.5B}"
PROMPT_WAV="${COSYVOICE_PROMPT_WAV:-$ROOT/assets/prompt.wav}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$ROOT/third_party" "$ROOT/pretrained_models" "$ROOT/assets"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "==> Cloning FunAudioLLM/CosyVoice (with submodules)…"
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$REPO_DIR"
else
  echo "==> CosyVoice already present at $REPO_DIR"
  git -C "$REPO_DIR" submodule update --init --recursive
fi

echo "==> Installing CosyVoice inference dependencies…"
"$PYTHON" -m pip install -U pip
"$PYTHON" -m pip install -r "$ROOT/requirements-cosyvoice.txt"

# Matcha-TTS is a CosyVoice submodule dependency path
if [[ -f "$REPO_DIR/third_party/Matcha-TTS/requirements.txt" ]]; then
  "$PYTHON" -m pip install -r "$REPO_DIR/third_party/Matcha-TTS/requirements.txt" || true
fi

echo "==> Downloading CosyVoice2-0.5B weights to $MODEL_DIR …"
"$PYTHON" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    "FunAudioLLM/CosyVoice2-0.5B",
    local_dir=r"${MODEL_DIR}",
)
print("Model ready:", r"${MODEL_DIR}")
PY

if [[ ! -f "$PROMPT_WAV" ]]; then
  echo "==> Fetching default zero-shot prompt wav…"
  PROMPT_SRC="$REPO_DIR/asset/zero_shot_prompt.wav"
  if [[ -f "$PROMPT_SRC" ]]; then
    cp "$PROMPT_SRC" "$PROMPT_WAV"
  else
    curl -fsSL \
      -o "$PROMPT_WAV" \
      "https://github.com/FunAudioLLM/CosyVoice/raw/main/asset/zero_shot_prompt.wav"
  fi
  echo "Prompt saved to $PROMPT_WAV"
else
  echo "==> Prompt wav already exists: $PROMPT_WAV"
fi

cat <<EOF

Setup complete.

Next:
  1. cp .env.example .env   # then set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
  2. pip install -r requirements.txt
  3. python main.py

VRAM tip (RTX 3050 6GB):
  - Moonshine STT stays on CPU
  - CosyVoice2 loads with FP16 (COSYVOICE_FP16=true)
  - Prefer MOONSHINE_MODEL_ARCH=small_streaming (default) or tiny_streaming
EOF
