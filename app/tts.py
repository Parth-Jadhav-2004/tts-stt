"""Text-to-Speech using Piper (offline neural TTS)."""
from __future__ import annotations

import io
import os
import threading
import wave
from pathlib import Path

from piper import PiperVoice
from piper.download_voices import download_voice

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models")) / "piper"
VOICE_NAME = os.environ.get("PIPER_VOICE", "en_US-lessac-medium")

_lock = threading.Lock()
_voice: PiperVoice | None = None


def _ensure_voice_files() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx = MODELS_DIR / f"{VOICE_NAME}.onnx"
    if not onnx.exists():
        print(f"[tts] downloading Piper voice '{VOICE_NAME}' -> {MODELS_DIR} ...")
        download_voice(VOICE_NAME, MODELS_DIR)
        print("[tts] voice download complete")
    return onnx


def load() -> PiperVoice:
    """Load (and lazily download) the Piper voice. Thread-safe & cached."""
    global _voice
    if _voice is None:
        with _lock:
            if _voice is None:
                onnx = _ensure_voice_files()
                _voice = PiperVoice.load(str(onnx))
    return _voice


def synthesize(text: str) -> bytes:
    """Synthesize `text` and return WAV audio bytes."""
    voice = load()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(text, wf)
    return buf.getvalue()
