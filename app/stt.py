"""Speech-to-Text using Vosk (offline). Audio is normalized via ffmpeg."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import wave
import zipfile
from pathlib import Path
from urllib.request import urlopen

from vosk import KaldiRecognizer, Model, SetLogLevel

SetLogLevel(-1)

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models")) / "vosk"
MODEL_NAME = os.environ.get("VOSK_MODEL", "vosk-model-small-en-us-0.15")
MODEL_URL = os.environ.get(
    "VOSK_MODEL_URL",
    f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip",
)
SAMPLE_RATE = 16000

_lock = threading.Lock()
_model: Model | None = None


def _ensure_model_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = MODELS_DIR / MODEL_NAME
    if not model_dir.exists():
        print(f"[stt] downloading Vosk model '{MODEL_NAME}' ...")
        zip_path = MODELS_DIR / f"{MODEL_NAME}.zip"
        with urlopen(MODEL_URL) as resp, open(zip_path, "wb") as f:
            f.write(resp.read())
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(MODELS_DIR)
        zip_path.unlink(missing_ok=True)
        print("[stt] model download complete")
    return model_dir


def load() -> Model:
    """Load (and lazily download) the Vosk model. Thread-safe & cached."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = Model(str(_ensure_model_dir()))
    return _model


def _to_wav_16k_mono(audio_bytes: bytes) -> bytes:
    """Convert arbitrary audio bytes to 16kHz mono PCM WAV via ffmpeg."""
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", "pipe:1",
        ],
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise ValueError(
            "Could not decode audio: " + proc.stderr.decode("utf-8", "ignore")[:500]
        )
    return proc.stdout


class StreamRecognizer:
    """Incremental recognizer for realtime 16 kHz mono PCM (int16) chunks."""

    def __init__(self) -> None:
        self._rec = KaldiRecognizer(load(), SAMPLE_RATE)
        self._rec.SetWords(True)

    def accept(self, pcm_bytes: bytes) -> dict:
        """Feed a chunk. Returns {"final": text} when an utterance ends,
        otherwise {"partial": text} with the in-progress hypothesis."""
        if self._rec.AcceptWaveform(pcm_bytes):
            return {"final": json.loads(self._rec.Result()).get("text", "")}
        return {"partial": json.loads(self._rec.PartialResult()).get("partial", "")}

    def final(self) -> str:
        return json.loads(self._rec.FinalResult()).get("text", "")


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe audio bytes (any ffmpeg-decodable format) to text."""
    model = load()
    wav_bytes = _to_wav_16k_mono(audio_bytes)

    import io

    wf = wave.open(io.BytesIO(wav_bytes), "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    parts: list[str] = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            parts.append(json.loads(rec.Result()).get("text", ""))
    parts.append(json.loads(rec.FinalResult()).get("text", ""))
    return " ".join(p for p in parts if p).strip()
