# AGENTS.md

## Project overview

`tts-stt` is a small, mostly-offline voice app:

- **Backend/UI**: FastAPI app in `app/main.py`. Voice-assistant UI is `app/static/assistant.html` (served at `/`); the TTS/STT tool page is `app/static/index.html` (served at `/tools`).
- **TTS**: Piper (`app/tts.py`) — voice `en_US-lessac-medium`.
- **STT**: Vosk (`app/stt.py`) — model `vosk-model-small-en-us-0.15`; audio is normalized with `ffmpeg` to 16 kHz mono. Realtime streaming STT is the `/ws/stt` WebSocket (client sends 16 kHz mono int16 PCM; server returns `{"partial"}`/`{"final"}`).
- **LLM**: `app/llm.py` — uses an OpenAI-compatible API when `OPENAI_API_KEY` is set, otherwise a small local model (`Qwen/Qwen2.5-0.5B-Instruct`) via transformers/torch (CPU).

Standard commands (setup, run, API examples) live in `README.md` — use those rather than duplicating here.

## Cursor Cloud specific instructions

- Use the project virtualenv: `source .venv/bin/activate` before running anything.
- Run the dev server with `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` (see `README.md`). There is no separate frontend build — the UI is static HTML served by FastAPI.
- **ML models are downloaded lazily on first use** into `models/` (git-ignored, ~40 MB Vosk + ~60 MB Piper). The very first `/api/tts` or `/api/stt` call after a fresh checkout triggers a network download and will be slow; subsequent calls are fast. The `models/` directory persists across restarts, so pre-warm by hitting the endpoints once.
- **`ffmpeg` is a required system dependency** (STT decodes/normalizes all uploads through it). It is preinstalled on the VM image; it is intentionally NOT in the pip requirements or the update script. If STT returns a decode error, verify `ffmpeg` is on `PATH`.
- Quickest end-to-end smoke test (round-trip) once the server is up:
  ```bash
  curl -s -X POST localhost:8000/api/tts -H 'Content-Type: application/json' \
    -d '{"text":"Hello world"}' --output /tmp/s.wav
  curl -s -X POST localhost:8000/api/stt -F "audio=@/tmp/s.wav"
  ```
- There is no test suite or linter configured yet.
- Browser mic recording in the UI requires `getUserMedia`, which needs a secure context (localhost is treated as secure). File upload and the "Transcribe synthesized" round-trip work without a mic.
- **LLM**: with no `OPENAI_API_KEY`, the app runs a local model via transformers/torch (CPU). The first `/api/chat` downloads `Qwen/Qwen2.5-0.5B-Instruct` (~1 GB) to the HuggingFace cache and CPU inference takes a few seconds per reply — pre-warm by hitting `/api/chat` once. For production/quality, set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL` / `LLM_MODEL`) as a secret.
- **torch is CPU-only** — `requirements.txt` includes `--extra-index-url https://download.pytorch.org/whl/cpu` and pins `torch==2.13.0+cpu` to avoid pulling large CUDA wheels.
- To manually demo realtime mic input without a physical mic, launch Chrome with a fake audio device fed from a WAV file:
  `google-chrome --use-fake-ui-for-media-stream --use-fake-device-for-media-stream --use-file-for-fake-audio-capture=/path/to/question.wav http://localhost:8000` (the WAV must be 16 kHz mono).
