# tts-stt

A minimal, fully offline **Voice Assistant** plus **Text-to-Speech** and **Speech-to-Text** tools.

- **Voice assistant** (`/`) — speak → live on-screen transcript → press **Enter** → an LLM replies → the answer is spoken back.
- **TTS** — [Piper](https://github.com/OHF-Voice/piper1-gpl) neural voices (`en_US-lessac-medium` by default)
- **STT** — [Vosk](https://alphacephei.com/vosk/) (`vosk-model-small-en-us-0.15` by default), realtime over WebSocket
- **LLM** — OpenAI-compatible API when `OPENAI_API_KEY` is set, else a small local model (`Qwen/Qwen2.5-0.5B-Instruct`) via transformers
- **API/UI** — [FastAPI](https://fastapi.tiangolo.com/) + a small single-page web UI (`/` assistant, `/tools` TTS/STT tools)

ML models are downloaded lazily on first use into `models/` / the HuggingFace cache. No API keys required (a local model is used as fallback).

## Requirements

- Python 3.12
- `ffmpeg` (used to normalize uploaded audio to 16 kHz mono for STT)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (development)

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000

## Voice assistant flow

1. Open `/`, click the mic, and start speaking — your words appear live.
2. Press **Enter** (or Send). The transcript is sent to the LLM.
3. The reply is shown as text and read aloud automatically.

You can also type instead of speaking. The realtime transcript streams over
`ws://<host>/ws/stt` (client sends 16 kHz mono PCM; server returns
`{"partial": ...}` / `{"final": ...}` frames).

## API

| Method | Path          | Body                          | Returns                    |
|--------|---------------|-------------------------------|----------------------------|
| GET    | `/api/health` | –                             | JSON status + LLM info     |
| POST   | `/api/tts`    | JSON `{"text": "..."}`        | `audio/wav` bytes          |
| POST   | `/api/stt`    | multipart `audio=@file`       | JSON `{"text": ""}`        |
| POST   | `/api/chat`   | JSON `{"text": "...", "history"?}` | JSON `{"reply": ""}`  |
| WS     | `/ws/stt`     | binary 16 kHz mono PCM frames | JSON partial/final frames  |

### Examples

```bash
# Text -> speech
curl -s -X POST localhost:8000/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello world"}' --output speech.wav

# Speech -> text
curl -s -X POST localhost:8000/api/stt -F "audio=@speech.wav"

# Ask the LLM
curl -s -X POST localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"text":"What is the capital of Japan?"}'
```

## Configuration (optional env vars)

| Variable          | Default                        | Description                              |
|-------------------|--------------------------------|------------------------------------------|
| `MODELS_DIR`      | `models`                       | Where TTS/STT models are stored          |
| `PIPER_VOICE`     | `en_US-lessac-medium`          | Piper voice name                         |
| `VOSK_MODEL`      | `vosk-model-small-en-us-0.15`  | Vosk model name                          |
| `VOSK_MODEL_URL`  | derived from `VOSK_MODEL`      | Override model download URL               |
| `OPENAI_API_KEY`  | –                              | If set, use an OpenAI-compatible LLM API |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1`    | OpenAI-compatible base URL               |
| `LLM_MODEL`       | `gpt-4o-mini`                  | Model name for the API path              |
| `LOCAL_LLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct`   | Local model used when no API key is set  |
| `LLM_MAX_TOKENS`  | `160`                          | Max tokens in the reply                  |
