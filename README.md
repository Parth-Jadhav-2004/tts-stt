# tts-stt

A minimal, fully offline **Text-to-Speech** and **Speech-to-Text** web app.

- **TTS** — [Piper](https://github.com/OHF-Voice/piper1-gpl) neural voices (`en_US-lessac-medium` by default)
- **STT** — [Vosk](https://alphacephei.com/vosk/) (`vosk-model-small-en-us-0.15` by default)
- **API/UI** — [FastAPI](https://fastapi.tiangolo.com/) + a small single-page web UI

ML models are downloaded lazily on first use into `models/` (git-ignored). No API keys required.

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

## API

| Method | Path          | Body                         | Returns            |
|--------|---------------|------------------------------|--------------------|
| GET    | `/api/health` | –                            | JSON status        |
| POST   | `/api/tts`    | JSON `{"text": "..."}`       | `audio/wav` bytes  |
| POST   | `/api/stt`    | multipart `audio=@file`      | JSON `{"text": ""}`|

### Examples

```bash
# Text -> speech
curl -s -X POST localhost:8000/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello world"}' --output speech.wav

# Speech -> text
curl -s -X POST localhost:8000/api/stt -F "audio=@speech.wav"
```

## Configuration (optional env vars)

| Variable         | Default                        | Description                     |
|------------------|--------------------------------|---------------------------------|
| `MODELS_DIR`     | `models`                       | Where models are stored         |
| `PIPER_VOICE`    | `en_US-lessac-medium`          | Piper voice name                |
| `VOSK_MODEL`     | `vosk-model-small-en-us-0.15`  | Vosk model name                 |
| `VOSK_MODEL_URL` | derived from `VOSK_MODEL`      | Override model download URL      |
