# Voice Chat — Moonshine STT + CosyVoice2 TTS + LLM API

Simple hands-free voice loop for an RTX 3050 (6GB VRAM) / 16GB RAM machine:

1. Start the mic  
2. Ask a question  
3. Moonshine transcribes it  
4. Your LLM API answers  
5. CosyVoice2 speaks the reply  

```
mic  →  Moonshine STT (CPU)  →  LLM API  →  CosyVoice2-0.5B TTS (GPU FP16)  →  speakers
```

## Why this fits 6GB VRAM

| Component | Where it runs | Notes |
|-----------|---------------|--------|
| **Moonshine** STT | CPU | `small_streaming` by default (~123M). No GPU needed. |
| **LLM** | Remote API | You supply the key / base URL. Zero local VRAM. |
| **CosyVoice2-0.5B** | GPU FP16 | ~0.5B TTS; loaded without TensorRT/vLLM. |

Models are not held on GPU at the same time for STT+LLM — only CosyVoice uses CUDA.

## Quick start

### 1. Python env

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. CosyVoice2 (one-time)

```bash
bash scripts/setup_cosyvoice.sh
```

This clones [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice), installs lean inference deps, downloads [`FunAudioLLM/CosyVoice2-0.5B`](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B), and copies a default voice prompt into `assets/prompt.wav`.

System packages that help on Ubuntu:

```bash
sudo apt-get install -y sox libsox-dev portaudio19-dev
```

### 3. Configure your LLM API

```bash
cp .env.example .env
```

Edit `.env`:

```env
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

Any OpenAI-compatible endpoint works (OpenAI, Groq, Together, local vLLM/Ollama with an OpenAI shim, etc.).

### 4. Run

```bash
python main.py
```

Speak, pause briefly, hear the answer. `Ctrl+C` to quit.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_API_KEY` | *(required)* | API key |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model id |
| `MOONSHINE_MODEL_ARCH` | `small_streaming` | `tiny_streaming` / `small_streaming` / `medium_streaming` |
| `MOONSHINE_LANGUAGE` | `en` | STT language |
| `COSYVOICE_FP16` | `true` | Keep true on 6GB GPUs |
| `COSYVOICE_PROMPT_WAV` | `assets/prompt.wav` | Reference voice for zero-shot TTS |
| `COSYVOICE_PROMPT_TEXT` | English prompt text | Transcript of the prompt wav |

Swap `assets/prompt.wav` for any short, clean clip of the voice you want (a few seconds is enough), and set `COSYVOICE_PROMPT_TEXT` to what that clip says.

## Project layout

```
main.py                 # entry point
voice_chat/
  app.py                # conversation loop
  stt.py                # Moonshine MicTranscriber
  tts.py                # CosyVoice2 wrapper (FP16)
  llm.py                # OpenAI-compatible client
  config.py             # env loading
scripts/setup_cosyvoice.sh
requirements.txt
requirements-cosyvoice.txt
.env.example
```

## Troubleshooting

- **CUDA OOM** — confirm `COSYVOICE_FP16=true`, use `MOONSHINE_MODEL_ARCH=tiny_streaming`, close other GPU apps, and reboot the process so VRAM is clear.
- **No mic / PortAudio errors** — install `portaudio19-dev` and check `python -c "import sounddevice as sd; print(sd.query_devices())"`.
- **CosyVoice import errors** — re-run `bash scripts/setup_cosyvoice.sh` so `third_party/CosyVoice` and its Matcha-TTS submodule exist.
- **LLM 401 / connection errors** — verify `LLM_API_KEY`, `LLM_BASE_URL`, and that the model name matches your provider.

## Credits

- STT: [moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine) (`moonshine-voice`)
- TTS: [FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)
