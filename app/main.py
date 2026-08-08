"""FastAPI app exposing TTS and STT endpoints plus a small web UI."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import stt, tts

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="tts-stt", description="Text-to-Speech & Speech-to-Text demo")


class TTSRequest(BaseModel):
    text: str


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "tts_voice": tts.VOICE_NAME,
        "stt_model": stt.MODEL_NAME,
    }


@app.post("/api/tts")
def text_to_speech(req: TTSRequest) -> Response:
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` must not be empty")
    audio = tts.synthesize(text)
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="speech.wav"'},
    )


@app.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...)) -> JSONResponse:
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio upload")
    try:
        text = stt.transcribe(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"text": text})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
