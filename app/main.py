"""FastAPI app exposing TTS and STT endpoints plus a small web UI."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import llm, stt, tts

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="tts-stt", description="Text-to-Speech & Speech-to-Text demo")


class TTSRequest(BaseModel):
    text: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    text: str
    history: list[ChatMessage] | None = None


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "tts_voice": tts.VOICE_NAME,
        "stt_model": stt.MODEL_NAME,
        "llm": llm.info(),
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> JSONResponse:
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` must not be empty")
    history = [m.model_dump() for m in (req.history or [])]
    try:
        reply = llm.chat(text, history)
    except Exception as exc:  # noqa: BLE001 - surface provider errors to client
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc
    return JSONResponse({"reply": reply, "provider": llm.provider()})


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


@app.websocket("/ws/stt")
async def ws_stt(ws: WebSocket) -> None:
    """Realtime speech-to-text.

    Client streams raw 16 kHz mono PCM (int16, little-endian) as binary
    frames. Server replies with JSON text frames: {"partial": "..."} as the
    user speaks and {"final": "..."} when an utterance completes. A text
    frame "reset" starts a fresh recognizer for a new turn.
    """
    await ws.accept()
    recognizer = stt.StreamRecognizer()
    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes") is not None:
                result = recognizer.accept(msg["bytes"])
                await ws.send_json(result)
            elif msg.get("text") is not None:
                if msg["text"] == "reset":
                    recognizer = stt.StreamRecognizer()
                    await ws.send_json({"reset": True})
                elif msg["text"] == "flush":
                    await ws.send_json({"final": recognizer.final()})
    except WebSocketDisconnect:
        pass


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "assistant.html")


@app.get("/tools")
def tools() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
