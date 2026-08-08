"""Pluggable LLM backend.

Two providers, selected automatically:

1. OpenAI-compatible HTTP API  — used when ``OPENAI_API_KEY`` is set.
   Configurable via ``OPENAI_BASE_URL`` and ``LLM_MODEL``. This is the
   recommended production path and works with OpenAI, Azure, Groq,
   OpenRouter, Together, local Ollama (``/v1``), etc.

2. Local transformers model — offline fallback used when no API key is
   present, so the app works out of the box. Defaults to a small,
   CPU-friendly instruct model (``Qwen/Qwen2.5-0.5B-Instruct``).
"""
from __future__ import annotations

import os
import threading

SYSTEM_PROMPT = os.environ.get(
    "LLM_SYSTEM_PROMPT",
    "You are a helpful, concise voice assistant. Reply in 1-3 short "
    "sentences suitable for being read aloud. Do not use markdown, "
    "bullet points, or emojis.",
)

MAX_NEW_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "160"))

_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
_OPENAI_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
_OPENAI_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
_LOCAL_MODEL = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

_lock = threading.Lock()
_pipe = None  # lazy transformers pipeline


def provider() -> str:
    return "openai" if _OPENAI_KEY else "local"


def info() -> dict:
    return {
        "provider": provider(),
        "model": _OPENAI_MODEL if _OPENAI_KEY else _LOCAL_MODEL,
    }


def _openai_chat(messages: list[dict]) -> str:
    import requests

    resp = requests.post(
        f"{_OPENAI_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {_OPENAI_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": _OPENAI_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": MAX_NEW_TOKENS,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _load_local():
    global _pipe
    if _pipe is None:
        with _lock:
            if _pipe is None:
                print(f"[llm] loading local model '{_LOCAL_MODEL}' (first run may download)...")
                import torch
                from transformers import pipeline

                _pipe = pipeline(
                    "text-generation",
                    model=_LOCAL_MODEL,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                )
                print("[llm] local model ready")
    return _pipe


def _local_chat(messages: list[dict]) -> str:
    pipe = _load_local()
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    out = pipe(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    return out[0]["generated_text"].strip()


def chat(user_text: str, history: list[dict] | None = None) -> str:
    """Generate an assistant reply for ``user_text``.

    ``history`` is an optional list of prior ``{"role", "content"}`` turns.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    if _OPENAI_KEY:
        return _openai_chat(messages)
    return _local_chat(messages)
