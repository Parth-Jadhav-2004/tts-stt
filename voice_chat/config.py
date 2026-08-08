from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_system_prompt: str

    moonshine_model_arch: str
    moonshine_language: str

    cosyvoice_model_dir: Path
    cosyvoice_repo_dir: Path
    cosyvoice_prompt_wav: Path
    cosyvoice_prompt_text: str
    cosyvoice_fp16: bool

    turn_debounce_ms: int

    @classmethod
    def load(cls, env_file: str | Path | None = None) -> "Config":
        load_dotenv(env_file or ROOT / ".env", override=env_file is not None)
        def path_from_env(key: str, default: str) -> Path:
            raw = os.getenv(key, default)
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = ROOT / path
            return path

        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "LLM_API_KEY is missing. Copy .env.example to .env and set your key."
            )

        return cls(
            llm_api_key=api_key,
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip(
                "/"
            ),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            llm_system_prompt=os.getenv(
                "LLM_SYSTEM_PROMPT",
                "You are a helpful voice assistant. Keep answers concise and "
                "conversational, usually 1-3 short sentences, since they will "
                "be spoken aloud.",
            ),
            moonshine_model_arch=os.getenv(
                "MOONSHINE_MODEL_ARCH", "small_streaming"
            ).lower(),
            moonshine_language=os.getenv("MOONSHINE_LANGUAGE", "en"),
            cosyvoice_model_dir=path_from_env(
                "COSYVOICE_MODEL_DIR", "pretrained_models/CosyVoice2-0.5B"
            ),
            cosyvoice_repo_dir=path_from_env(
                "COSYVOICE_REPO_DIR", "third_party/CosyVoice"
            ),
            cosyvoice_prompt_wav=path_from_env(
                "COSYVOICE_PROMPT_WAV", "assets/prompt.wav"
            ),
            cosyvoice_prompt_text=os.getenv(
                "COSYVOICE_PROMPT_TEXT",
                "希望你以后能够做的比我还好呦。",
            ),
            cosyvoice_fp16=_bool(os.getenv("COSYVOICE_FP16"), True),
            turn_debounce_ms=int(os.getenv("TURN_DEBOUNCE_MS", "400")),
        )
