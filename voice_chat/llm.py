from __future__ import annotations

from typing import Iterable

from openai import OpenAI

from voice_chat.config import Config


class LLMClient:
    """Thin OpenAI-compatible chat client."""

    def __init__(self, config: Config):
        self._model = config.llm_model
        self._system = config.llm_system_prompt
        self._history: list[dict[str, str]] = [
            {"role": "system", "content": self._system}
        ]
        self._client = OpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        )

    def reset(self) -> None:
        self._history = [{"role": "system", "content": self._system}]

    def chat(self, user_text: str) -> str:
        self._history.append({"role": "user", "content": user_text})
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._history,
            temperature=0.7,
        )
        reply = (response.choices[0].message.content or "").strip()
        self._history.append({"role": "assistant", "content": reply})
        # Keep context bounded for a long voice session
        if len(self._history) > 21:  # system + 10 turns
            self._history = [self._history[0]] + self._history[-20:]
        return reply

    def chat_stream(self, user_text: str) -> Iterable[str]:
        """Optional streaming helper (unused by default voice loop)."""
        self._history.append({"role": "user", "content": user_text})
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=self._history,
            temperature=0.7,
            stream=True,
        )
        chunks: list[str] = []
        for event in stream:
            delta = event.choices[0].delta.content or ""
            if delta:
                chunks.append(delta)
                yield delta
        reply = "".join(chunks).strip()
        self._history.append({"role": "assistant", "content": reply})
