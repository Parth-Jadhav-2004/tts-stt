from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from moonshine_voice import MicTranscriber, ModelArch, string_to_model_arch

from voice_chat.config import Config

ARCH_ALIASES = {
    "tiny": "tiny_streaming",
    "tiny_streaming": "tiny_streaming",
    "small": "small_streaming",
    "small_streaming": "small_streaming",
    "medium": "medium_streaming",
    "medium_streaming": "medium_streaming",
    "base": "base_streaming",
    "base_streaming": "base_streaming",
}


class SpeechToText:
    """Moonshine microphone STT with turn completion callbacks."""

    def __init__(
        self,
        config: Config,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
    ):
        self._on_partial = on_partial
        self._on_final = on_final
        self._debounce_s = max(config.turn_debounce_ms, 0) / 1000.0
        self._pending: Optional[str] = None
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

        arch_name = ARCH_ALIASES.get(
            config.moonshine_model_arch, config.moonshine_model_arch
        )
        try:
            arch: ModelArch = string_to_model_arch(arch_name)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"Unknown MOONSHINE_MODEL_ARCH={config.moonshine_model_arch!r}. "
                f"Use tiny_streaming, small_streaming, or medium_streaming."
            ) from exc

        self._mic = (
            MicTranscriber()
            .language(config.moonshine_language)
            .model_arch(arch)
            .on_text(self._handle_partial)
            .on_line(self._handle_line)
        )

    def load(self) -> None:
        # CPU by default — leaves the 6GB GPU for CosyVoice
        self._mic.load()

    def start(self) -> None:
        self._mic.start()

    def mute(self, muted: bool = True) -> None:
        self._mic.mute(muted)

    def stop(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self._mic.stop()

    def close(self) -> None:
        self.stop()
        self._mic.close()

    def _handle_partial(self, text: str) -> None:
        if self._on_partial:
            self._on_partial(text)

    def _handle_line(self, line) -> None:
        text = (getattr(line, "text", None) or str(line) or "").strip()
        if not text:
            return
        with self._lock:
            self._pending = text
            if self._timer is not None:
                self._timer.cancel()
            if self._debounce_s <= 0:
                utterance = self._pending
                self._pending = None
                callback = self._on_final
            else:
                self._timer = threading.Timer(self._debounce_s, self._emit_pending)
                self._timer.daemon = True
                self._timer.start()
                return
        if callback:
            callback(utterance)

    def _emit_pending(self) -> None:
        with self._lock:
            utterance = self._pending
            self._pending = None
            self._timer = None
            callback = self._on_final
        if utterance and callback:
            # Tiny yield so mute/UI updates settle before the turn pipeline runs
            time.sleep(0.01)
            callback(utterance)
