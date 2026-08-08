from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from voice_chat.audio import play_audio
from voice_chat.config import Config


class TextToSpeech:
    """CosyVoice2-0.5B wrapper tuned for ~6GB VRAM (FP16, no TensorRT/vLLM)."""

    def __init__(self, config: Config):
        self._config = config
        self._model = None
        self._sample_rate = 24000
        self._spk_id = "voice_chat_spk"

    def load(self) -> None:
        repo = self._config.cosyvoice_repo_dir
        if not repo.exists():
            raise SystemExit(
                f"CosyVoice repo not found at {repo}.\n"
                "Run: bash scripts/setup_cosyvoice.sh"
            )

        matcha = repo / "third_party" / "Matcha-TTS"
        for path in (str(repo), str(matcha)):
            if path not in sys.path:
                sys.path.insert(0, path)

        prompt = self._config.cosyvoice_prompt_wav
        if not prompt.exists():
            raise SystemExit(
                f"Voice prompt wav missing: {prompt}\n"
                "Run: bash scripts/setup_cosyvoice.sh"
            )

        model_dir = self._config.cosyvoice_model_dir
        if not model_dir.exists():
            raise SystemExit(
                f"CosyVoice2 model missing: {model_dir}\n"
                "Run: bash scripts/setup_cosyvoice.sh"
            )

        # Free as much fragmentation as possible before loading TTS
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore

        fp16 = self._config.cosyvoice_fp16 and torch.cuda.is_available()
        self._model = AutoModel(
            model_dir=str(model_dir),
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=fp16,
        )
        self._sample_rate = int(self._model.sample_rate)

        # Cache speaker embedding once so each reply is cheaper
        ok = self._model.add_zero_shot_spk(
            self._config.cosyvoice_prompt_text,
            str(prompt),
            self._spk_id,
        )
        if not ok:
            raise RuntimeError("Failed to register CosyVoice zero-shot speaker")

    def synthesize(self, text: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TTS not loaded. Call load() first.")
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

        chunks: list[np.ndarray] = []
        for out in self._model.inference_zero_shot(
            text,
            "",
            "",
            zero_shot_spk_id=self._spk_id,
            stream=False,
        ):
            speech = out["tts_speech"]
            if isinstance(speech, torch.Tensor):
                speech = speech.detach().cpu().float().numpy()
            chunks.append(np.asarray(speech, dtype=np.float32))

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate([c.reshape(-1) for c in chunks], axis=0)
        return audio

    def speak(self, text: str) -> None:
        audio = self.synthesize(text)
        if audio.size == 0:
            return
        play_audio(audio, self._sample_rate)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def unload(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
