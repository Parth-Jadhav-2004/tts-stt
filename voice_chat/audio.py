from __future__ import annotations

import numpy as np
import sounddevice as sd


def play_audio(wav: np.ndarray, sample_rate: int) -> None:
    """Play a mono/stereo float or int tensor/array blocking until done."""
    audio = np.asarray(wav)
    if audio.ndim > 1:
        # CosyVoice yields [1, T] — sounddevice wants [T] or [T, C]
        if audio.shape[0] == 1:
            audio = audio[0]
        elif audio.shape[1] == 1:
            audio = audio[:, 0]
        else:
            audio = audio.T
    audio = audio.astype(np.float32, copy=False)
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    sd.play(audio, sample_rate)
    sd.wait()
