"""Short UI sounds: 'I'm listening' and 'going back to sleep'."""

import numpy as np


def _tone(freq: float, seconds: float, sample_rate: int, volume: float) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    wave = np.sin(2 * np.pi * freq * t)
    fade = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)  # 10 ms fade in/out: no clicks
    return (volume * wave * fade).astype(np.float32)


def listening(sample_rate: int) -> np.ndarray:
    """Two quick rising notes."""
    return np.concatenate([_tone(880, 0.07, sample_rate, 0.25), _tone(1320, 0.09, sample_rate, 0.25)])


def sleeping(sample_rate: int) -> np.ndarray:
    """Two soft falling notes."""
    return np.concatenate([_tone(990, 0.07, sample_rate, 0.15), _tone(660, 0.1, sample_rate, 0.15)])
