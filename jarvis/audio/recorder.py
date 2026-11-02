"""Microphone capture (push-to-talk) and small audio helpers."""

import io
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000  # what speech-to-text models expect


class Recorder:
    """Records mono 16 kHz audio between start() and stop()."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, max_seconds: float = 60.0):
        self.sample_rate = sample_rate
        self._max_frames = int(max_seconds * sample_rate)
        self._chunks: list[np.ndarray] = []
        self._frames = 0
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time_info, status) -> None:
        if self._frames < self._max_frames:
            self._chunks.append(indata[:, 0].copy())
            self._frames += frames

    def start(self) -> None:
        self._chunks, self._frames = [], 0
        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="int16", callback=self._callback
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self._chunks)


def duration(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return len(samples) / sample_rate


def peak_level(samples: np.ndarray) -> int:
    """Loudest sample (0–32767). Near-zero means silence or no mic permission."""
    return int(np.abs(samples.astype(np.int32)).max()) if len(samples) else 0


def to_wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # int16
        w.setframerate(sample_rate)
        w.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()
