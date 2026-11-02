"""Text-to-speech engines. Every engine implements `TTS`."""

import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 24_000


class TTS(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray:
        """Return mono float32 audio in [-1, 1] at `sample_rate`."""
        ...


class KokoroTTS:
    """Kokoro-82M running locally via ONNX. British voices: bm_george, bm_lewis, bm_daniel, bm_fable."""

    sample_rate = SAMPLE_RATE

    def __init__(self, model_path: str, voices_path: str, voice: str = "bm_george", speed: float = 1.0):
        for path in (model_path, voices_path):
            if not Path(path).exists():
                raise SystemExit(
                    f"Kokoro file not found: {path}\n"
                    "Download the model files with:  uv run python -m jarvis.speak.download"
                )
        from kokoro_onnx import Kokoro  # imported lazily: it's slow to import

        self._kokoro = Kokoro(model_path, voices_path)
        if voice not in self._kokoro.get_voices():
            raise SystemExit(f"Unknown Kokoro voice '{voice}'. Try: bm_george, bm_lewis, bm_daniel, bm_fable")
        self.voice = voice
        self.speed = speed
        self.lang = "en-gb" if voice.startswith("b") else "en-us"

    def synthesize(self, text: str) -> np.ndarray:
        audio, sr = self._kokoro.create(text, voice=self.voice, speed=self.speed, lang=self.lang)
        if sr != self.sample_rate:
            raise RuntimeError(f"Kokoro returned {sr} Hz, expected {self.sample_rate} Hz")
        return audio.astype(np.float32)


class SayTTS:
    """macOS built-in `say`. Zero setup, robotic. Good fallback."""

    sample_rate = SAMPLE_RATE

    def __init__(self, voice: str | None = "Daniel"):
        self.voice = voice

    def synthesize(self, text: str) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "speech.wav"
            cmd = ["say", "-o", str(out), "--file-format=WAVE", f"--data-format=LEI16@{self.sample_rate}"]
            if self.voice:
                cmd += ["-v", self.voice]
            subprocess.run([*cmd, text], check=True, capture_output=True)
            with wave.open(str(out), "rb") as w:
                frames = w.readframes(w.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
