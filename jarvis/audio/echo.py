"""Acoustic echo cancellation: remove JARVIS's own voice from the microphone signal.

The speaker reports every block it plays to `EchoReference`; the mic pipeline asks
`EchoCanceller` to subtract that reference from what the microphone hears (WebRTC AEC3,
via the livekit package). This is what lets you talk over JARVIS on laptop speakers.
"""

import threading
from math import gcd

import numpy as np
from scipy.signal import resample_poly

MIC_RATE = 16_000
_CHUNK = 160  # WebRTC processes exactly 10 ms at a time


class EchoReference:
    """Thread-safe FIFO of what's being played, resampled to the mic's 16 kHz."""

    def __init__(self, source_rate: int, max_seconds: float = 2.0):
        g = gcd(MIC_RATE, source_rate)
        self._up, self._down = MIC_RATE // g, source_rate // g
        self._max = int(max_seconds * MIC_RATE)
        self._buf = np.zeros(0, dtype=np.int16)
        self._lock = threading.Lock()

    def push(self, audio: np.ndarray) -> None:
        """Called by the speaker with float32 audio at `source_rate`, just before playing it."""
        if self._up != self._down:
            audio = resample_poly(audio, self._up, self._down)
        samples = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with self._lock:
            self._buf = np.concatenate([self._buf, samples])[-self._max :]

    def pull(self, n: int) -> np.ndarray:
        """Next n samples of reference audio (silence if nothing is playing)."""
        with self._lock:
            out, self._buf = self._buf[:n], self._buf[n:]
        return out if len(out) == n else np.pad(out, (0, n - len(out)))


class EchoCanceller:
    def __init__(self, reference: EchoReference, delay_ms: int = 60):
        from livekit import rtc  # native WebRTC audio processing

        self._rtc = rtc
        self._reference = reference
        self._apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=True, high_pass_filter=True)
        self._apm.set_stream_delay_ms(delay_ms)  # a hint; AEC3 estimates the real delay itself

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Echo-cancel one int16 mic frame (any multiple of 10 ms)."""
        rtc = self._rtc
        out = np.empty_like(frame)
        for i in range(0, len(frame) - _CHUNK + 1, _CHUNK):
            far = rtc.AudioFrame(self._reference.pull(_CHUNK).tobytes(), MIC_RATE, 1, _CHUNK)
            self._apm.process_reverse_stream(far)
            near = rtc.AudioFrame(frame[i : i + _CHUNK].tobytes(), MIC_RATE, 1, _CHUNK)
            self._apm.process_stream(near)
            out[i : i + _CHUNK] = np.frombuffer(bytes(near.data.cast("B")), dtype=np.int16)
        return out
