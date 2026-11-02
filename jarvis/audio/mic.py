"""Always-on microphone stream that hands out fixed 80 ms frames."""

import queue

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1_280  # 80 ms, the frame size openWakeWord is built around
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE


class MicStream:
    def __init__(self, sample_rate: int = SAMPLE_RATE, frame_samples: int = FRAME_SAMPLES):
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_samples,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status) -> None:
        self._q.put(indata[:, 0].copy())

    def start(self) -> None:
        self._stream.start()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        """Next 80 ms frame, or None if nothing arrived within `timeout`."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Throw away buffered audio (e.g. what the mic picked up while JARVIS was talking)."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return
