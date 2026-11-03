"""Always-on microphone stream that hands out fixed 80 ms frames."""

import queue
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1_280  # 80 ms, the frame size openWakeWord is built around
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE


class MicStream:
    """Mic → (optional processor, e.g. echo cancellation) → frames for read().

    The processor runs on its own thread as audio arrives, so it stays in step with
    real time even when the main thread is busy (waiting on the LLM, etc.).
    """

    def __init__(
        self,
        processor: Callable[[np.ndarray], np.ndarray] | None = None,
        sample_rate: int = SAMPLE_RATE,
        frame_samples: int = FRAME_SAMPLES,
    ):
        self._processor = processor
        self._raw: queue.Queue[np.ndarray | None] = queue.Queue()
        self._out: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_samples,
            callback=self._callback,
        )
        self._worker = threading.Thread(target=self._process_loop, daemon=True) if processor else None

    def _callback(self, indata, frames, time_info, status) -> None:
        (self._raw if self._processor else self._out).put(indata[:, 0].copy())

    def _process_loop(self) -> None:
        while (frame := self._raw.get()) is not None:
            self._out.put(self._processor(frame))

    def start(self) -> None:
        if self._worker:
            self._worker.start()
        self._stream.start()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
        if self._worker:
            self._raw.put(None)

    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        """Next 80 ms frame, or None if nothing arrived within `timeout`."""
        try:
            return self._out.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Throw away buffered audio (e.g. what the mic picked up while JARVIS was talking)."""
        while True:
            try:
                self._out.get_nowait()
            except queue.Empty:
                return
