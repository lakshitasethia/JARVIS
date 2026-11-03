"""Barge-in: notice the user talking over JARVIS, so it can stop and listen."""

import threading
from collections import deque
from collections.abc import Callable

import numpy as np

from jarvis import log
from jarvis.audio.mic import FRAME_SECONDS

Frame = tuple[np.ndarray, float]  # (audio, speech probability)


class BargeInMonitor:
    """Reads the (echo-cancelled) mic on a background thread while JARVIS responds.

    Two stages, so it *feels* instant without being trigger-happy:
      1. The first speech-like frame (~80 ms) calls `on_maybe`: JARVIS ducks its volume.
         If it turns out to be nothing, `on_clear` restores it.
      2. `trigger_frames` of the last `window_frames` frames being speech calls `on_barge_in`:
         JARVIS stops. (Not "N in a row": while both are talking, echo cancellation dents
         the user's voice now and then, and a single dip shouldn't reset the count.)
    After triggering it keeps recording so nothing the user said is lost; stop() returns it.
    Only one thread may read the mic at a time: call pause()/resume() around any other reader.
    """

    def __init__(
        self,
        mic,
        vad,
        on_barge_in: Callable[[], None],
        on_maybe: Callable[[], None] = lambda: None,
        on_clear: Callable[[], None] = lambda: None,
        threshold: float = 0.5,
        trigger_frames: int = 3,
        window_frames: int = 4,
        clear_after_frames: int = 3,
        # Keep this much audio from before the trigger: a short "Stop." or "No," followed by a
        # pause often doesn't trigger on its own, and we don't want to lose it.
        preroll_seconds: float = 1.5,
    ):
        self._mic, self._vad = mic, vad
        self._on_barge_in, self._on_maybe, self._on_clear = on_barge_in, on_maybe, on_clear
        self._threshold = threshold
        self._trigger_frames = trigger_frames
        self._window: deque[float] = deque(maxlen=window_frames)
        self._clear_after = clear_after_frames
        self._recent: deque[Frame] = deque(maxlen=max(window_frames, round(preroll_seconds / FRAME_SECONDS)))
        self._captured: list[Frame] = []
        self.triggered = threading.Event()
        self._stopping = threading.Event()
        self._paused = threading.Event()
        self._reading = threading.Lock()  # held while this thread is using the mic/VAD
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "BargeInMonitor":
        self._vad.reset()
        self._thread.start()
        return self

    def pause(self) -> None:
        """Stop reading the mic (returns once the monitor thread has let go of it)."""
        self._paused.set()
        with self._reading:
            pass

    def resume(self) -> None:
        self._recent.clear()
        self._window.clear()
        self._paused.clear()

    def stop(self) -> list[Frame]:
        """Stop monitoring. Returns the user's audio if they barged in, else []."""
        self._stopping.set()
        self._thread.join(timeout=2)
        return self._captured if self.triggered.is_set() else []

    def _loop(self) -> None:
        ducked = False
        quiet = 0
        while not self._stopping.is_set():
            if self._paused.is_set():
                self._stopping.wait(0.05)
                continue
            with self._reading:
                if self._paused.is_set():
                    continue
                frame = self._mic.read(timeout=0.1)
                if frame is None:
                    continue
                prob = self._vad.speech_probability(frame)
            if self.triggered.is_set():
                self._captured.append((frame, prob))
                continue

            self._recent.append((frame, prob))
            self._window.append(prob)
            speechy = prob >= self._threshold
            quiet = 0 if speechy else quiet + 1

            if sum(p >= self._threshold for p in self._window) >= self._trigger_frames:
                self._captured = list(self._recent)
                self.triggered.set()
                log.event("barge_in", recent_probs=[round(p, 2) for _, p in list(self._recent)[-8:]])
                self._on_barge_in()
            elif speechy and not ducked:
                ducked = True
                self._on_maybe()
            elif ducked and quiet >= self._clear_after:
                ducked = False
                self._on_clear()
