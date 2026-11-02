"""Decide when the user has started and finished talking (pure logic, no audio I/O)."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from jarvis.audio.mic import FRAME_SECONDS


class Status(Enum):
    WAITING = "waiting"  # no speech yet
    SPEAKING = "speaking"  # user is talking
    DONE = "done"  # user finished; audio() is ready
    TIMEOUT = "timeout"  # nobody said anything


@dataclass(frozen=True)
class EndpointConfig:
    speech_threshold: float = 0.5  # VAD probability that counts as speech
    silence_seconds: float = 0.8  # this much quiet after speech = you're done
    no_speech_timeout: float = 6.0  # give up if nothing is said for this long
    max_seconds: float = 30.0  # hard cap on one utterance
    min_speech_seconds: float = 0.24  # shorter blips (cough, click) are ignored
    preroll_seconds: float = 0.3  # keep a little audio before speech started
    frame_seconds: float = FRAME_SECONDS


class UtteranceCollector:
    def __init__(self, config: EndpointConfig = EndpointConfig()):
        self.c = config
        self._frames: list[np.ndarray] = []
        self._start: int | None = None  # index of first speech frame
        self._last_speech: int = -1
        self._speech_frames = 0

    def _seconds(self, n_frames: int) -> float:
        return n_frames * self.c.frame_seconds

    @property
    def trailing_silence_seconds(self) -> float:
        return self._seconds(len(self._frames) - 1 - self._last_speech) if self._start is not None else 0.0

    def add(self, frame: np.ndarray, speech_prob: float) -> Status:
        self._frames.append(frame)
        i = len(self._frames) - 1
        if speech_prob >= self.c.speech_threshold:
            if self._start is None:
                self._start = i
            self._last_speech = i
            self._speech_frames += 1

        elapsed = self._seconds(len(self._frames))
        if self._start is None:
            return Status.TIMEOUT if elapsed >= self.c.no_speech_timeout else Status.WAITING

        finished = (
            self.trailing_silence_seconds >= self.c.silence_seconds
            or self._seconds(i - self._start + 1) >= self.c.max_seconds
        )
        if not finished:
            return Status.SPEAKING
        if self._seconds(self._speech_frames) >= self.c.min_speech_seconds:
            return Status.DONE
        # Just a blip. Forget it and keep waiting.
        self._start, self._last_speech, self._speech_frames = None, -1, 0
        return Status.TIMEOUT if elapsed >= self.c.no_speech_timeout else Status.WAITING

    def audio(self) -> np.ndarray:
        """The utterance with a little pre-roll and tail, as one int16 array."""
        if self._start is None:
            return np.zeros(0, dtype=np.int16)
        pre = round(self.c.preroll_seconds / self.c.frame_seconds)
        tail = 2  # ~160 ms after the last speech frame
        return np.concatenate(self._frames[max(0, self._start - pre) : self._last_speech + 1 + tail])
