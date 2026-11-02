"""Wake word detection ("Hey Jarvis") with openWakeWord, fully on-device."""

import os

import numpy as np

_SILENCE = np.zeros(1_280, dtype=np.int16)


def bundled_model_path(name: str = "hey_jarvis_v0.1") -> str:
    import openwakeword

    return os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models", f"{name}.onnx")


class WakeWordDetector:
    def __init__(self, threshold: float = 0.5, model_path: str | None = None):
        from openwakeword.model import Model  # slow import, keep it lazy

        self.threshold = threshold
        self._model = Model(wakeword_model_paths=[model_path or bundled_model_path()])
        self._name = next(iter(self._model.models))

    def score(self, frame: np.ndarray) -> float:
        """Feed one 80 ms int16 frame; returns 0–1 confidence that the wake word just ended."""
        return float(self._model.predict(frame)[self._name])

    def detected(self, frame: np.ndarray) -> bool:
        return self.score(frame) >= self.threshold

    def reset(self) -> None:
        """Clear internal audio history so the last 'Hey Jarvis' can't re-trigger."""
        self._model.reset()
        for _ in range(20):  # ~1.6 s of silence flushes the feature buffer
            self._model.predict(_SILENCE)
        self._model.reset()
