"""Voice activity detection: is someone speaking in this frame? (Silero VAD, on-device)."""

import numpy as np


class VoiceActivity:
    def __init__(self):
        from openwakeword.vad import VAD  # Silero VAD (ONNX), bundled with openWakeWord

        self._vad = VAD()

    def speech_probability(self, frame: np.ndarray) -> float:
        """0–1 probability that the 80 ms int16 frame contains speech."""
        return float(self._vad.predict(frame, frame_size=640))

    def reset(self) -> None:
        self._vad.reset_states()
