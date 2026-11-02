"""Speaks sentences in order. Synthesizes sentence N+1 while sentence N is playing."""

import queue
import threading
import time

import numpy as np
import sounddevice as sd

from jarvis.speak.tts import TTS

_BLOCK = 2_400  # 0.1 s at 24 kHz, so stop() takes effect within ~0.1 s
_GAP_SECONDS = 0.12  # small breath between sentences


class Speaker:
    def __init__(self, tts: TTS):
        self.tts = tts
        self._text_q: queue.Queue = queue.Queue()
        self._audio_q: queue.Queue = queue.Queue()
        self._generation = 0  # bumped by stop(); stale work from older generations is dropped
        self._pending = 0
        self._lock = threading.Lock()
        self.first_audio_at: float | None = None
        self._gap = np.zeros(int(_GAP_SECONDS * tts.sample_rate), dtype=np.float32)
        self._out = sd.OutputStream(samplerate=tts.sample_rate, channels=1, dtype="float32")
        self._out.start()
        threading.Thread(target=self._synth_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    # --- public API -------------------------------------------------------

    def say(self, text: str) -> None:
        with self._lock:
            self._pending += 1
        self._text_q.put((self._generation, text))

    def begin_turn(self) -> None:
        self.first_audio_at = None

    def stop(self) -> None:
        """Silence immediately and discard anything queued."""
        self._generation += 1

    def is_speaking(self) -> bool:
        with self._lock:
            return self._pending > 0

    def wait(self) -> None:
        """Block until everything queued has been spoken (Ctrl+C friendly)."""
        while self.is_speaking():
            time.sleep(0.03)

    def warm_up(self) -> None:
        """First synthesis is slow; do it in the background at startup."""
        threading.Thread(target=lambda: self.tts.synthesize("Ready."), daemon=True).start()

    def close(self) -> None:
        self.stop()
        self._out.stop()
        self._out.close()

    # --- workers ------------------------------------------------------------

    def _done_one(self) -> None:
        with self._lock:
            self._pending -= 1

    def _synth_loop(self) -> None:
        while True:
            gen, text = self._text_q.get()
            if gen != self._generation:
                self._done_one()
                continue
            try:
                audio = self.tts.synthesize(text)
            except Exception as e:  # never let one bad sentence kill the voice
                print(f"\n[tts error: {e}]")
                self._done_one()
                continue
            self._audio_q.put((gen, audio))

    def _play_loop(self) -> None:
        while True:
            gen, audio = self._audio_q.get()
            try:
                for i in range(0, len(audio), _BLOCK):
                    if gen != self._generation:
                        break
                    if self.first_audio_at is None:
                        self.first_audio_at = time.perf_counter()
                    self._out.write(audio[i : i + _BLOCK])
                else:
                    self._out.write(self._gap)
            finally:
                self._done_one()
