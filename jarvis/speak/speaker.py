"""Speaks sentences in order. Synthesizes sentence N+1 while sentence N is playing."""

import queue
import threading
import time
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from jarvis.speak.tts import TTS

_BLOCK = 480  # 20 ms at 24 kHz: stop() is near-instant and the echo reference stays fine-grained
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
        # Called with every block just before it's played (feeds the echo canceller).
        self.on_play: Callable[[np.ndarray], None] | None = None
        self.volume = 1.0  # target loudness; changes are ramped over one block (no clicks)
        self.broken = False  # the audio device failed (e.g. unplugged); JARVIS should restart
        self._current_volume = 1.0
        self._gap = np.zeros(int(_GAP_SECONDS * tts.sample_rate), dtype=np.float32)
        self._out = sd.OutputStream(samplerate=tts.sample_rate, channels=1, dtype="float32", latency="low")
        self._out.start()
        threading.Thread(target=self._synth_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    # --- public API -------------------------------------------------------

    def say(self, text: str) -> None:
        with self._lock:
            self._pending += 1
        self._text_q.put((self._generation, text))

    def play(self, audio: np.ndarray) -> None:
        """Queue raw audio (e.g. an earcon) at the TTS sample rate, skipping synthesis."""
        with self._lock:
            self._pending += 1
        self._audio_q.put((self._generation, audio))

    def begin_turn(self) -> None:
        self.first_audio_at = None
        self.volume = 1.0

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

    def duck(self, level: float = 0.25) -> None:
        """Lower the voice (e.g. the user might be starting to talk)."""
        self.volume = level

    def unduck(self) -> None:
        self.volume = 1.0

    def _write(self, block: np.ndarray) -> None:
        target = self.volume
        if target != self._current_volume:
            block = block * np.linspace(self._current_volume, target, len(block), dtype=np.float32)
            self._current_volume = target
        elif target != 1.0:
            block = block * target
        if self.on_play is not None:
            self.on_play(block)
        self._out.write(block)

    def _play_loop(self) -> None:
        while True:
            gen, audio = self._audio_q.get()
            try:
                for i in range(0, len(audio), _BLOCK):
                    if gen != self._generation:
                        break
                    if self.first_audio_at is None:
                        self.first_audio_at = time.perf_counter()
                    self._write(audio[i : i + _BLOCK])
                else:
                    self._write(self._gap)
            except Exception as e:  # sounddevice.PortAudioError etc.
                if not self.broken:
                    print(f"\n[audio output failed: {e}]")
                self.broken = True
            finally:
                self._done_one()
