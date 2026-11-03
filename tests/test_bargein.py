import queue
import threading
import time

import numpy as np

from jarvis.audio.echo import EchoCanceller, EchoReference
from jarvis.listen.bargein import BargeInMonitor


class FakeMic:
    """Frames whose first sample encodes 'speech' (1) or 'silence' (0), delivered in ~real time."""

    def __init__(self, pattern, frame_delay=0.005):
        self.q = queue.Queue()
        for v in pattern:
            self.q.put(np.full(1280, v, dtype=np.int16))
        self.delay = frame_delay
        self.reads = 0

    def read(self, timeout=0.5):
        try:
            frame = self.q.get(timeout=min(timeout, 0.05))
        except queue.Empty:
            return None
        self.reads += 1
        time.sleep(self.delay)
        return frame


class FakeVAD:
    def speech_probability(self, frame):
        return 0.9 if frame[0] else 0.05

    def reset(self):
        pass


def wait_for(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_sustained_speech_triggers_and_keeps_recording():
    fired = threading.Event()
    pattern = [0] * 5 + [1] * 4 + [1, 0, 1]
    mon = BargeInMonitor(FakeMic(pattern), FakeVAD(), fired.set, preroll_seconds=0.5).start()
    assert wait_for(fired.is_set)
    assert wait_for(lambda: mon._mic.q.empty())
    time.sleep(0.05)
    values = [int(f[0][0]) for f in mon.stop()]
    # fires on the 3rd speech frame; keeps 0.5 s of pre-roll (6 frames) and everything after
    assert values == [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
    assert mon.triggered.is_set()


def test_a_dip_mid_speech_does_not_reset_the_count():
    """Echo cancellation dents the user's voice during double-talk; 3 of 4 still fires."""
    fired = threading.Event()
    mon = BargeInMonitor(FakeMic([0, 0, 1, 0, 1, 1, 0, 0]), FakeVAD(), fired.set).start()
    assert wait_for(fired.is_set)
    mon.stop()


def test_scattered_blips_never_trigger():
    fired = threading.Event()
    pattern = [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0] * 3  # never 3 speech frames within any 4
    mon = BargeInMonitor(FakeMic(pattern), FakeVAD(), fired.set).start()
    assert wait_for(lambda: mon._mic.q.empty())
    time.sleep(0.05)
    assert mon.stop() == []
    assert not fired.is_set()


def test_ducks_on_first_speech_and_restores_after_quiet():
    events = []
    pattern = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # two isolated blips
    mon = BargeInMonitor(
        FakeMic(pattern), FakeVAD(), lambda: events.append("stop"),
        on_maybe=lambda: events.append("duck"), on_clear=lambda: events.append("unduck"),
    ).start()
    assert wait_for(lambda: mon._mic.q.empty())
    time.sleep(0.05)
    mon.stop()
    assert events == ["duck", "unduck", "duck", "unduck"]


def test_pause_stops_reading_so_someone_else_can():
    mic = FakeMic([0] * 200, frame_delay=0.002)
    mon = BargeInMonitor(mic, FakeVAD(), lambda: None).start()
    assert wait_for(lambda: mic.reads > 3)
    mon.pause()
    reads = mic.reads
    time.sleep(0.1)
    assert mic.reads == reads  # nothing read while paused
    mon.resume()
    assert wait_for(lambda: mic.reads > reads)
    mon.stop()


def test_echo_reference_resamples_and_pads():
    ref = EchoReference(source_rate=24_000)
    ref.push(np.full(2400, 0.5, dtype=np.float32))  # 100 ms at 24 kHz
    out = ref.pull(1600)  # 100 ms at 16 kHz
    assert out.dtype == np.int16 and len(out) == 1600
    assert abs(int(np.median(out)) - 16383) < 200
    assert not ref.pull(160).any()  # nothing left: silence


def test_echo_reference_caps_memory():
    ref = EchoReference(source_rate=16_000, max_seconds=0.5)
    for _ in range(20):
        ref.push(np.zeros(1600, dtype=np.float32))
    assert len(ref._buf) == 8000


def test_echo_canceller_removes_a_pure_echo():
    """Mic hears only a delayed copy of what's played: after convergence it should be ~silent."""
    rng = np.random.default_rng(0)
    played = (rng.normal(0, 0.1, 16_000 * 4)).astype(np.float32)
    # make it speech-like: amplitude-modulated noise
    played *= (0.5 + 0.5 * np.sin(np.linspace(0, 40, len(played)))).astype(np.float32)
    ref = EchoReference(source_rate=16_000)
    aec = EchoCanceller(ref, delay_ms=20)
    delay = 320  # 20 ms
    heard = np.concatenate([np.zeros(delay, np.float32), played])[: len(played)] * 0.5
    out = []
    for i in range(0, len(played), 1280):
        ref.push(played[i : i + 1280])
        out.append(aec.process((heard[i : i + 1280] * 32767).astype(np.int16)))
    out = np.concatenate(out).astype(np.float32)
    before = np.sqrt(np.mean((heard[-16_000:] * 32767) ** 2))
    after = np.sqrt(np.mean(out[-16_000:] ** 2))
    assert after < before * 0.1  # at least 20 dB of echo removed once converged
