import threading
import time

import numpy as np
import pytest

import jarvis.speak.speaker as speaker_mod
from jarvis.speak.speaker import Speaker


class FakeStream:
    def __init__(self, **kw):
        self.written: list[np.ndarray] = []

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass

    def write(self, block):
        self.written.append(block)
        time.sleep(0.001)


class FakeTTS:
    """Encodes each sentence as a constant-valued signal so we can check play order."""

    sample_rate = 24_000

    def __init__(self, gate: threading.Event | None = None):
        self.gate = gate
        self.ids = {"one": 0.1, "two": 0.2, "three": 0.3}

    def synthesize(self, text):
        if self.gate:
            self.gate.wait(2)
        return np.full(4_800, self.ids.get(text, 0.9), dtype=np.float32)


@pytest.fixture
def fake_stream(monkeypatch):
    monkeypatch.setattr(speaker_mod.sd, "OutputStream", FakeStream)


def played_values(sp: Speaker) -> list[float]:
    vals = [round(float(b[0]), 1) for b in sp._out.written if len(b) and b[0] != 0]
    return [v for i, v in enumerate(vals) if i == 0 or vals[i - 1] != v]


def test_sentences_play_in_order(fake_stream):
    sp = Speaker(FakeTTS())
    for t in ("one", "two", "three"):
        sp.say(t)
    sp.wait()
    assert played_values(sp) == [0.1, 0.2, 0.3]
    assert sp.first_audio_at is not None
    assert not sp.is_speaking()


def test_stop_discards_queued_sentences(fake_stream):
    gate = threading.Event()
    sp = Speaker(FakeTTS(gate))
    for t in ("one", "two", "three"):
        sp.say(t)
    sp.stop()  # before anything was synthesized
    gate.set()
    sp.wait()
    assert played_values(sp) == []
    sp.say("one")  # still usable afterwards
    sp.wait()
    assert played_values(sp) == [0.1]
