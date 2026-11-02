import numpy as np

from jarvis.listen.endpoint import EndpointConfig, Status, UtteranceCollector
from jarvis.main import is_sleep_phrase
from jarvis.speak import earcons

FRAME = 0.08


def frame(value: int) -> np.ndarray:
    return np.full(1280, value, dtype=np.int16)


def run(probs, **cfg):
    """Feed a list of per-frame speech probabilities; return statuses and the collector."""
    c = UtteranceCollector(EndpointConfig(**cfg))
    statuses = [c.add(frame(i), p) for i, p in enumerate(probs)]
    return statuses, c


def test_speech_then_silence_is_done():
    probs = [0.0] * 5 + [0.9] * 10 + [0.0] * 10  # 0.4 s quiet, 0.8 s speech, 0.8 s quiet
    statuses, c = run(probs, silence_seconds=0.8)
    assert statuses[4] is Status.WAITING
    assert statuses[5] is Status.SPEAKING
    assert statuses[-1] is Status.DONE
    assert statuses[-2] is Status.SPEAKING  # one frame short of 0.8 s silence
    assert abs(c.trailing_silence_seconds - 0.8) < 1e-9


def test_audio_includes_preroll_and_tail():
    probs = [0.0] * 5 + [0.9] * 10 + [0.0] * 10
    _, c = run(probs, silence_seconds=0.8, preroll_seconds=0.24)
    audio = c.audio()
    first, last = int(audio[0]), int(audio[-1])
    assert first == 2  # speech starts at frame 5, pre-roll of 3 frames → frame 2
    assert last == 16  # last speech frame 14 + 2-frame tail
    assert len(audio) == (16 - 2 + 1) * 1280


def test_no_speech_times_out():
    statuses, c = run([0.0] * 20, no_speech_timeout=1.6)
    assert statuses[-1] is Status.TIMEOUT
    assert statuses[-2] is Status.WAITING
    assert len(c.audio()) == 0


def test_short_blip_is_ignored():
    probs = [0.9] * 2 + [0.0] * 12  # 0.16 s "cough", then quiet
    statuses, _ = run(probs, silence_seconds=0.8, min_speech_seconds=0.24, no_speech_timeout=10)
    assert Status.DONE not in statuses
    assert statuses[-1] is Status.WAITING


def test_brief_pauses_mid_sentence_dont_end_the_turn():
    probs = [0.9] * 5 + [0.0] * 5 + [0.9] * 5 + [0.0] * 10  # 0.4 s pause in the middle
    statuses, _ = run(probs, silence_seconds=0.8)
    assert statuses.index(Status.DONE) == len(probs) - 1


def test_max_length_cuts_off_monologue():
    statuses, _ = run([0.9] * 50, max_seconds=2.0)
    assert Status.DONE in statuses
    assert statuses.index(Status.DONE) == round(2.0 / FRAME) - 1


def test_sleep_phrases():
    for text in ["Never mind.", "Go to sleep", "Jarvis, stop.", "Hey Jarvis, that's all", "Cancel!"]:
        assert is_sleep_phrase(text), text
    for text in ["Stop the music in five minutes", "What's the time?", "Don't stop"]:
        assert not is_sleep_phrase(text), text


def test_earcons_are_short_and_quiet():
    for make in (earcons.listening, earcons.sleeping):
        audio = make(24_000)
        assert audio.dtype == np.float32
        assert 0.1 < len(audio) / 24_000 < 0.3
        assert np.abs(audio).max() <= 0.25 + 1e-6
        assert abs(audio[0]) < 0.01 and abs(audio[-1]) < 0.01  # faded, no click
