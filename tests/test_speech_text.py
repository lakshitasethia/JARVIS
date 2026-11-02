import io
import wave

import numpy as np

from jarvis.audio.recorder import duration, peak_level, to_wav_bytes
from jarvis.speak.sentences import SentenceSplitter, clean_for_speech


def feed_all(chunks, **kw):
    s = SentenceSplitter(**kw)
    out = []
    for c in chunks:
        out += s.feed(c)
    return out, s.flush()


def test_sentences_emitted_as_soon_as_complete():
    s = SentenceSplitter()
    assert s.feed("Good eve") == []
    assert s.feed("ning, sir. All sys") == ["Good evening, sir."]
    assert s.feed("tems online! Ready?") == ["All systems online!"]
    assert s.flush() == ["Ready?"]
    assert s.flush() == []


def test_decimals_are_not_sentence_breaks():
    out, rest = feed_all(["It is 21.5 degrees. ", "Nice."])
    assert out == ["It is 21.5 degrees."]
    assert rest == ["Nice."]


def test_newlines_break_sentences():
    out, rest = feed_all(["First line\nSecond line"])
    assert out == ["First line"]
    assert rest == ["Second line"]


def test_long_run_on_text_is_split_at_a_comma_or_space():
    text = "word, " * 60  # 360 chars, no sentence end
    out, rest = feed_all([text], max_chars=100)
    assert out and all(len(s) <= 101 for s in out)
    assert " ".join(out + rest).replace(" ", "") == text.replace(" ", "").rstrip()


def test_clean_for_speech_strips_markdown_and_urls():
    assert clean_for_speech("**Sure!** See https://example.com/x") == "Sure! See a link"
    assert clean_for_speech("- item one") == "item one"
    assert clean_for_speech("# Heading `code`") == "Heading code"


def test_audio_helpers():
    samples = np.array([0, 1000, -2000, 500], dtype=np.int16)
    assert peak_level(samples) == 2000
    assert peak_level(np.zeros(0, dtype=np.int16)) == 0
    assert duration(np.zeros(8000, dtype=np.int16)) == 0.5
    with wave.open(io.BytesIO(to_wav_bytes(samples)), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (16000, 1, 4)
