"""Speech-to-text. Every engine implements `STT`."""

import re
from typing import Protocol

from groq import Groq

# Phrases Whisper is known to invent from silence/noise (learned from YouTube subtitles).
# Nobody says these to an assistant, so they're always dropped.
_HALLUCINATIONS = {
    "thanks for watching", "thank you for watching", "thanks for watching and see you next time",
    "please subscribe", "like and subscribe", "subtitles by the amaraorg community",
    "transcribed by", "you",
}


def is_junk_transcript(text: str) -> bool:
    """True for transcripts that aren't real speech: '...', '.', '♪', known Whisper hallucinations."""
    letters = re.sub(r"[^a-z ]", "", text.lower())
    if len(letters.replace(" ", "")) < 2:
        return True
    return " ".join(letters.split()) in _HALLUCINATIONS


class STT(Protocol):
    def transcribe(self, wav: bytes) -> str: ...


class GroqSTT:
    """Groq-hosted Whisper. Fast, English-only here, ~$0.04 per audio hour."""

    def __init__(self, api_key: str, model: str = "whisper-large-v3-turbo"):
        self._client = Groq(api_key=api_key)
        self.model = model

    def transcribe(self, wav: bytes) -> str:
        result = self._client.audio.transcriptions.create(
            model=self.model,
            file=("speech.wav", wav),
            language="en",
            response_format="json",
            temperature=0.0,
        )
        return result.text.strip()
