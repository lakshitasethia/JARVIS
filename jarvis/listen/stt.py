"""Speech-to-text. Every engine implements `STT`."""

from typing import Protocol

from groq import Groq


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
