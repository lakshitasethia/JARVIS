"""The LLM interface. Every brain (Groq today, maybe Ollama later) implements `LLM`."""

from collections.abc import Iterator
from typing import Protocol

from groq import Groq


class LLM(Protocol):
    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield the reply text in chunks as soon as the model produces them."""
        ...


class GroqLLM:
    def __init__(self, api_key: str, model: str, reasoning_effort: str = "low"):
        self._client = Groq(api_key=api_key)
        self.model = model
        self._reasoning_effort = reasoning_effort

    def _reasoning_kwargs(self) -> dict:
        # gpt-oss and qwen3 models "think" before answering. Low effort keeps voice
        # replies fast, and we hide the reasoning text (we only want the answer).
        if self.model.startswith("openai/gpt-oss") or self.model.startswith("qwen/"):
            return {"reasoning_effort": self._reasoning_effort, "include_reasoning": False}
        return {}

    def stream(self, messages: list[dict]) -> Iterator[str]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **self._reasoning_kwargs(),
        )
        for chunk in response:
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content
            if text:
                yield text
