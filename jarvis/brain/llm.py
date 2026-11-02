"""The LLM interface. Every brain (Groq today, maybe Ollama later) implements `LLM`."""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

import groq
from groq import Groq


class ToolCallRejected(Exception):
    """The model tried to call a tool that doesn't exist or with malformed arguments."""


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model produced it


# A stream yields reply text (str) as it's generated, then, if the model wants tools,
# one final list[ToolCallRequest].
StreamItem = str | list[ToolCallRequest]


class LLM(Protocol):
    model: str

    def stream(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[StreamItem]: ...


def _retry_after_seconds(e: groq.RateLimitError) -> float | None:
    try:
        return float(e.response.headers.get("retry-after"))
    except (TypeError, ValueError, AttributeError):
        return None


def _is_tool_call_rejection(e: groq.BadRequestError) -> bool:
    body = e.body if isinstance(e.body, dict) else {}
    code = body.get("code") or (body.get("error") or {}).get("code")
    return code == "tool_use_failed" or "tool call validation failed" in str(e).lower()


class GroqLLM:
    MAX_RATE_LIMIT_WAIT = 15.0  # seconds; longer than this and we give up and say so

    def __init__(
        self,
        api_key: str,
        model: str,
        reasoning_effort: str = "low",
        web_search: bool = True,
        on_wait: Callable[[float], None] | None = None,
    ):
        # We handle rate-limit retries ourselves so the user hears/sees why there's a pause.
        self._client = Groq(api_key=api_key, max_retries=0)
        self._on_wait = on_wait or (lambda seconds: None)
        self.model = model
        self._reasoning_effort = reasoning_effort
        self._is_gpt_oss = model.startswith("openai/gpt-oss")
        # Groq-hosted web search runs server-side; only gpt-oss models support it.
        self.web_search = web_search and self._is_gpt_oss

    def _reasoning_kwargs(self) -> dict:
        # gpt-oss and qwen3 models "think" before answering. Low effort keeps voice
        # replies fast, and we hide the reasoning text (we only want the answer).
        if self._is_gpt_oss or self.model.startswith("qwen/"):
            return {"reasoning_effort": self._reasoning_effort, "include_reasoning": False}
        return {}

    def stream(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[StreamItem]:
        all_tools = list(tools or [])
        if self.web_search:
            all_tools.append({"type": "browser_search"})
        kwargs = {"tools": all_tools, "tool_choice": "auto"} if all_tools else {}

        request = dict(model=self.model, messages=messages, stream=True, **kwargs, **self._reasoning_kwargs())
        calls: dict[int, ToolCallRequest] = {}
        try:
            for chunk in self._create(request):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
                for tc in delta.tool_calls or []:
                    call = calls.setdefault(tc.index, ToolCallRequest(id="", name="", arguments=""))
                    if tc.id:
                        call.id = tc.id
                    if tc.function and tc.function.name:
                        call.name += tc.function.name
                    if tc.function and tc.function.arguments:
                        call.arguments += tc.function.arguments
        except groq.BadRequestError as e:
            if _is_tool_call_rejection(e):
                raise ToolCallRejected(str(e)) from e
            raise
        if calls:
            yield [calls[i] for i in sorted(calls)]

    def _create(self, request: dict):
        """Start the request, waiting out short free-tier rate limits (and saying so)."""
        waited = 0.0
        while True:
            try:
                return self._client.chat.completions.create(**request)
            except groq.RateLimitError as e:
                wait = _retry_after_seconds(e)
                if wait is None or waited + wait > self.MAX_RATE_LIMIT_WAIT:
                    raise
                self._on_wait(wait)
                time.sleep(wait)
                waited += wait
