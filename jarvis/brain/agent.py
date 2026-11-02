"""The agent loop: think → (maybe) use tools → think again → answer."""

import json
import re
from collections.abc import Generator, Iterator
from dataclasses import dataclass

from jarvis.brain.conversation import Conversation
from jarvis.brain.llm import LLM, ToolCallRejected, ToolCallRequest
from jarvis.tools.registry import Confirmer, ToolRegistry

MAX_ROUNDS = 5  # tool round-trips per user turn before giving up
ATTEMPTS_PER_ROUND = 2  # retry once if the model returns nothing or calls a bogus tool
FUMBLE = "Sorry, I fumbled that. Could you say it again?"


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolStarted:
    name: str
    arguments: str


@dataclass
class ToolFinished:
    name: str
    output: str
    ok: bool


Event = TextDelta | ToolStarted | ToolFinished


class CitationStripper:
    """Removes web-search citation markers like 【1†L9-L17】 and odd Unicode spacing from streaming text."""

    _SPACES = re.compile("[   ]")

    def __init__(self):
        self._inside = False

    def feed(self, text: str) -> str:
        out = []
        for ch in self._SPACES.sub(" ", text).replace("‑", "-"):
            if ch == "【":
                self._inside = True
            elif ch == "】" and self._inside:
                self._inside = False
            elif not self._inside:
                out.append(ch)
        return "".join(out)


# "…a gentle breeze.London is currently…": a sentence end glued to a capital letter with no
# space is where gpt-oss starts writing its answer a second time.
_SEAM = re.compile(r"[a-z0-9)\]\"'’%°][.!?](?=[A-Z])")


class RepeatGuard:
    """Cuts off the duplicate answer gpt-oss sometimes appends after using a tool."""

    MIN_CHARS = 20  # don't trip on very short replies

    def __init__(self):
        self._prev = ""  # last 2 chars seen (the regex needs a little left context)
        self._length = 0
        self.tripped = False

    def feed(self, text: str) -> str:
        if self.tripped:
            return ""
        window = self._prev + text
        match = _SEAM.search(window)
        if match and self._length + max(0, match.end() - len(self._prev)) >= self.MIN_CHARS:
            self.tripped = True
            text = text[: max(0, match.end() - len(self._prev))]
        self._length += len(text)
        self._prev = (self._prev + text)[-2:]
        return text


def summarize_arguments(raw: str, limit: int = 80) -> str:
    try:
        args = json.loads(raw or "{}")
        text = ", ".join(f"{k}={v!r}" for k, v in args.items())
    except json.JSONDecodeError:
        text = raw
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Agent:
    def __init__(self, llm: LLM, tools: ToolRegistry):
        self.llm = llm
        self.tools = tools

    def _one_round(self, convo: Conversation) -> Generator[TextDelta, None, tuple[str, list[ToolCallRequest]]]:
        """Stream one model response. Yields text; returns (full text, tool calls)."""
        stripper, guard = CitationStripper(), RepeatGuard()
        parts: list[str] = []
        calls: list[ToolCallRequest] = []
        for item in self.llm.stream(convo.messages(), self.tools.schemas()):
            if isinstance(item, str):
                if clean := guard.feed(stripper.feed(item)):
                    parts.append(clean)
                    yield TextDelta(clean)
            else:
                calls = item
        return "".join(parts).strip(), calls

    def run(self, convo: Conversation, confirmer: Confirmer) -> Iterator[Event]:
        """Answer the latest user message, running tools as needed. Updates `convo`.

        Always leaves the conversation ending in an assistant message.
        """
        for _ in range(MAX_ROUNDS):
            text, calls = "", []
            for attempt in range(ATTEMPTS_PER_ROUND):
                streamed = False
                try:
                    round_events = self._one_round(convo)
                    while True:
                        try:
                            event = next(round_events)
                        except StopIteration as done:
                            text, calls = done.value
                            break
                        streamed = True
                        yield event
                except ToolCallRejected:
                    if streamed or attempt == ATTEMPTS_PER_ROUND - 1:
                        convo.add_assistant(FUMBLE)
                        yield TextDelta(FUMBLE)
                        return
                    continue  # nothing was said yet, so a silent retry is safe
                if text or calls or attempt == ATTEMPTS_PER_ROUND - 1:
                    break

            if not calls:
                if not text:
                    text = FUMBLE
                    yield TextDelta(text)
                convo.add_assistant(text)
                return

            convo.add_assistant(text, calls)
            for call in calls:
                yield ToolStarted(call.name, call.arguments)
                output, ok = self.tools.run(call.name, call.arguments, confirmer)
                convo.add_tool_result(call.id, call.name, output)
                yield ToolFinished(call.name, output, ok)

        message = "Sorry, I got tangled up trying to do that. Could you try asking differently?"
        convo.add_assistant(message)
        yield TextDelta(message)
