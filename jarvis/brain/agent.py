"""The agent loop: think → (maybe) use tools → think again → answer."""

import json
import re
from collections.abc import Generator, Iterator
from dataclasses import dataclass

from jarvis import log
from jarvis.brain.conversation import Conversation
from jarvis.brain.llm import LLM, ToolCallRejected, ToolCallRequest
from jarvis.tools.registry import Confirmer, ToolRegistry

MAX_ROUNDS = 5  # tool round-trips per user turn before giving up
ATTEMPTS_PER_ROUND = 2  # retry once if the model returns nothing, junk, or calls a bogus tool
FUMBLE = "Sorry, I fumbled that. Could you say it again?"
SCRAMBLED = "Sorry, my thoughts got scrambled there. Could you ask that again?"


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


class Scrambled(Exception):
    """The model's output degenerated into junk; abandon this reply."""


class JunkLineFilter:
    """Hides lines with nothing but punctuation ("...", "……") and detects degenerate output.

    Text is held back only until the current line contains a letter or digit, so normal
    replies still stream word by word.
    """

    MAX_JUNK_LINES = 2
    MAX_BLANK_RUN = 3  # this many newlines in a row is not a normal spoken reply

    def __init__(self):
        self._line = ""  # current line, held back while it's still punctuation-only
        self._line_ok = False
        self._newlines = 0
        self.junk_lines = 0

    def feed(self, text: str) -> str:
        out = []
        for ch in text:
            if ch == "\n":
                if self._line_ok:
                    out.append("\n")
                elif self._line.strip():
                    self.junk_lines += 1
                    if self.junk_lines >= self.MAX_JUNK_LINES:
                        raise Scrambled("too many punctuation-only lines")
                self._newlines += 1
                if self._newlines >= self.MAX_BLANK_RUN:
                    raise Scrambled("too many blank lines")
                self._line, self._line_ok = "", False
                continue
            if not ch.isspace():
                self._newlines = 0
            if self._line_ok:
                out.append(ch)
                continue
            self._line += ch
            if ch.isalnum():
                self._line_ok = True
                out.append(self._line)
        return "".join(out)


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
        stripper, guard, junk = CitationStripper(), RepeatGuard(), JunkLineFilter()
        raw: list[str] = []
        parts: list[str] = []
        calls: list[ToolCallRequest] = []
        try:
            for item in self.llm.stream(convo.messages(), self.tools.schemas()):
                if isinstance(item, str):
                    raw.append(item)
                    if clean := junk.feed(guard.feed(stripper.feed(item))):
                        parts.append(clean)
                        yield TextDelta(clean)
                else:
                    calls = item
        finally:
            log.event(
                "model_output",
                raw="".join(raw),
                tool_calls=[{"name": c.name, "arguments": c.arguments} for c in calls],
                repeat_cut=guard.tripped,
                junk_lines=junk.junk_lines,
            )
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
                except (ToolCallRejected, Scrambled) as e:
                    log.event("model_glitch", glitch=type(e).__name__, detail=str(e), attempt=attempt)
                    if streamed or attempt == ATTEMPTS_PER_ROUND - 1:
                        message = SCRAMBLED if isinstance(e, Scrambled) else FUMBLE
                        convo.add_assistant(message)
                        yield TextDelta((" " if streamed else "") + message)
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
                log.event("tool", name=call.name, arguments=call.arguments, ok=ok, output=output)
                convo.add_tool_result(call.id, call.name, output)
                yield ToolFinished(call.name, output, ok)

        message = "Sorry, I got tangled up trying to do that. Could you try asking differently?"
        convo.add_assistant(message)
        yield TextDelta(message)
