"""Turn a stream of LLM text chunks into speakable sentences, as early as possible."""

import re

# A sentence ends at . ! ? (optionally followed by closing quotes/brackets) + whitespace,
# or at a line break.
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")
_URL = re.compile(r"https?://\S+")
_MARKDOWN = re.compile(r"[*_`#>|]+")
_BULLET = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.MULTILINE)


class SentenceSplitter:
    """feed() chunks in, get complete sentences out; flush() returns the remainder."""

    def __init__(self, max_chars: int = 220, first_clause_chars: int = 40):
        self.max_chars = max_chars
        # The voice can't start until the first piece is synthesized, so if the opening
        # sentence runs long, start speaking at its first comma past this many characters.
        self.first_clause_chars = first_clause_chars
        self._buf = ""
        self._emitted = False

    def feed(self, chunk: str) -> list[str]:
        self._buf += chunk
        out: list[str] = []
        while True:
            match = _BOUNDARY.search(self._buf)
            if not self._emitted and not out and not match:
                comma = self._buf.find(", ", self.first_clause_chars)
                if comma != -1:
                    sentence, self._buf = self._buf[: comma + 1], self._buf[comma + 2 :]
                    out.append(sentence.strip())
                    continue
            if match:
                sentence, self._buf = self._buf[: match.start()], self._buf[match.end() :]
                if sentence.strip():
                    out.append(sentence.strip())
                continue
            if len(self._buf) > self.max_chars:
                # No sentence end in sight: break at the last comma or space.
                cut = max(self._buf.rfind(", ", 0, self.max_chars), self._buf.rfind(" ", 0, self.max_chars))
                if cut <= 0:
                    cut = self.max_chars
                sentence, self._buf = self._buf[: cut + 1], self._buf[cut + 1 :]
                if sentence.strip():
                    out.append(sentence.strip())
                continue
            self._emitted = self._emitted or bool(out)
            return out

    def flush(self) -> list[str]:
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest else []


def clean_for_speech(text: str) -> str:
    """Strip things that sound wrong when read aloud (markdown, URLs, bullets).

    Returns "" for text with nothing pronounceable ("...", "—"): the voice model makes
    eerie breathing noises if asked to speak bare punctuation.
    """
    text = _URL.sub("a link", text)
    text = _BULLET.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if re.search(r"[A-Za-z0-9]", text) else ""
