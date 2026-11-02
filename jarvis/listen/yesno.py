"""Understand spoken/typed answers to yes/no confirmation questions. When unsure: not a yes."""

import re

_NO = ["no", "nope", "nah", "don't", "do not", "dont", "cancel", "stop", "never mind", "nevermind",
       "negative", "wait", "hold on", "not"]
_YES = ["yes", "yeah", "yep", "yup", "ya", "sure", "ok", "okay", "do it", "go ahead", "go for it",
        "confirm", "confirmed", "please", "affirmative", "absolutely", "of course", "correct", "y"]


def _has(text: str, phrases: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(p)}\b", text) for p in phrases)


def parse_yes_no(text: str) -> bool | None:
    """True for a clear yes, False for no, None if unclear. Any 'no' signal wins."""
    t = re.sub(r"[^a-z' ]", " ", text.lower())
    if _has(t, _NO):
        return False
    if _has(t, _YES):
        return True
    return None
