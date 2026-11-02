"""Countdown timers that JARVIS announces out loud when they finish."""

import itertools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from jarvis.tools.registry import Tool, params


def spoken_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h} hour{'s' * (h != 1)}")
    if m:
        parts.append(f"{m} minute{'s' * (m != 1)}")
    if s or not parts:
        parts.append(f"{s} second{'s' * (s != 1)}")
    return " ".join(parts)


@dataclass
class _Timer:
    id: int
    label: str
    ends_at: float
    handle: threading.Timer


class TimerManager:
    def __init__(self, announce: Callable[[str], None], clock: Callable[[], float] = time.monotonic):
        self._announce = announce
        self._clock = clock
        self._timers: dict[int, _Timer] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def set_timer(self, duration_seconds: int, label: str = "") -> str:
        if duration_seconds <= 0:
            return "Error: duration must be positive."
        if duration_seconds > 24 * 3600:
            return "Error: timers can be at most 24 hours."
        label = label.strip() or spoken_duration(duration_seconds)
        timer_id = next(self._ids)
        handle = threading.Timer(duration_seconds, self._fire, args=(timer_id,))
        handle.daemon = True
        with self._lock:
            self._timers[timer_id] = _Timer(timer_id, label, self._clock() + duration_seconds, handle)
        handle.start()
        return f"Timer '{label}' set for {spoken_duration(duration_seconds)}."

    def _fire(self, timer_id: int) -> None:
        with self._lock:
            timer = self._timers.pop(timer_id, None)
        if timer:
            self._announce(f"Your {timer.label} timer is done.")

    def list_timers(self) -> str:
        with self._lock:
            timers = sorted(self._timers.values(), key=lambda t: t.ends_at)
        if not timers:
            return "No timers running."
        now = self._clock()
        return " ".join(f"'{t.label}': {spoken_duration(t.ends_at - now)} left." for t in timers)

    def cancel_timer(self, label: str = "") -> str:
        with self._lock:
            timers = list(self._timers.values())
            if not timers:
                return "No timers running."
            if label.strip().lower() == "all":
                chosen = timers
            elif label.strip():
                chosen = [t for t in timers if label.lower() in t.label.lower()]
                if not chosen:
                    names = ", ".join(t.label for t in timers)
                    return f"No timer matching '{label}'. Running: {names}."
            elif len(timers) == 1:
                chosen = timers
            else:
                names = ", ".join(t.label for t in timers)
                return f"Several timers are running ({names}). Which one?"
            for t in chosen:
                t.handle.cancel()
                del self._timers[t.id]
        return "Cancelled " + ", ".join(f"'{t.label}'" for t in chosen) + "."

    def cancel_all(self) -> None:
        with self._lock:
            for t in self._timers.values():
                t.handle.cancel()
            self._timers.clear()

    def tools(self) -> list[Tool]:
        return [
            Tool(
                "set_timer",
                "Start a countdown timer; it's announced aloud when done.",
                params(
                    ["duration_seconds"],
                    duration_seconds={"type": "integer", "description": "Total seconds, e.g. 600 for 10 min"},
                    label={"type": "string", "description": "Optional name, e.g. 'pasta'"},
                ),
                self.set_timer,
            ),
            Tool("list_timers", "List running timers and time left.", params(), self.list_timers),
            Tool(
                "cancel_timer",
                "Cancel a timer by (part of) its label, or 'all'.",
                params(label={"type": "string"}),
                self.cancel_timer,
            ),
        ]
