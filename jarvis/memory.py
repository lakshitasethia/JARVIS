"""Long-term memory: facts about the user that persist across sessions.

Stored in a local SQLite file (data/memory.db, git-ignored). Everything JARVIS remembers is
shown to the model in its system prompt, so it can use it naturally. The model adds, edits and
removes memories itself through tools (see jarvis/tools/memory_tools.py).
"""

import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

MAX_MEMORY_CHARS = 300
PROMPT_BUDGET_CHARS = 3_000  # ~750 tokens; keeps requests small on Groq's free tier


@dataclass(frozen=True)
class Memory:
    id: int
    text: str
    created: str
    updated: str


def _normalize(text: str) -> str:
    """For duplicate detection: lowercase, no punctuation, single spaces."""
    return " ".join(re.sub(r"[^a-z0-9 ]", "", text.lower()).split())


class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       text TEXT NOT NULL,
                       normalized TEXT NOT NULL,
                       created TEXT NOT NULL,
                       updated TEXT NOT NULL
                   )"""
            )

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _clean(text: str) -> str:
        text = " ".join(text.split())
        if not text:
            raise ValueError("a memory can't be empty")
        if len(text) > MAX_MEMORY_CHARS:
            raise ValueError(f"keep memories short (max {MAX_MEMORY_CHARS} characters)")
        return text

    def add(self, text: str) -> tuple[Memory, bool]:
        """Save a memory. Returns (memory, created?). Exact duplicates aren't stored twice."""
        text = self._clean(text)
        norm, now = _normalize(text), self._now()
        with self._lock, self._db:
            row = self._db.execute("SELECT id FROM memories WHERE normalized = ?", (norm,)).fetchone()
            if row:
                self._db.execute("UPDATE memories SET updated = ? WHERE id = ?", (now, row[0]))
                return self.get(row[0]), False
            cur = self._db.execute(
                "INSERT INTO memories (text, normalized, created, updated) VALUES (?, ?, ?, ?)",
                (text, norm, now, now),
            )
        return self.get(cur.lastrowid), True

    def update(self, memory_id: int, text: str) -> Memory | None:
        text = self._clean(text)
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE memories SET text = ?, normalized = ?, updated = ? WHERE id = ?",
                (text, _normalize(text), self._now(), memory_id),
            )
        return self.get(memory_id) if cur.rowcount else None

    def delete(self, memory_id: int) -> Memory | None:
        memory = self.get(memory_id)
        if memory:
            with self._lock, self._db:
                self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return memory

    def clear(self) -> int:
        with self._lock, self._db:
            return self._db.execute("DELETE FROM memories").rowcount

    def get(self, memory_id: int) -> Memory | None:
        row = self._db.execute(
            "SELECT id, text, created, updated FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return Memory(*row) if row else None

    def all(self) -> list[Memory]:
        """Most recently touched first."""
        rows = self._db.execute(
            "SELECT id, text, created, updated FROM memories ORDER BY updated DESC, id DESC"
        ).fetchall()
        return [Memory(*r) for r in rows]

    def prompt_section(self, budget: int = PROMPT_BUDGET_CHARS) -> str:
        """Memories formatted for the system prompt, newest first, within a size budget."""
        memories = self.all()
        if not memories:
            return "You don't know anything about the user yet."
        lines, used = [], 0
        for m in memories:
            line = f"[{m.id}] {m.text}"
            if used + len(line) > budget:
                lines.append(f"(+{len(memories) - len(lines)} older memories not shown)")
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)
