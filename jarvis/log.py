"""Session log: one JSON line per event in logs/<session>.jsonl (stays on this Mac, git-ignored).

Records what JARVIS heard, what the model said (raw, before any cleanup), which tools ran,
and errors, so odd behaviour can be diagnosed afterwards instead of guessed at.
"""

import json
import threading
from datetime import datetime
from pathlib import Path

_lock = threading.Lock()
_path: Path | None = None


def start(directory: str = "logs") -> Path:
    global _path
    folder = Path(directory)
    folder.mkdir(exist_ok=True)
    _path = folder / f"{datetime.now():%Y-%m-%d_%H%M%S}.jsonl"
    event("session_start")
    return _path


def event(kind: str, **data) -> None:
    """Append one event. A no-op until start() is called (e.g. in tests)."""
    if _path is None:
        return
    record = {"t": datetime.now().isoformat(timespec="milliseconds"), "event": kind, **data}
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _lock, _path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
