"""Tools that let JARVIS manage its long-term memory of the user."""

from jarvis.memory import MemoryStore
from jarvis.tools.registry import Tool, params


def memory_tools(store: MemoryStore) -> list[Tool]:
    def remember(fact: str) -> str:
        try:
            memory, created = store.add(fact)
        except ValueError as e:
            return f"Error: {e}."
        return f"Saved as memory [{memory.id}]." if created else f"Already known (memory [{memory.id}])."

    def update_memory(memory_id: int, fact: str) -> str:
        try:
            memory = store.update(int(memory_id), fact)
        except ValueError as e:
            return f"Error: {e}."
        return f"Memory [{memory_id}] updated." if memory else f"Error: there's no memory [{memory_id}]."

    def forget(memory_id: int) -> str:
        memory = store.delete(int(memory_id))
        return f"Forgot: {memory.text}" if memory else f"Error: there's no memory [{memory_id}]."

    def forget_everything() -> str:
        return f"Forgot all {store.clear()} memories."

    return [
        Tool(
            "remember",
            "Save a lasting fact about the user (preference, person, routine, detail they'd expect you to "
            "recall later). One short self-contained sentence in third person, e.g. 'Takes coffee black.'",
            params(["fact"], fact={"type": "string"}),
            remember,
        ),
        Tool(
            "update_memory",
            "Correct or replace a memory by its [id] when a fact changes.",
            params(["memory_id", "fact"], memory_id={"type": "integer"}, fact={"type": "string"}),
            update_memory,
        ),
        Tool(
            "forget",
            "Delete one memory by its [id] when the user asks you to forget it.",
            params(["memory_id"], memory_id={"type": "integer"}),
            forget,
        ),
        Tool(
            "forget_everything",
            "Delete ALL memories about the user. Only when explicitly asked.",
            params(),
            forget_everything,
            confirm=lambda: "Forget everything I know about you? This can't be undone.",
        ),
    ]
