"""Short-term conversation memory (the current session's chat history)."""

from collections.abc import Callable


class Conversation:
    def __init__(self, system_prompt: str | Callable[[], str], max_history_messages: int = 20):
        # A callable is re-evaluated every turn (e.g. so the prompt can include the current time).
        self._system_prompt = system_prompt
        self.max_history_messages = max_history_messages
        self.history: list[dict] = []

    @property
    def system_prompt(self) -> str:
        return self._system_prompt() if callable(self._system_prompt) else self._system_prompt

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str, tool_calls: list | None = None) -> None:
        message: dict = {"role": "assistant", "content": text or None}
        if tool_calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in tool_calls
            ]
        self.history.append(message)

    def add_tool_result(self, call_id: str, name: str, content: str) -> None:
        self.history.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": content})

    def rollback_to(self, length: int) -> None:
        """Drop everything after `length` messages (used when a turn fails half-way)."""
        del self.history[length:]

    def reset(self) -> None:
        self.history.clear()

    def messages(self) -> list[dict]:
        """System prompt + the most recent history, always starting on a user turn.

        Starting on a user turn also guarantees we never split an assistant tool call
        from its tool results.
        """
        recent = self.history[-self.max_history_messages :]
        while recent and recent[0]["role"] != "user":
            recent = recent[1:]
        return [{"role": "system", "content": self.system_prompt}, *recent]
