"""Short-term conversation memory (the current session's chat history)."""


class Conversation:
    def __init__(self, system_prompt: str, max_history_messages: int = 20):
        self.system_prompt = system_prompt
        self.max_history_messages = max_history_messages
        self.history: list[dict] = []

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def reset(self) -> None:
        self.history.clear()

    def messages(self) -> list[dict]:
        """System prompt + the most recent history, always starting on a user turn."""
        recent = self.history[-self.max_history_messages :]
        while recent and recent[0]["role"] != "user":
            recent = recent[1:]
        return [{"role": "system", "content": self.system_prompt}, *recent]
