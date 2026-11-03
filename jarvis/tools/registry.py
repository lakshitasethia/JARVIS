"""Tool definitions, the registry, and permission checks."""

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

MAX_OUTPUT_CHARS = 2_000  # keep tool results small: Groq's free tier allows ~8k tokens/minute
DECLINED = (
    "NOT DONE. The user answered NO to the confirmation, so this action did not happen and nothing "
    "changed. Tell them you've left things as they were. Do not say it was done."
)

# Asks the user a yes/no question; returns True only on a clear "yes".
Confirmer = Callable[[str], bool]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema for the arguments
    func: Callable[..., str]
    # For sensitive tools: given the call's arguments, the question to ask before running.
    confirm: Callable[..., str] | None = None

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def params(required: list[str] | None = None, **properties: dict) -> dict:
    """Shorthand for an object JSON Schema."""
    return {"type": "object", "properties": properties, "required": required or []}


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def run(self, name: str, raw_arguments: str, confirmer: Confirmer) -> tuple[str, bool]:
        """Execute a tool call. Returns (output for the model, succeeded?). Never raises."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: there is no tool called '{name}'.", False
        try:
            args = json.loads(raw_arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            return f"Error: invalid arguments ({e}).", False

        if tool.confirm is not None:
            try:
                question = tool.confirm(**args)
            except TypeError as e:
                return f"Error: bad arguments for {name} ({e}).", False
            if not confirmer(question):
                return DECLINED, False

        try:
            output = tool.func(**args)
        except TypeError as e:
            return f"Error: bad arguments for {name} ({e}).", False
        except Exception as e:  # tools touch the outside world; report, don't crash
            return f"Error: {e}", False
        output = str(output)
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + " …(truncated)"
        # Tools report expected failures (unknown app, bad city…) as "Error: …" text.
        return output, not output.startswith("Error:")


def applescript_string(text: str) -> str:
    """Quote a Python string as an AppleScript string literal."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def osascript(script: str, timeout: float = 30) -> str:
    """Run AppleScript and return its output; raises RuntimeError with a readable message."""
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        message = result.stderr.strip()
        if "Not authorized" in message or "-1743" in message:
            message = (
                "macOS blocked this. Allow your terminal app under System Settings → "
                "Privacy & Security → Automation, then try again."
            )
        raise RuntimeError(message)
    return result.stdout.strip()
