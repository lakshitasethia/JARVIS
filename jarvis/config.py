"""Settings loaded from environment variables / .env file."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    model: str
    reasoning_effort: str
    user_name: str
    max_history_messages: int
    show_timing: bool


def load_settings() -> Settings:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "GROQ_API_KEY is missing.\n"
            "  1. Get a key at https://console.groq.com/keys\n"
            "  2. cp .env.example .env\n"
            "  3. Paste the key into .env"
        )
    return Settings(
        groq_api_key=key,
        model=os.getenv("JARVIS_MODEL", "openai/gpt-oss-120b"),
        reasoning_effort=os.getenv("JARVIS_REASONING_EFFORT", "low"),
        user_name=os.getenv("JARVIS_USER_NAME", "sir"),
        max_history_messages=int(os.getenv("JARVIS_MAX_HISTORY", "20")),
        show_timing=os.getenv("JARVIS_SHOW_TIMING", "true").lower() == "true",
    )
