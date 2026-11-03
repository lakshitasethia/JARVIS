"""Settings loaded from environment variables / .env file."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    # Brain
    model: str
    reasoning_effort: str
    user_name: str
    max_history_messages: int
    show_timing: bool
    log_sessions: bool
    # Tools
    home_location: str
    web_search: bool
    # Ears
    stt_model: str
    wake_threshold: float
    silence_seconds: float
    listen_timeout: float
    followup_seconds: float
    barge_in: bool
    echo_cancel: bool
    # Voice
    tts_engine: str  # kokoro | say | none
    voice: str
    speech_speed: float
    kokoro_model_path: str
    kokoro_voices_path: str


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
        log_sessions=os.getenv("JARVIS_LOG", "true").lower() == "true",
        home_location=os.getenv("JARVIS_HOME_LOCATION", "").strip(),
        web_search=os.getenv("JARVIS_WEB_SEARCH", "true").lower() == "true",
        stt_model=os.getenv("JARVIS_STT_MODEL", "whisper-large-v3-turbo"),
        wake_threshold=float(os.getenv("JARVIS_WAKE_THRESHOLD", "0.5")),
        silence_seconds=float(os.getenv("JARVIS_SILENCE_SECONDS", "0.8")),
        listen_timeout=float(os.getenv("JARVIS_LISTEN_TIMEOUT", "8")),
        followup_seconds=float(os.getenv("JARVIS_FOLLOWUP_SECONDS", "6")),
        barge_in=os.getenv("JARVIS_BARGE_IN", "true").lower() == "true",
        echo_cancel=os.getenv("JARVIS_ECHO_CANCEL", "true").lower() == "true",
        tts_engine=os.getenv("JARVIS_TTS", "kokoro").lower(),
        voice=os.getenv("JARVIS_VOICE", "bm_george"),
        speech_speed=float(os.getenv("JARVIS_SPEECH_SPEED", "1.0")),
        kokoro_model_path=os.getenv("JARVIS_KOKORO_MODEL", "models/kokoro-v1.0.int8.onnx"),
        kokoro_voices_path=os.getenv("JARVIS_KOKORO_VOICES", "models/voices-v1.0.bin"),
    )
