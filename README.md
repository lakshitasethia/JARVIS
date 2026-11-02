# JARVIS

A personal JARVIS-style AI assistant for macOS, built step by step.
See [research.md](research.md) for the research, architecture, and roadmap.

**Status:** Phase 3: tools. JARVIS can act: open/quit apps, volume, Spotify/Music, timers, weather, Apple Reminders, web search.

| Ask | Tool |
|---|---|
| "What time is it in Tokyo?" | `get_time` |
| "Open Spotify" / "Quit Safari" *(asks first)* | `open_app`, `quit_app` |
| "Turn it up a bit" / "Volume to 30" | `set_volume` |
| "Play" / "Skip this song" | `media_control` (Spotify or Music) |
| "Set a 10 minute pasta timer" / "How long's left?" / "Cancel it" | `set_timer`, `list_timers`, `cancel_timer` |
| "Do I need an umbrella tomorrow?" | `get_weather` (Open-Meteo, no key) |
| "Remind me to call Mum at 6pm" | `create_reminder` (Apple Reminders) |
| "Who won the match last night?" | web search (Groq built-in) |

Sensitive actions (like quitting an app) ask for a yes/no first. JARVIS enforces that in code, not just in the prompt.
The first time JARVIS controls an app like Reminders or Spotify, macOS asks you to allow it (Automation permission).

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you).

```bash
cp .env.example .env        # then paste your Groq key from https://console.groq.com/keys
uv sync
uv run python -m jarvis.speak.download   # Kokoro voice model (~120 MB, one time)
uv run jarvis
```

**Hands-free (default):** say **"Hey Jarvis"**, wait for the chime, and talk. JARVIS notices when you stop.
After it answers, just keep talking for follow-ups. Stay quiet (or say "never mind") and it goes back to sleep.
**Ctrl+C** stops JARVIS mid-sentence; Ctrl+C while it's idle quits.

**Push-to-talk / typing:** `uv run jarvis --ptt`, then press Enter on an empty line to talk (Enter again to stop), or type.
Commands: `/mute`, `/reset`, `/help`, `/quit`.

The first time you talk, macOS will ask to give your terminal app microphone access. Allow it.

## Tests

```bash
uv run pytest
```

## Layout

```
jarvis/
├── config.py            settings from .env
├── main.py              main loop: listen → think → speak
├── audio/
│   ├── mic.py           always-on mic stream (80 ms frames)
│   └── recorder.py      push-to-talk capture
├── listen/
│   ├── wake.py          "Hey Jarvis" wake word (openWakeWord, local)
│   ├── vad.py           voice activity detection (Silero, local)
│   ├── endpoint.py      decides when you started / finished talking
│   ├── stt.py           speech-to-text (Groq Whisper)
│   └── yesno.py         understands yes/no answers to confirmations
├── tools/
│   ├── registry.py      Tool definition, registry, confirmation enforcement
│   ├── system.py        time, open/quit apps, volume, music
│   ├── timers.py        spoken countdown timers
│   ├── weather.py       Open-Meteo weather
│   └── reminders.py     Apple Reminders
├── brain/
│   ├── agent.py         agent loop: think → use tools → answer; retries & safeguards
│   ├── llm.py           LLM interface + Groq implementation (tool calls, rate limits)
│   ├── persona.py       JARVIS system prompt
│   └── conversation.py  session chat history
└── speak/
    ├── sentences.py     split streaming text into sentences
    ├── tts.py           text-to-speech (Kokoro, macOS say)
    ├── speaker.py       plays sentences in order; synth N+1 while N plays
    ├── earcons.py       listening / sleeping chimes
    └── download.py      fetches Kokoro model files
```
