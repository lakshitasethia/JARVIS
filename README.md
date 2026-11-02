# JARVIS

A personal JARVIS-style AI assistant for macOS, built step by step.
See [research.md](research.md) for the research, architecture, and roadmap.

**Status:** Phase 2: hands-free. Say "Hey Jarvis" → it listens until you stop → thinks (Groq) → answers aloud (Kokoro), then keeps listening for follow-ups.

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
│   └── stt.py           speech-to-text (Groq Whisper)
├── brain/
│   ├── llm.py           LLM interface + Groq implementation
│   ├── persona.py       JARVIS system prompt
│   └── conversation.py  session chat history
└── speak/
    ├── sentences.py     split streaming text into sentences
    ├── tts.py           text-to-speech (Kokoro, macOS say)
    ├── speaker.py       plays sentences in order; synth N+1 while N plays
    ├── earcons.py       listening / sleeping chimes
    └── download.py      fetches Kokoro model files
```
