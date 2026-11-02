# JARVIS

A personal JARVIS-style AI assistant for macOS, built step by step.
See [research.md](research.md) for the research, architecture, and roadmap.

**Status:** Phase 1: push-to-talk voice. Speak (Groq Whisper) → think (Groq gpt-oss-120b) → answer aloud (Kokoro, local), sentence by sentence.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you).

```bash
cp .env.example .env        # then paste your Groq key from https://console.groq.com/keys
uv sync
uv run python -m jarvis.speak.download   # Kokoro voice model (~120 MB, one time)
uv run jarvis
```

Press **Enter on an empty line** to talk, then **Enter again** when you're done. Or just type.
Commands: `/mute` (toggle voice), `/reset`, `/help`, `/quit`. **Ctrl+C** stops JARVIS mid-sentence.

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
├── audio/recorder.py    microphone capture (push-to-talk)
├── listen/stt.py        speech-to-text (Groq Whisper)
├── brain/
│   ├── llm.py           LLM interface + Groq implementation
│   ├── persona.py       JARVIS system prompt
│   └── conversation.py  session chat history
└── speak/
    ├── sentences.py     split streaming text into sentences
    ├── tts.py           text-to-speech (Kokoro, macOS say)
    ├── speaker.py       plays sentences in order; synth N+1 while N plays
    └── download.py      fetches Kokoro model files
```
