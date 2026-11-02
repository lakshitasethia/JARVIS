# JARVIS

A personal JARVIS-style AI assistant for macOS, built step by step.
See [research.md](research.md) for the research, architecture, and roadmap.

**Status:** Phase 0: text chat in the terminal (Groq brain, JARVIS persona, streaming).

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you).

```bash
cp .env.example .env        # then paste your Groq key from https://console.groq.com/keys
uv sync
uv run jarvis
```

Commands inside the chat: `/reset`, `/help`, `/quit`. Press Ctrl+C to stop a reply mid-sentence.

## Tests

```bash
uv run pytest
```

## Layout

```
jarvis/
├── config.py            settings from .env
├── main.py              terminal chat loop
└── brain/
    ├── llm.py           LLM interface + Groq implementation
    ├── persona.py       JARVIS system prompt
    └── conversation.py  session chat history
```
