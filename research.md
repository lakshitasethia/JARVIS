# JARVIS — Research Notes

> Goal: build our own JARVIS-style personal AI assistant, step by step.
> Researched: 2 Nov 2026 · Target machine: MacBook, Apple M5, 16 GB RAM, macOS 26.5

---

## 1. What JARVIS actually is (and what we're really building)

**J.A.R.V.I.S.** ("Just A Rather Very Intelligent System") is Tony Stark's AI in the Marvel films (voiced by Paul Bettany, 2008–2015). Strip away the movie magic and it's a useful product spec. JARVIS is:

| Movie behavior | What it means in real engineering terms |
|---|---|
| Always listening for "JARVIS" | **Wake word detection**, running locally and cheaply, 24/7 |
| Understands Tony instantly, natural back-and-forth | **Speech-to-text (STT)** + **turn detection** + low latency |
| Can be interrupted mid-sentence | **Barge-in**: stop speaking the moment the user talks |
| Witty, consistent personality | **System prompt / persona** on top of a strong LLM |
| Answers anything | **LLM brain** + **web search** + knowledge |
| Does things: runs the house, the lab, the suit | **Tool use / actions**: APIs, smart home, apps, files, browser |
| Remembers Tony's preferences and history | **Long-term memory** |
| Speaks up on its own ("Sir, you have a call") | **Proactivity**: schedules, triggers, notifications |
| Sees the lab, scans objects | **Vision**: camera/screen understanding (multimodal LLM) |
| Only obeys Tony | **Security**: identity, permissions, confirmation of risky actions |

**The key insight:** a real-world JARVIS is not one model. It's a **pipeline** of specialized components wrapped around an **agent** (an LLM that can call tools). Every serious open-source assistant follows this shape.

---

## 2. Anatomy of a modern voice assistant

```
 ┌──────────┐   ┌─────┐   ┌─────┐   ┌──────────────┐   ┌───────────────────────┐   ┌─────┐   ┌─────────┐
 │   Mic    │──▶│Wake │──▶│ VAD │──▶│ STT          │──▶│ BRAIN (LLM agent)     │──▶│ TTS │──▶│ Speaker │
 │ (always) │   │word │   │     │   │ speech→text  │   │ persona + tools +     │   │     │   │         │
 └──────────┘   └─────┘   └─────┘   └──────────────┘   │ memory                │   └─────┘   └─────────┘
                                ▲                       └──────────┬────────────┘                │
                                │                                  │ tool calls                  │
                                │                     ┌────────────▼────────────┐                │
                                │                     │ TOOLS: web, calendar,   │                │
                                │                     │ email, files, apps,     │                │
                                │                     │ smart home (MCP)        │                │
                                │                     └─────────────────────────┘                │
                                └──────────────── barge-in: user speaks → stop playback ◀────────┘
```

| Stage | Job | Why it matters |
|---|---|---|
| **Wake word** | Tiny always-on model that only detects "Hey Jarvis" | Privacy + cost: nothing else runs until it fires |
| **VAD** (voice activity detection) | Detects when speech starts/stops | Cuts audio into utterances; enables barge-in |
| **Turn detection** | Decides whether the user is *finished*, not just pausing | The difference between "feels natural" and "keeps interrupting me" |
| **STT** | Audio → text | Accuracy + speed; runs locally on Apple Silicon fine |
| **Brain** | Understands, reasons, decides which tools to call | Where intelligence lives |
| **Tools** | Do real things | What makes it JARVIS rather than a chatbot |
| **Memory** | Facts about you, persisting across sessions | Personalization |
| **TTS** | Text → voice | Personality; must stream sentence-by-sentence for low latency |

### Latency is the whole game

Humans expect a reply within about a second. Rough budget for the time from when you stop talking to when the first audio plays (our target is **< ~1.5 s**; we'll measure real numbers in Phase 1):

| Step | Typical range | Lever |
|---|---|---|
| End-of-turn detection | ~0.3–0.8 s | Silero VAD + smart turn model; tune silence threshold |
| STT (short utterance) | ~0.2–0.5 s on Apple Silicon (Parakeet MLX / Core ML) | Pick a fast model |
| LLM first token | a few hundred ms to ~1 s (cloud) | Streaming, prompt caching, low effort for chit-chat |
| TTS first audio chunk | ~0.2–0.5 s | Stream sentence-by-sentence, don't wait for the full reply |

**Rule:** stream everything. Start speaking sentence 1 while the LLM is still writing sentence 3.

### Two architectures

| | **Cascaded** (STT → LLM → TTS) | **Speech-to-speech** (one audio model) |
|---|---|---|
| Examples | Home Assistant Assist, Pipecat pipelines, OVOS | OpenAI Realtime, Gemini Live; open: Kyutai Moshi, NVIDIA NemotronLabs VoiceChat (Sep 2026 paper) |
| Latency | Good (~1 s with tuning) | Best (~300–500 ms reported for cloud APIs) |
| Intelligence + tool use | **Best**: use any top LLM | Weaker; open full-duplex models still struggle with tool arguments |
| Swap parts / run locally | Yes, every piece is replaceable | Mostly locked to one vendor |
| Debuggable | Yes, you can see the text at each step | Hard |

**Verdict:** cascaded is the industry standard for assistants that *do things*, and it's the best way to learn. Speech-to-speech is an optional future upgrade.

---

## 3. Open-source landscape

### 3a. Complete assistants (the "JARVIS alternatives")

| Project | What it is | Strengths | Weaknesses for us |
|---|---|---|---|
| **[OpenClaw](https://github.com/openclaw/openclaw)** (MIT, ~394k★) | General-purpose personal agent on your own hardware: gateway, 20+ chat channels (iMessage, WhatsApp, Telegram, Slack…), skills/plugins, swappable LLMs (Claude, local), macOS computer control | Most popular "AI that does things" project; huge plugin ecosystem | Chat-first, not voice-first; large and complex; tools run on host unless sandboxed; big attack surface |
| **[Home Assistant Assist](https://www.home-assistant.io/voice_control/)** | Voice pipeline in the leading open smart-home platform. Wyoming protocol plugs in openWakeWord + Whisper + Piper + any LLM | Best-in-class home control; modular; mature | Home-centric; needs an HA server; less suited to a desktop/personal-agent use |
| **[OpenVoiceOS (OVOS)](https://openvoiceos.org/)** | Community successor to Mycroft (Mycroft the company shut down in 2023); non-profit foundation | Classic skill-based voice OS, privacy-focused, Mycroft skills compatible | Older "intent + skill" design; LLM integration bolted on |
| **Leon** | Open-source personal assistant (Node.js) | Simple and hackable | Activity unclear (topic listing last updated 2024) |
| **Open Interpreter** | CLI agent that runs code on your machine | Strong at "do it on my computer" | Not a voice assistant; terminal-based |
| **[llm-guy/jarvis](https://github.com/llm-guy/jarvis)** | Small hobby project: wake word → Qwen3 1.7B via Ollama + LangChain → pyttsx3 | Tiny, readable reference | Toy-level: one tool, robotic TTS, hardcoded mic |
| **[kishanrajput23/Jarvis-Desktop-Voice-Assistant](https://github.com/kishanrajput23/Jarvis-Desktop-Voice-Assistant)** (~1k★) | Classic Python "Jarvis" tutorial: speech recognition + if/else commands | Popular learning resource | Pre-LLM design (keyword matching) |

**Takeaway:** there's no single polished, voice-first, open-source JARVIS. The pieces are excellent, though. OpenClaw is the closest thing to an "agent that does things", and Home Assistant has the best voice pipeline. **Building our own from best-in-class components is the right call**, and we can borrow ideas (and possibly MCP servers and skills) from both.

### 3b. Voice-agent frameworks (glue for the pipeline)

| Framework | License | Notes |
|---|---|---|
| **[Pipecat](https://github.com/pipecat-ai/pipecat)** (by Daily) | BSD-2 | Python, composable "frames" pipeline, transport-agnostic, built-in Silero VAD + Smart Turn + barge-in; many STT/LLM/TTS plugins incl. Anthropic. Most flexible. |
| **[LiveKit Agents](https://github.com/livekit/agents)** | Apache-2.0 | Built on WebRTC media server; Python + Node; telephony/SIP. Faster to production for apps/phones, heavier for a local desktop assistant. |
| **HF [speech-to-speech](https://pypi.org/project/speech-to-speech/)** | Apache-2.0 | Modular VAD→STT→LLM→TTS pipeline with an OpenAI Realtime-compatible WebSocket API |
| **Chuchote** (PyPI) | n/a | Fully local: openWakeWord + Silero VAD + faster-whisper + Piper + Ollama. A good reference design. |

### 3c. Component options by layer

**Wake word**
- **openWakeWord** (Apache-2.0). The open-source default, has a pre-trained "hey jarvis" model, and lets you train custom words. ✅ *Pick*
- Picovoice Porcupine: very accurate, but needs a commercial license/key.

**VAD + turn detection**
- **Silero VAD**: the de-facto standard (MIT). ✅ *Pick*
- **Smart Turn** (Pipecat) or LiveKit turn detector: ML model that decides "is the user done?". Add in Phase 4.

**Speech-to-text** (on Apple Silicon)
- **Parakeet** (NVIDIA, via `parakeet-mlx` or FluidAudio Core ML) is the fastest in most 2026 Mac benchmarks (~0.2–0.5 s for a short utterance). English + ~25 European languages. ✅ *Pick for English*
- **Whisper** (`mlx-whisper`, `whisper.cpp`, WhisperKit) supports ~99 languages, so use it if we need Hindi or Hinglish. Slower but very robust.
- Moonshine: tiny, edge/streaming focused.

**Text-to-speech**
- **Kokoro-82M** (Apache-2.0): best quality-to-size ratio, runs without a GPU, MLX ports (`mlx-audio`, `kokoro-mlx`) stream sentence by sentence. ✅ *Pick*
- **Piper** (MIT): ultra-fast and robotic-ish, great fallback / Raspberry Pi.
- **Chatterbox** (MIT): voice cloning from ~10 s of audio, if we want a custom "JARVIS" voice later. (Don't clone Paul Bettany's voice; use a legally clean voice.)
- macOS `say`: zero setup, good enough for Phase 0.
- Cloud: ElevenLabs etc. for premium quality.

**Brain (LLM)**: see §4.

**Tools**
- **MCP (Model Context Protocol)**: open standard (Anthropic, 2024) for plugging tools into an LLM. Thousands of servers exist: filesystem, GitHub, Playwright (browser), Google Workspace, Notion, Slack, Home Assistant… ✅ *Use MCP as the plugin system*, so JARVIS gains abilities without custom code per integration.
- Native tools we'll write ourselves: time/date, macOS apps (AppleScript / Shortcuts), volume, timers, reminders.

**Memory**
| Option | Model | Best for |
|---|---|---|
| **Mem0** (Apache-2.0, ~60k★) | Memory layer you bolt onto an app | User preferences/facts. Simplest. |
| **Zep / Graphiti** (~28k★) | Temporal knowledge graph | Facts that change over time |
| **Letta** (ex-MemGPT, ~24k★) | Full agent runtime with self-managed memory tiers | When you want the agent *to be* the memory system |
| **DIY: SQLite + a "memory" tool** | LLM decides what to save/recall | Learning, full control, zero dependencies ✅ *Start here* |

(Benchmark claims between these vendors conflict, so don't trust any single score.)

---

## 4. The brain: cloud vs local

### Cloud: Claude (Anthropic API)

Current models (pricing per 1M tokens, input / output):

| Model | ID | Price | Fit for JARVIS |
|---|---|---|---|
| Claude Opus 5.5 | `claude-opus-5-5` | $4 / $20 | Smartest everyday model; best for complex multi-step tasks. Use `effort: low` for chat to keep latency down. |
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | $2 / $10 | Strong and fast; good all-rounder |
| Claude Haiku 5.5 | `claude-haiku-5-5` | $0.10 / $0.50 | Cheapest and fastest; great for quick voice replies |

All support **streaming**, **tool use**, **vision**, **prompt caching**, server-side **web search**, and **MCP**.

**Rough cost per voice turn** (~2k input tokens of system prompt + tools + history, ~150 output tokens, before caching discounts):
- Opus 5.5 ≈ $0.011/turn → ~100 turns/day ≈ **$1/day**
- Haiku 5.5 ≈ $0.0003/turn → ~100 turns/day ≈ **$0.03/day**
- Prompt caching cuts the repeated input cost substantially.

### Local: Ollama / MLX

- 16 GB of unified memory realistically fits **~4–8B parameter models at 4-bit** (e.g. Qwen 3 8B, Llama 3.x 8B) alongside STT/TTS.
- Pros: free, private, offline. Cons: noticeably weaker at multi-step tool use; slower first token on a laptop.

### Recommendation

> **Update:** we chose **Groq** as the provider. See §8.

**Hybrid, behind one interface:** a cloud Claude model as the main brain (quality + reliable tool use), with a pluggable `LLM` interface so a local Ollama model can be dropped in for offline/private mode. The speech layers (wake word, VAD, STT, TTS) all run **locally**, so audio never leaves the Mac. Only the transcript goes to the brain.

---

## 5. Safety & security (JARVIS can *act*, so this matters)

1. **Permission tiers for tools:** *safe* (time, weather, search) run freely → *sensitive* (send email, delete files, purchases, smart locks) require **spoken confirmation**.
2. **Prompt injection:** content from web pages, emails and documents is *data*, never instructions. Don't let a webpage tell JARVIS to email your files.
3. **Secrets:** API keys in `.env` (git-ignored) or the macOS Keychain, never in the repo.
4. **Mic privacy:** wake word runs locally; nothing is recorded/sent until it fires; visible indicator when listening.
5. **Audit log:** every tool call logged locally.
6. **Least privilege for MCP servers:** prefer official/well-known servers; scope filesystem access to specific folders.

---

## 6. Proposed architecture for our JARVIS

```
jarvis/
├── audio/        mic input, speaker output, device selection
├── wake/         openWakeWord ("hey jarvis")
├── listen/       Silero VAD + turn detection + STT (Parakeet / Whisper)
├── brain/        LLM interface (Claude ⇄ Ollama), persona, agent loop, streaming
├── tools/        native tools + MCP client, permission tiers
├── memory/       SQLite memory store + memory tools
├── speak/        TTS (say → Kokoro), sentence streaming, barge-in
├── config.py     settings (.env)
└── main.py       state machine: SLEEPING → LISTENING → THINKING → SPEAKING
```

- **Language:** Python 3.12 via `uv` (the entire voice ecosystem is Python-first; system Python 3.9 is too old).
- **Design principle:** every layer sits behind a small interface, so any component can be swapped (e.g. Kokoro → ElevenLabs, Claude → Ollama).
- **Text mode first:** a keyboard chat mode shares the same brain/tools/memory, which makes testing much faster.

---

## 7. Build roadmap (step by step)

Each phase ends with something that **works and can be demoed**.

| Phase | Build | Done when… |
|---|---|---|
| **0. Foundation** ✅ | `uv` project, config/.env, logging, Claude API key, text chat with JARVIS persona + streaming | You can type to JARVIS in the terminal and it replies in character |
| **1. Voice I/O (push-to-talk)** ✅ | Mic capture, STT (Groq Whisper turbo), TTS (Kokoro int8, local; `say` fallback), sentence streaming. Measure latency per stage. | Hold a key, speak, and hear JARVIS answer |
| **2. Hands-free** ✅ | openWakeWord "hey jarvis", Silero VAD end-of-speech, follow-up window (keep listening ~8 s after a reply) | "Hey Jarvis, what time is it?" works with no keyboard |
| **3. Tools** | Agent loop + native tools (time, weather, web search, open apps, volume, timers, macOS reminders/notes) + permission tiers + confirmation | "Open Spotify and set a 10-minute timer" works |
| **4. Natural conversation** | Barge-in (interrupt while speaking), smart turn detection, latency tuning | You can cut JARVIS off mid-sentence and it stops |
| **5. Memory** | SQLite memory + remember/recall tools, user profile injected into the prompt | "Remember I like my coffee black" persists across restarts |
| **6. MCP integrations** | MCP client; connect calendar, email, files, browser (Playwright), GitHub | "What's on my calendar tomorrow?" / "Summarize my unread email" |
| **7. Proactivity** | Scheduler: morning briefing, reminders, alerts | JARVIS speaks first: "Good morning. You have three meetings today…" |
| **8. Vision & polish** | Screen/camera understanding, menu-bar app UI, launch at login, local LLM offline mode | Feels like a product |

Stretch: smart-home via Home Assistant MCP, custom wake word "Jarvis" (no "hey"), custom voice, phone access via Telegram/iMessage channel, speech-to-speech mode.

---

## 8. Decisions (2 Nov 2026)

| Decision | Choice |
|---|---|
| Who writes the code | Claude Code (this assistant), phase by phase |
| API provider | **Groq** (one key for brain + speech-to-text) |
| Language | **English only** |
| Interface | **Terminal** |
| Pipeline | Hand-rolled (recommended, pending confirmation; see below). Pipecat is the upgrade path. |

### What Groq gives us (checked against console.groq.com/docs, Nov 2026)

| Need | Groq model | Notes |
|---|---|---|
| **Brain** | `openai/gpt-oss-120b` (production) | ~500 tok/s, $0.15 / $0.60 per 1M tokens, 131k context, tool use + JSON mode + Groq built-in tools (web search). `reasoning_effort=low` for fast voice replies. |
| Brain (fastest) | `openai/gpt-oss-20b` | ~1000 tok/s, half the price, a bit less smart |
| Brain (alt) | `qwen/qwen3.8-27b` (preview) | Supports *parallel* tool calls; preview = may disappear |
| **Speech-to-text** | `whisper-large-v3-turbo` | $0.04/audio-hour, 10 s minimum billed per request. Zero local setup. |
| Text-to-speech | `canopylabs/orpheus-v1-english` (preview) | Free tier only **100 requests/day**, too low for daily use → **use local Kokoro instead** |

Llama 3.x models were removed from free/Developer accounts on 16 Aug 2026 (enterprise-only now), so don't use them.

**Free-tier limits to design around** (per org; source: third-party summary of Groq's Sep 2026 limits, so verify in console → Limits):
- gpt-oss: 30 requests/min, 1,000 requests/day, **8,000 tokens/min**, 200k tokens/day.
  → One voice turn ≈ 1.5–3k tokens (system prompt + history + tool schemas), so only ~3–4 turns/minute. Every tool-call round trip is another request. **Keep history trimmed and tool schemas lean**, or upgrade to the pay-as-you-go Developer tier.
- Whisper: 20 requests/min, 2,000/day, 7,200 audio-seconds/hour.

**Updated stack:**
`openWakeWord (local) → Silero VAD (local) → Groq Whisper turbo (cloud) → Groq gpt-oss-120b (cloud) → Kokoro (local)`.
Local Parakeet stays the option if we want offline / private speech-to-text later.

### Decision 4 explained: hand-rolled pipeline vs Pipecat

Think of it like building a car: **hand-rolled** = we assemble the engine ourselves from parts; **Pipecat** = buy a chassis with the engine mounted and bolt our parts on.

| | Hand-rolled (our own ~300 lines of glue) | Pipecat framework |
|---|---|---|
| How it works | Our own state machine: SLEEPING → LISTENING → THINKING → SPEAKING. We call mic, VAD, Groq, Kokoro directly. | Everything is a "processor" in a pipeline passing "frames" (audio, text, interruptions) along. |
| Understanding | You can read every line and know exactly what happens | Much happens inside the framework |
| Barge-in / smart turn-taking | We build it (Phase 4, the hardest part) | Built in |
| Groq / Kokoro support | We write small adapters | Has plugins (verify versions when we adopt it) |
| Debugging | Plain Python, easy | Need to learn its concepts first |
| Lock-in | None | Tied to its abstractions and release cycle |

**Recommendation:** hand-roll Phases 0–3 (turn-based conversation is simple and you learn how it all works). In Phase 4, if barge-in gets messy, swap only the audio layer for Pipecat. Brain, tools and memory stay untouched because they sit behind our own interfaces.

---

## Sources

**Assistants & projects**
- OpenClaw: https://github.com/openclaw/openclaw · https://docs.openclaw.ai/nodes/computer-use · https://www.macstories.net/stories/clawdbot-showed-me-what-the-future-of-personal-ai-assistants-looks-like/
- OpenVoiceOS history: https://blog.openvoiceos.org/posts/2023-02-15-a-brief-history-of-open-voice-os · https://www.cnx-software.com/2025/02/24/the-openvoiceos-foundation-aims-to-enable-open-source-privacy-and-customization-for-voice-assistants/
- llm-guy/jarvis: https://github.com/llm-guy/jarvis
- kishanrajput23 Jarvis: https://github.com/kishanrajput23/Jarvis-Desktop-Voice-Assistant
- GitHub topic jarvis-ai: https://github.com/topics/jarvis-ai
- Chuchote: https://pypi.org/project/chuchote/
- Local voice assistant (openWakeWord + Whisper + Ollama + Piper): https://github.com/altynaiiskakova/local-voice-assistant
- Local stack guides: https://www.promptquorum.com/power-local-llm/build-local-voice-assistant-2026 · https://www.kunalganglani.com/blog/local-ai-voice-assistant-whisper-piper-ollama
- Home Assistant voice on Seeed hardware: https://sensecraft.seeed.cc/ai-lab/en/projects/8

**Frameworks, turn-taking, speech-to-speech**
- Pipecat vs LiveKit: https://www.forasoft.com/blog/article/pipecat-vs-livekit-agents · https://livekit.com/blog/best-open-source-voice-and-video-ai-agent-frameworks
- HF speech-to-speech: https://pypi.org/project/speech-to-speech/
- Turn detection: https://livekit.com/blog/turn-detection-voice-agents-vad-endpointing-model-based-detection · https://livekit.com/blog/turn-detection-and-interruption-handling · https://docs.pinecall.io/concepts/turn-detection · https://micdrop.dev/docs/client/latency.md
- NemotronLabs VoiceChat (full-duplex + tools): https://arxiv.org/pdf/2609.21967
- Kyutai / full-duplex overview: https://www.coval.dev/blog/the-future-of-speech-to-speech-ai-inside-gradium-and-kyutai-s-approach-to-full-duplex-conversation
- Cloud realtime APIs: https://tokenmix.ai/blog/voice-ai-api-realtime-vs-gemini-live-vs-elevenlabs-2026 · https://edesy.in/ai-voice-assistant/blog/gemini-live-vs-openai-realtime

**STT / TTS**
- STT on Mac: https://localaimaster.com/blog/best-speech-to-text-mac · https://spokenly.app/blog/parakeet-vs-whisper · https://www.onresonant.com/resources/local-stt-models-2026 · https://contracollective.com/blog/local-speech-to-text-whisper-parakeet-mlx-m5-max-2026
- TTS: https://pinggy.io/blog/best_open_source_self_hosted_text_to_speech_models/ · https://codesota.com/guides/tts-models · https://github.com/gabrimatic/kokoro-mlx

**Tools & memory**
- MCP: https://mcp.so/blog/what-is-mcp · https://parallel.ai/learn/mcp-server
- Memory: https://atlan.com/know/best-ai-agent-memory-frameworks-2026/ · https://vectorize.io/articles/mem0-vs-letta · https://dev.to/plur9/mem0-vs-letta-vs-zep-which-should-you-use-for-agent-memory-1n8m

**Claude API**: model IDs and pricing from Anthropic's Claude API reference (cached 2026-10-06).

**Groq**
- Models & pricing: https://console.groq.com/docs/models
- Tool use support table: https://console.groq.com/docs/tool-use/overview
- Free tier limits & Llama removal: https://klymentiev.com/blog/groq-pricing · https://dev.to/unicodefawn/free-speech-to-text-and-text-to-speech-api-quotas-oct-2026-with-the-catches-17hg · https://www.maximem.ai/blog/groq-api-pricing-2026

> ⚠️ Many 2026 sources are vendor blogs with conflicting benchmark numbers. Treat latency/accuracy figures as rough until we measure on this Mac in Phase 1.
