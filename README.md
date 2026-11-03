# JARVIS

A personal JARVIS-style AI assistant for macOS, built step by step.
See [research.md](research.md) for the research, architecture, and roadmap.

**Status:** Phase 5: memory. JARVIS remembers you across sessions (local SQLite). Earlier: Phase 4, natural conversation. Talk over JARVIS to interrupt it (echo-cancelled, works on laptop speakers). Earlier: Phase 3, tools. JARVIS can act: open/quit apps, volume, Spotify/Music, timers, weather, Apple Reminders, web search.

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
| "Unlock my phone" *(asks first)* / "Lock my phone" / "Show my phone" | `unlock_phone`, `lock_phone`, `show_phone` (Android over Wi-Fi; see Phone) |
| "Open WhatsApp on my phone" / "Go home on my phone" | `open_phone_app` (unlocks first if needed), `phone_home` |
| "I'm vegetarian" / "Remember my sister is Riya" | `remember` (also `update_memory`, `forget`, `forget_everything` *(asks first)*) |

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
**Interrupt by talking:** just start speaking while JARVIS talks: it stops within ~0.3 s and listens
("Stop, what's the capital of France?"). Ctrl+C also stops it; Ctrl+C while idle quits.
Session logs go to `logs/` (git-ignored) so odd behaviour can be diagnosed.

**Push-to-talk / typing:** `uv run jarvis --ptt`, then press Enter on an empty line to talk (Enter again to stop), or type.
Commands: `/mute`, `/reset`, `/help`, `/quit`.

The first time you talk, macOS will ask to give your terminal app microphone access. Allow it.

## Phone (Android, e.g. Galaxy S25)

JARVIS can wake and unlock your phone, lock it, and mirror it in a window on the Mac, over Wi-Fi via
Android's official Wireless debugging. One-time setup:

1. **Phone:** Settings → About phone → Software information → tap **Build number** 7 times (enables Developer options).
2. **Phone:** Settings → Developer options → **Wireless debugging** → on. Phone and Mac must be on the same Wi-Fi.
   If the switch won't turn on: Settings → Security and privacy → **Auto Blocker** → off.
3. **Phone:** Wireless debugging → **Pair device with pairing code**. **Mac:** `adb pair <ip:port>` and enter the code.
4. **Mac:** `adb connect <ip:port>` (the address on the main Wireless debugging screen), then `adb devices` should list the phone.
5. **Mac:** save your PIN in the macOS Keychain (typed hidden, stored encrypted):
   `security add-generic-password -a jarvis -s jarvis-phone-pin -w`

Safety rules (enforced in code): no unlocking while the Mac is locked; a spoken "yes" is required; the PIN is
only typed when the phone is verifiably at its lock screen; one attempt, then a 5-minute cooldown (Samsung
locks you out after repeated wrong PINs); the PIN never goes to the AI model or the logs.
Limits: after the phone restarts, Wireless debugging is off and the first unlock must be by hand.
Anyone who can talk to your *unlocked* Mac could ask JARVIS to unlock your phone. Keep that in mind.

## Always on (no terminal)

Run JARVIS in the background so it starts when you log in. Just say "Hey Jarvis" from anywhere:

```bash
uv run jarvis-service install     # builds ~/Applications/JARVIS.app and adds it to login
```

macOS will ask **"JARVIS would like to access the microphone"**. Click **Allow**, then run
`uv run jarvis-service restart`. Other commands: `status`, `logs` (`-f` to follow), `stop`, `start`,
`restart` (after changing `.env` or the code), `uninstall`.

- No Dock icon or window. The orange mic dot in the menu bar shows it's listening.
- Only one JARVIS listens at a time. To type to it while the service runs: `uv run jarvis --ptt`.
- If the mic or speaker stops working (sleep, unplugging headphones), JARVIS restarts itself.
- Output goes to `logs/service.log`.

## Tests

```bash
uv run pytest
```

## Layout

```
jarvis/
├── config.py            settings from .env
├── main.py              main loop: listen → think → speak
├── log.py               session log (logs/*.jsonl)
├── memory.py            long-term memory (data/memory.db, local only)
├── audio/
│   ├── echo.py          echo cancellation (WebRTC AEC3): removes JARVIS's voice from the mic
│   ├── mic.py           always-on mic stream (80 ms frames) + processing thread
│   └── recorder.py      push-to-talk capture
├── listen/
│   ├── wake.py          "Hey Jarvis" wake word (openWakeWord, local)
│   ├── vad.py           voice activity detection (Silero, local)
│   ├── endpoint.py      decides when you started / finished talking
│   ├── bargein.py       notices you talking over JARVIS
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
├── service.py           background service: JARVIS.app + login item (jarvis-service)
├── service_launcher.c   tiny app launcher so macOS permissions are granted to "JARVIS"
└── speak/
    ├── sentences.py     split streaming text into sentences
    ├── tts.py           text-to-speech (Kokoro, macOS say)
    ├── speaker.py       plays sentences in order; synth N+1 while N plays
    ├── earcons.py       listening / sleeping chimes
    └── download.py      fetches Kokoro model files
```
