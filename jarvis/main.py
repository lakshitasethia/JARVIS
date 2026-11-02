"""Phase 1: push-to-talk voice. Press Enter to talk, or type a message."""

import time

import groq
from rich.console import Console

from jarvis.audio.recorder import Recorder, duration, peak_level, to_wav_bytes
from jarvis.brain.conversation import Conversation
from jarvis.brain.llm import GroqLLM
from jarvis.brain.persona import system_prompt
from jarvis.config import Settings, load_settings
from jarvis.listen.stt import GroqSTT
from jarvis.speak.sentences import SentenceSplitter, clean_for_speech
from jarvis.speak.speaker import Speaker
from jarvis.speak.tts import KokoroTTS, SayTTS

HELP = (
    "[dim]Enter on an empty line = talk (Enter again to stop) · or just type\n"
    "Commands: /mute (toggle voice) · /reset (forget chat) · /help · /quit · Ctrl+C stops JARVIS[/dim]"
)
MIN_SECONDS = 0.4
MIN_PEAK = 300  # below this the recording is effectively silent


def groq_error_message(e: groq.APIError) -> str:
    if isinstance(e, groq.AuthenticationError):
        return "Groq rejected the API key. Check GROQ_API_KEY in .env."
    if isinstance(e, groq.RateLimitError):
        return "Groq rate limit hit (free tier). Wait a moment and try again."
    if isinstance(e, groq.APIConnectionError):
        return "Can't reach Groq. Check your internet connection."
    if isinstance(e, groq.APIStatusError):
        return f"Groq error {e.status_code}: {e.message}"
    return f"Groq error: {e}"


def build_speaker(s: Settings) -> Speaker | None:
    if s.tts_engine == "none":
        return None
    tts = SayTTS() if s.tts_engine == "say" else KokoroTTS(
        s.kokoro_model_path, s.kokoro_voices_path, s.voice, s.speech_speed
    )
    return Speaker(tts)


class Jarvis:
    def __init__(self, settings: Settings, console: Console):
        self.s = settings
        self.console = console
        self.llm = GroqLLM(settings.groq_api_key, settings.model, settings.reasoning_effort)
        self.stt = GroqSTT(settings.groq_api_key, settings.stt_model)
        self.recorder = Recorder()
        self.speaker = build_speaker(settings)
        self.muted = False
        self.convo = Conversation(system_prompt(settings.user_name), settings.max_history_messages)

    # --- ears -----------------------------------------------------------------

    def listen(self) -> tuple[str | None, float]:
        """Record until Enter, transcribe. Returns (text or None, seconds spent transcribing)."""
        self.recorder.start()
        try:
            self.console.input("[bold red]● listening…[/bold red] [dim](Enter to stop)[/dim] ")
        finally:
            audio = self.recorder.stop()

        if duration(audio) < MIN_SECONDS:
            self.console.print("[dim]Too short. Hold on a little longer next time.[/dim]")
            return None, 0.0
        if peak_level(audio) < MIN_PEAK:
            self.console.print(
                "[yellow]I didn't hear anything.[/yellow] [dim]If you spoke, check that your terminal app "
                "has microphone access: System Settings → Privacy & Security → Microphone.[/dim]"
            )
            return None, 0.0

        start = time.perf_counter()
        text = self.stt.transcribe(to_wav_bytes(audio))
        stt_seconds = time.perf_counter() - start
        if not text:
            self.console.print("[dim]Couldn't make that out. Try again?[/dim]")
            return None, stt_seconds
        self.console.print(f"[bold green]you ›[/bold green] {text}")
        return text, stt_seconds

    # --- brain + mouth ----------------------------------------------------------

    def respond(self, text: str, turn_start: float, stt_seconds: float | None = None) -> None:
        self.convo.add_user(text)
        speaker = None if self.muted else self.speaker
        splitter = SentenceSplitter()
        if speaker:
            speaker.begin_turn()

        def speak(sentences: list[str]) -> None:
            if speaker:
                for sentence in sentences:
                    if spoken := clean_for_speech(sentence):
                        speaker.say(spoken)

        self.console.print("[bold cyan]jarvis ›[/bold cyan] ", end="")
        reply, first_token_at = [], None
        try:
            for chunk in self.llm.stream(self.convo.messages()):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                reply.append(chunk)
                self.console.print(chunk, end="", markup=False, highlight=False)
                speak(splitter.feed(chunk))
            speak(splitter.flush())
            self.console.print()
            if speaker:
                speaker.wait()
        except KeyboardInterrupt:
            if speaker:
                speaker.stop()
            self.console.print(" [dim](interrupted)[/dim]")

        full = "".join(reply).strip()
        if full:
            self.convo.add_assistant(full)
        else:
            self.convo.history.pop()  # nothing came back; don't leave a dangling user turn

        if self.s.show_timing and first_token_at is not None:
            parts = []
            if stt_seconds is not None:
                parts.append(f"heard {stt_seconds:.2f}s")
            parts.append(f"first word {first_token_at - turn_start:.2f}s")
            if speaker and speaker.first_audio_at:
                parts.append(f"first sound {speaker.first_audio_at - turn_start:.2f}s")
            self.console.print(f"[dim]{' · '.join(parts)}[/dim]")

    # --- loop -------------------------------------------------------------------

    def run(self) -> None:
        voice = "muted" if not self.speaker else f"{self.s.tts_engine}"
        self.console.print(
            f"[bold cyan]JARVIS[/bold cyan] online · brain: {self.llm.model} · ears: {self.stt.model} · voice: {voice}"
        )
        self.console.print(HELP)
        if self.speaker:
            self.speaker.warm_up()

        while True:
            try:
                typed = self.console.input("\n[bold green]you ›[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if typed in ("/quit", "/exit"):
                break
            if typed == "/help":
                self.console.print(HELP)
                continue
            if typed == "/reset":
                self.convo.reset()
                self.console.print("[dim]Conversation cleared.[/dim]")
                continue
            if typed == "/mute":
                self.muted = not self.muted
                self.console.print(f"[dim]Voice {'off' if self.muted else 'on'}.[/dim]")
                continue

            try:
                if typed:
                    self.respond(typed, time.perf_counter())
                else:
                    text, stt_seconds = self.listen()
                    if text:
                        # latency clock starts when you stopped talking
                        self.respond(text, time.perf_counter() - stt_seconds, stt_seconds)
            except KeyboardInterrupt:
                self.console.print("[dim](cancelled)[/dim]")
            except groq.APIError as e:
                if self.convo.history and self.convo.history[-1]["role"] == "user":
                    self.convo.history.pop()
                self.console.print(f"\n[red]{groq_error_message(e)}[/red]")

        self.console.print("\n[cyan]Goodbye.[/cyan]")
        if self.speaker:
            self.speaker.close()


def main() -> None:
    console = Console()
    Jarvis(load_settings(), console).run()


if __name__ == "__main__":
    main()
