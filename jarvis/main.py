"""JARVIS main loop.

  uv run jarvis          hands-free: say "Hey Jarvis", then talk
  uv run jarvis --ptt    push-to-talk / typing mode (press Enter to talk, or type)
"""

import argparse
import re
import time

import groq
from rich.console import Console

from jarvis.audio.mic import MicStream
from jarvis.audio.recorder import Recorder, duration, peak_level, to_wav_bytes
from jarvis.brain.conversation import Conversation
from jarvis.brain.llm import GroqLLM
from jarvis.brain.persona import system_prompt
from jarvis.config import Settings, load_settings
from jarvis.listen.endpoint import EndpointConfig, Status, UtteranceCollector
from jarvis.listen.stt import GroqSTT
from jarvis.speak import earcons
from jarvis.speak.sentences import SentenceSplitter, clean_for_speech
from jarvis.speak.speaker import Speaker
from jarvis.speak.tts import KokoroTTS, SayTTS

PTT_HELP = (
    "[dim]Enter on an empty line = talk (Enter again to stop) · or just type\n"
    "Commands: /mute (toggle voice) · /reset (forget chat) · /help · /quit · Ctrl+C stops JARVIS[/dim]"
)
HANDSFREE_HELP = (
    "[dim]Say [bold]\"Hey Jarvis\"[/bold], wait for the chime, then talk. After JARVIS answers you can "
    "keep talking without the wake word.\nSay \"never mind\" / \"go to sleep\" to end · Ctrl+C stops "
    "JARVIS speaking · Ctrl+C while idle quits[/dim]"
)
MIN_SECONDS = 0.4
MIN_PEAK = 300  # below this a push-to-talk recording is effectively silent
SLEEP_PHRASES = {"never mind", "nevermind", "go to sleep", "stop", "cancel", "that's all", "that is all", "nothing"}


def is_sleep_phrase(text: str) -> bool:
    normalized = re.sub(r"[^a-z' ]", "", text.lower()).strip()
    normalized = re.sub(r"^(hey |ok |okay )?jarvis ?", "", normalized).strip()
    return normalized in SLEEP_PHRASES


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
        self.speaker = build_speaker(settings)
        self.muted = False
        self.convo = Conversation(system_prompt(settings.user_name), settings.max_history_messages)

    def banner(self, mode: str) -> None:
        voice = self.s.tts_engine if self.speaker else "none"
        self.console.print(
            f"[bold cyan]JARVIS[/bold cyan] online · {mode} · brain: {self.llm.model} · "
            f"ears: {self.stt.model} · voice: {voice}"
        )
        if self.speaker:
            self.speaker.warm_up()

    def transcribe(self, audio) -> tuple[str, float]:
        start = time.perf_counter()
        text = self.stt.transcribe(to_wav_bytes(audio))
        return text, time.perf_counter() - start

    def handle_groq_error(self, e: groq.APIError) -> None:
        if self.convo.history and self.convo.history[-1]["role"] == "user":
            self.convo.history.pop()
        self.console.print(f"\n[red]{groq_error_message(e)}[/red]")

    # --- brain + mouth (shared by both modes) -------------------------------------

    def respond(self, text: str, turn_start: float, stages: list[tuple[str, float]] = ()) -> None:
        """Stream the reply to the screen and speak it sentence by sentence.

        turn_start: when the user stopped talking/typing (for latency numbers).
        stages: earlier steps to show in the timing line, e.g. [("heard", 0.4)].
        """
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
            parts = [f"{label} {secs:.2f}s" for label, secs in stages]
            parts.append(f"first word {first_token_at - turn_start:.2f}s")
            if speaker and speaker.first_audio_at:
                parts.append(f"first sound {speaker.first_audio_at - turn_start:.2f}s")
            self.console.print(f"[dim]{' · '.join(parts)}[/dim]")

    # --- hands-free mode --------------------------------------------------------------

    def run_handsfree(self) -> None:
        from jarvis.listen.vad import VoiceActivity
        from jarvis.listen.wake import WakeWordDetector

        self.banner("hands-free")
        wake = WakeWordDetector(self.s.wake_threshold)
        vad = VoiceActivity()
        mic = MicStream()
        self.console.print(HANDSFREE_HELP)
        mic.start()
        try:
            while True:
                self.console.print("\n[dim]💤 sleeping, say \"Hey Jarvis\"[/dim]")
                self._wait_for_wake(mic, wake)
                self._chime(earcons.listening)
                self._conversation(mic, vad)
                self._chime(earcons.sleeping)
                wake.reset()
                mic.drain()
        except KeyboardInterrupt:
            pass
        finally:
            mic.close()
            self.console.print("\n[cyan]Goodbye.[/cyan]")
            if self.speaker:
                self.speaker.close()

    def _chime(self, make) -> None:
        if self.speaker and not self.muted:
            self.speaker.play(make(self.speaker.tts.sample_rate))

    def _wait_for_wake(self, mic: MicStream, wake) -> None:
        while True:
            frame = mic.read()
            if frame is not None and wake.detected(frame):
                return

    def _conversation(self, mic: MicStream, vad) -> None:
        """One wake-up: listen, answer, then keep listening for follow-ups until quiet."""
        first = True
        while True:
            timeout = self.s.listen_timeout if first else self.s.followup_seconds
            heard = self._capture(mic, vad, timeout)
            if heard is None:
                return
            audio, stopped_at, waited = heard
            try:
                text, stt_seconds = self.transcribe(audio)
            except groq.APIError as e:
                self.handle_groq_error(e)
                return
            if not text:
                continue
            self.console.print(f"[bold green]you ›[/bold green] {text}")
            if is_sleep_phrase(text):
                return
            try:
                self.respond(text, stopped_at, [("end of turn", waited), ("heard", stt_seconds)])
            except groq.APIError as e:
                self.handle_groq_error(e)
                return
            # Ignore what the mic picked up while JARVIS was talking (its own voice).
            time.sleep(0.15)
            mic.drain()
            first = False

    def _capture(self, mic: MicStream, vad, timeout: float):
        """Collect one utterance. Returns (audio, when user stopped talking, end-of-turn wait) or None."""
        vad.reset()
        collector = UtteranceCollector(
            EndpointConfig(silence_seconds=self.s.silence_seconds, no_speech_timeout=timeout)
        )
        self.console.print("[bold red]● listening…[/bold red]")
        speaking_shown = False
        while True:
            frame = mic.read()
            if frame is None:
                continue
            status = collector.add(frame, vad.speech_probability(frame))
            if status is Status.SPEAKING and not speaking_shown:
                self.console.print("[dim]  hearing you…[/dim]")
                speaking_shown = True
            elif status is Status.TIMEOUT:
                return None
            elif status is Status.DONE:
                waited = collector.trailing_silence_seconds
                return collector.audio(), time.perf_counter() - waited, waited

    # --- push-to-talk / typing mode -----------------------------------------------------

    def run_ptt(self) -> None:
        self.banner("push-to-talk")
        self.console.print(PTT_HELP)
        recorder = Recorder()

        while True:
            try:
                typed = self.console.input("\n[bold green]you ›[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if typed in ("/quit", "/exit"):
                break
            if typed == "/help":
                self.console.print(PTT_HELP)
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
                    continue
                recorder.start()
                try:
                    self.console.input("[bold red]● listening…[/bold red] [dim](Enter to stop)[/dim] ")
                finally:
                    audio = recorder.stop()
                if duration(audio) < MIN_SECONDS:
                    self.console.print("[dim]Too short. Hold on a little longer next time.[/dim]")
                    continue
                if peak_level(audio) < MIN_PEAK:
                    self.console.print(
                        "[yellow]I didn't hear anything.[/yellow] [dim]If you spoke, check that your terminal "
                        "app has microphone access: System Settings → Privacy & Security → Microphone.[/dim]"
                    )
                    continue
                text, stt_seconds = self.transcribe(audio)
                if not text:
                    self.console.print("[dim]Couldn't make that out. Try again?[/dim]")
                    continue
                self.console.print(f"[bold green]you ›[/bold green] {text}")
                # latency clock starts when you stopped talking
                self.respond(text, time.perf_counter() - stt_seconds, [("heard", stt_seconds)])
            except KeyboardInterrupt:
                self.console.print("[dim](cancelled)[/dim]")
            except groq.APIError as e:
                self.handle_groq_error(e)

        self.console.print("\n[cyan]Goodbye.[/cyan]")
        if self.speaker:
            self.speaker.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="jarvis", description="Your personal JARVIS.")
    parser.add_argument("--ptt", action="store_true", help="push-to-talk / typing mode instead of hands-free")
    args = parser.parse_args()
    jarvis = Jarvis(load_settings(), Console())
    jarvis.run_ptt() if args.ptt else jarvis.run_handsfree()


if __name__ == "__main__":
    main()
