"""JARVIS main loop.

  uv run jarvis          hands-free: say "Hey Jarvis", then talk
  uv run jarvis --ptt    push-to-talk / typing mode (press Enter to talk, or type)
"""

import argparse
import fcntl
import re
import signal
import sys
import threading
import time
from pathlib import Path

import groq
from rich.console import Console

from jarvis import log
from jarvis.audio.echo import EchoCanceller, EchoReference
from jarvis.audio.mic import MicStream
from jarvis.audio.recorder import Recorder, duration, peak_level, to_wav_bytes
from jarvis.brain.agent import Agent, TextDelta, ToolFinished, ToolStarted, summarize_arguments
from jarvis.brain.conversation import Conversation
from jarvis.brain.llm import GroqLLM
from jarvis.brain.persona import system_prompt
from jarvis.config import Settings, load_settings
from jarvis.listen.bargein import BargeInMonitor
from jarvis.listen.endpoint import EndpointConfig, Status, UtteranceCollector
from jarvis.listen.stt import GroqSTT, is_junk_transcript
from jarvis.listen.yesno import parse_yes_no
from jarvis.memory import MemoryStore
from jarvis.speak import earcons
from jarvis.speak.sentences import SentenceSplitter, clean_for_speech
from jarvis.speak.speaker import Speaker
from jarvis.speak.tts import KokoroTTS, SayTTS
from jarvis.tools import build_registry
from jarvis.tools.phone import AndroidPhone
from jarvis.tools.registry import DECLINED
from jarvis.tools.timers import TimerManager

PTT_HELP = (
    "[dim]Enter on an empty line = talk (Enter again to stop) · or just type\n"
    "Commands: /mute (toggle voice) · /reset (forget chat) · /tools · /help · /quit · Ctrl+C stops JARVIS[/dim]"
)
HANDSFREE_HELP = (
    "[dim]Say [bold]\"Hey Jarvis\"[/bold], wait for the chime, then talk. After JARVIS answers you can "
    "keep talking without the wake word.\nSay \"never mind\" / \"go to sleep\" to end · Ctrl+C stops "
    "JARVIS speaking · Ctrl+C while idle quits[/dim]"
)
MIN_SECONDS = 0.4
MIN_PEAK = 300  # below this a push-to-talk recording is effectively silent
FILLER_AFTER_SECONDS = 2.0  # say "One moment." if nothing has happened by then
SLEEP_PHRASES = {
    "never mind", "nevermind", "go to sleep", "stop", "cancel", "that's all", "that is all", "nothing",
    "goodbye", "good bye", "bye", "bye bye", "okay goodbye", "ok goodbye", "thanks goodbye",
    "thank you goodbye", "that's it", "that'll be all", "that will be all",
}


class BargeIn(Exception):
    """The user started talking while JARVIS was responding."""


class AudioDeviceLost(Exception):
    """Mic or speaker stopped working; restarting the process gets fresh audio streams."""


EXIT_RESTART = 75  # exit code that tells the background service to start JARVIS again
PROJECT_DIR = Path(__file__).resolve().parent.parent


def is_just_wake_word(text: str) -> bool:
    """'Hey Jarvis' on its own mid-conversation: the user re-woke it; nothing to answer."""
    return re.sub(r"[^a-z ]", "", text.lower()).strip() in {"jarvis", "hey jarvis", "ok jarvis", "okay jarvis"}


def is_sleep_phrase(text: str) -> bool:
    normalized = re.sub(r"[^a-z' ]", "", text.lower()).strip()
    normalized = re.sub(r"^(hey |ok |okay )?jarvis ?", "", normalized).strip()
    return normalized in SLEEP_PHRASES


def groq_error_message(e: groq.APIError) -> str:
    if isinstance(e, groq.AuthenticationError):
        return "Groq rejected the API key. Check GROQ_API_KEY in .env."
    if isinstance(e, groq.RateLimitError):
        return (
            "Groq's free-tier limit is used up for the moment (8,000 tokens a minute, 1,000 requests a day). "
            "Give it a minute, or upgrade to Groq's pay-as-you-go Developer tier for much higher limits."
        )
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
        self.llm = GroqLLM(
            settings.groq_api_key,
            settings.model,
            settings.reasoning_effort,
            settings.web_search,
            on_wait=lambda secs: self.console.print(
                f"\n[dim yellow]  (Groq free-tier limit reached, waiting {secs:.0f}s)[/dim yellow]"
            ),
        )
        self.stt = GroqSTT(settings.groq_api_key, settings.stt_model)
        self.speaker = build_speaker(settings)
        self.muted = False
        self.timers = TimerManager(self.announce)
        self.memory = MemoryStore(PROJECT_DIR / "data" / "memory.db")
        phone = AndroidPhone(settings.phone_serial, settings.adb_path)
        self.agent = Agent(self.llm, build_registry(self.timers, settings.home_location, self.memory, phone))
        self.convo = Conversation(
            lambda: system_prompt(settings.user_name, settings.home_location, self.memory.prompt_section()),
            settings.max_history_messages,
        )
        # How to ask the user yes/no; each mode installs its own.
        self.confirm = self.confirm_typed
        self._monitor: BargeInMonitor | None = None  # active while responding hands-free

    def banner(self, mode: str) -> None:
        voice = self.s.tts_engine if self.speaker else "none"
        search = " + web search" if self.llm.web_search else ""
        self.console.print(
            f"[bold cyan]JARVIS[/bold cyan] online · {mode} · brain: {self.llm.model} · "
            f"ears: {self.stt.model} · voice: {voice} · tools: {len(self.agent.tools.names())}{search}"
        )
        if self.speaker:
            self.speaker.warm_up()

    def transcribe(self, audio) -> tuple[str, float]:
        start = time.perf_counter()
        text = self.stt.transcribe(to_wav_bytes(audio))
        seconds = time.perf_counter() - start
        junk = is_junk_transcript(text)
        log.event("transcript", text=text, audio_seconds=round(len(audio) / 16_000, 2), junk=junk)
        return ("" if junk else text), seconds

    def say(self, text: str) -> None:
        """Speak a fixed line (not from the LLM)."""
        if self.speaker and not self.muted:
            self.speaker.say(text)

    def announce(self, message: str) -> None:
        """Called from timer threads: grab attention, then speak."""
        self.console.print(f"\n[bold yellow]⏰ {message}[/bold yellow]")
        if self.speaker and not self.muted:
            self.speaker.play(earcons.listening(self.speaker.tts.sample_rate))
            self.speaker.say(f"{self.s.user_name.capitalize()}, {message[0].lower()}{message[1:]}")

    # --- confirmation -------------------------------------------------------------------

    def confirm_typed(self, question: str) -> bool:
        if self.speaker and not self.muted:
            self.speaker.wait()
            self.say(question)
        answer = self.console.input(f"\n[bold yellow]? {question}[/bold yellow] [dim](y/n)[/dim] ")
        return parse_yes_no(answer) is True

    def confirm_spoken(self, question: str, mic: MicStream, vad) -> bool:
        monitor = self._monitor
        if monitor:
            monitor.pause()
        try:
            return self._ask_spoken(question, mic, vad)
        finally:
            if monitor:
                monitor.resume()

    def _ask_spoken(self, question: str, mic: MicStream, vad) -> bool:
        self.console.print(f"\n[bold yellow]? {question}[/bold yellow] [dim](say yes or no)[/dim]")
        if self.speaker and not self.muted:
            self.speaker.wait()
            self.say(question)
            self.speaker.wait()
        time.sleep(0.15)
        mic.drain()
        heard = self._capture(mic, vad, timeout=6.0, announce=False)
        if heard is None:
            self.console.print("[dim]  no answer, so I won't.[/dim]")
            return False
        text, _ = self.transcribe(heard[0])
        decision = parse_yes_no(text)
        self.console.print(f"[bold green]you ›[/bold green] {text} [dim]→ {'yes' if decision else 'no'}[/dim]")
        return decision is True

    # --- brain + mouth (shared by both modes) -------------------------------------------

    def respond(
        self,
        text: str,
        turn_start: float,
        stages: list[tuple[str, float]] = (),
        interrupt: threading.Event | None = None,
    ) -> bool:
        """Run the agent on `text`: stream the reply to the screen, speak it, run tools.

        turn_start: when the user stopped talking/typing (for latency numbers).
        stages: earlier steps to show in the timing line, e.g. [("heard", 0.4)].
        interrupt: set by the barge-in monitor when the user starts talking over JARVIS.
        Returns True if the user cut in before JARVIS said anything (they were still talking).
        """
        history_len = len(self.convo.history)
        log.event("user", text=text)
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

        # If nothing at all happens for a while (e.g. a web search), say so instead of going silent.
        filler = threading.Timer(FILLER_AFTER_SECONDS, lambda: speak(["One moment."]))
        filler.daemon = True
        if speaker:
            filler.start()

        self.console.print("[bold cyan]jarvis ›[/bold cyan] ", end="")
        reply: list[str] = []
        first_token_at = None
        mid_line = True  # the cursor is on a "jarvis ›" line
        line_has_text = False  # ...and reply text has been printed on it
        try:
            for event in self.agent.run(self.convo, lambda q: self.confirm(q)):
                if interrupt is not None and interrupt.is_set():
                    raise BargeIn
                filler.cancel()
                if isinstance(event, TextDelta):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    if not mid_line:
                        self.console.print("[bold cyan]jarvis ›[/bold cyan] ", end="")
                        mid_line = True
                    reply.append(event.text)
                    line_has_text = True
                    self.console.print(event.text, end="", markup=False, highlight=False)
                    speak(splitter.feed(event.text))
                elif isinstance(event, ToolStarted):
                    speak(splitter.flush())
                    if line_has_text:
                        self.console.print()
                    self.console.print(
                        f"[dim]  ⚙ {event.name}({summarize_arguments(event.arguments)})[/dim]", end=""
                    )
                elif isinstance(event, ToolFinished):
                    if event.ok:
                        mark = "[green]✓[/green]"
                    elif event.output == DECLINED:
                        mark = "[yellow]✗ declined[/yellow]"
                    else:
                        mark = f"[red]✗ {event.output[:120]}[/red]"
                    self.console.print(f" {mark}")
                    mid_line = line_has_text = False
            speak(splitter.flush())
            if mid_line:
                self.console.print()
            if speaker:
                while speaker.is_speaking():
                    if interrupt is not None and interrupt.is_set():
                        raise BargeIn
                    time.sleep(0.03)
        except (KeyboardInterrupt, BargeIn) as e:
            filler.cancel()
            if speaker:
                speaker.stop()
            cut_in = isinstance(e, BargeIn)
            self.console.print(" [dim](you cut in)[/dim]" if cut_in else " [dim](interrupted)[/dim]")
            log.event("interrupted", by="voice" if cut_in else "ctrl-c", partial="".join(reply))
            # A turn cut off mid-tool-call would leave the history inconsistent; keep just the gist.
            self.convo.rollback_to(history_len)
            self.convo.add_user(text)
            if partial := "".join(reply).strip():
                self.convo.add_assistant(partial)
            else:
                self.convo.rollback_to(history_len)
                return cut_in
        except groq.APIError as e:
            log.event("error", where="respond", error=repr(e))
            filler.cancel()
            if speaker:
                speaker.stop()
            self.convo.rollback_to(history_len)
            raise

        if self.s.show_timing and first_token_at is not None:
            parts = [f"{label} {secs:.2f}s" for label, secs in stages]
            parts.append(f"first word {first_token_at - turn_start:.2f}s")
            if speaker and speaker.first_audio_at:
                parts.append(f"first sound {speaker.first_audio_at - turn_start:.2f}s")
            self.console.print(f"[dim]{' · '.join(parts)}[/dim]")
        return False

    # --- hands-free mode ----------------------------------------------------------------

    def run_handsfree(self) -> None:
        from jarvis.listen.vad import VoiceActivity
        from jarvis.listen.wake import WakeWordDetector

        self.banner("hands-free")
        wake = WakeWordDetector(self.s.wake_threshold)
        vad = VoiceActivity()
        processor = None
        if self.speaker and self.s.echo_cancel:
            reference = EchoReference(self.speaker.tts.sample_rate)
            self.speaker.on_play = reference.push
            processor = EchoCanceller(reference).process
        mic = MicStream(processor=processor)
        if self.s.barge_in and self.speaker:
            how = "talk over JARVIS to interrupt" if processor else "barge-in without echo cancellation: use headphones"
            self.console.print(f"[dim]Echo cancellation {'on' if processor else 'off'} · {how}[/dim]")
        self.confirm = lambda question: self.confirm_spoken(question, mic, vad)
        self.console.print(HANDSFREE_HELP)
        mic.start()
        lost: AudioDeviceLost | None = None
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
        except AudioDeviceLost as e:
            lost = e
            log.event("audio_device_lost", reason=str(e))
            self.console.print(f"\n[yellow]Audio problem ({e}). Restarting to recover…[/yellow]")
        finally:
            mic.close()
            self._shutdown()
        if lost:
            sys.exit(EXIT_RESTART)  # the background service starts us again with fresh audio

    def _chime(self, make) -> None:
        if self.speaker and not self.muted:
            self.speaker.play(make(self.speaker.tts.sample_rate))

    def _wait_for_wake(self, mic: MicStream, wake) -> None:
        while True:
            frame = mic.read()
            if frame is None:
                if mic.stalled():
                    raise AudioDeviceLost("no audio from the microphone")
                continue
            if wake.detected(frame):
                if self.speaker and self.speaker.broken:
                    raise AudioDeviceLost("the speaker stopped working")
                return

    def _conversation(self, mic: MicStream, vad) -> None:
        """One wake-up: listen, answer, then keep listening for follow-ups until quiet."""
        first = True
        pending: list = []  # audio from a barge-in: the start of the user's next utterance
        unfinished = ""  # user text that was cut off by their own barge-in before JARVIS replied
        while True:
            timeout = self.s.listen_timeout if first else self.s.followup_seconds
            heard = self._capture(mic, vad, timeout, prefill=pending)
            pending = []
            if heard is None:
                return
            audio, stopped_at, waited = heard
            try:
                text, stt_seconds = self.transcribe(audio)
                if not text:
                    continue
                if unfinished:
                    text, unfinished = f"{unfinished} {text}", ""
                self.console.print(f"[bold green]you ›[/bold green] {text}")
                if is_sleep_phrase(text):
                    return
                if is_just_wake_word(text):
                    self._chime(earcons.listening)  # "I'm still here, go on"
                    continue
                monitor = self._start_monitor(mic, vad)
                try:
                    still_talking = self.respond(
                        text,
                        stopped_at,
                        [("end of turn", waited), ("heard", stt_seconds)],
                        interrupt=monitor.triggered if monitor else None,
                    )
                finally:
                    if monitor:
                        pending = monitor.stop()
                        self._monitor = None
                if still_talking:
                    unfinished = text  # they'd only paused; merge with what comes next
            except groq.APIError as e:
                self.console.print(f"\n[red]{groq_error_message(e)}[/red]")
                return
            if not pending:
                # Ignore what the mic picked up while JARVIS was talking (its own voice).
                time.sleep(0.15)
                mic.drain()
            first = False

    def _start_monitor(self, mic: MicStream, vad) -> BargeInMonitor | None:
        if not (self.s.barge_in and self.speaker and not self.muted):
            return None

        def on_barge_in() -> None:
            self.speaker.stop()
            self.speaker.unduck()

        self._monitor = BargeInMonitor(
            mic, vad, on_barge_in, on_maybe=self.speaker.duck, on_clear=self.speaker.unduck
        ).start()
        return self._monitor

    def _capture(self, mic: MicStream, vad, timeout: float, announce: bool = True, prefill: list = ()):
        """Collect one utterance. Returns (audio, when user stopped talking, end-of-turn wait) or None.

        prefill: (frame, speech probability) pairs already heard, e.g. by the barge-in monitor.
        """
        collector = UtteranceCollector(
            EndpointConfig(silence_seconds=self.s.silence_seconds, no_speech_timeout=timeout)
        )
        if prefill:
            for frame, prob in prefill:  # VAD state carries on from the monitor; don't reset it
                collector.add(frame, prob)
        else:
            vad.reset()
        if announce:
            self.console.print("[bold red]● listening…[/bold red]")
        speaking_shown = not announce
        while True:
            frame = mic.read()
            if frame is None:
                if mic.stalled():
                    raise AudioDeviceLost("no audio from the microphone")
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
        self.confirm = self.confirm_typed
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
            if typed == "/tools":
                self.console.print("[dim]" + ", ".join(self.agent.tools.names()) + "[/dim]")
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
                self.console.print(f"\n[red]{groq_error_message(e)}[/red]")

        self._shutdown()

    def _shutdown(self) -> None:
        self.timers.cancel_all()
        self.console.print("\n[cyan]Goodbye.[/cyan]")
        if self.speaker:
            self.speaker.close()


def _stop_on_sigterm(signum, frame) -> None:
    raise KeyboardInterrupt


_lock_file = None


LISTENING_LOCK = PROJECT_DIR / "logs" / ".listening.lock"


def _hold_single_listener_lock(path: Path = LISTENING_LOCK) -> None:
    """Only one JARVIS may listen at a time, or you'd get two answers to every question."""
    global _lock_file
    path.parent.mkdir(exist_ok=True)
    _lock_file = open(path, "w")
    try:
        fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(
            "JARVIS is already listening (probably the background service).\n"
            "  • Just say “Hey Jarvis”, or\n"
            "  • stop the background one first: uv run jarvis-service stop, or\n"
            "  • type to it alongside: uv run jarvis --ptt"
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="jarvis", description="Your personal JARVIS.")
    parser.add_argument("--ptt", action="store_true", help="push-to-talk / typing mode instead of hands-free")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _stop_on_sigterm)  # e.g. `jarvis-service stop`
    settings = load_settings()
    if not args.ptt:
        _hold_single_listener_lock()
    if settings.log_sessions:
        log.start()
    jarvis = Jarvis(settings, Console())
    jarvis.run_ptt() if args.ptt else jarvis.run_handsfree()


if __name__ == "__main__":
    main()
