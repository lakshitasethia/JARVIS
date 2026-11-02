"""Phase 0: talk to JARVIS by typing in the terminal."""

import time

import groq
from rich.console import Console

from jarvis.brain.conversation import Conversation
from jarvis.brain.llm import GroqLLM
from jarvis.brain.persona import system_prompt
from jarvis.config import load_settings

HELP = "[dim]Commands: /reset (forget this chat) · /help · /quit  ·  Ctrl+C stops a reply[/dim]"


def main() -> None:
    settings = load_settings()
    console = Console()
    llm = GroqLLM(settings.groq_api_key, settings.model, settings.reasoning_effort)
    convo = Conversation(system_prompt(settings.user_name), settings.max_history_messages)

    console.print(f"[bold cyan]JARVIS[/bold cyan] online · brain: {llm.model}")
    console.print(HELP)

    while True:
        try:
            text = console.input("\n[bold green]you ›[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[cyan]Goodbye.[/cyan]")
            return

        if not text:
            continue
        if text in ("/quit", "/exit"):
            console.print("[cyan]Goodbye.[/cyan]")
            return
        if text == "/help":
            console.print(HELP)
            continue
        if text == "/reset":
            convo.reset()
            console.print("[dim]Conversation cleared.[/dim]")
            continue

        convo.add_user(text)
        console.print("[bold cyan]jarvis ›[/bold cyan] ", end="")
        reply, start, first_token_at = [], time.perf_counter(), None
        try:
            for chunk in llm.stream(convo.messages()):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                reply.append(chunk)
                console.print(chunk, end="", markup=False, highlight=False)
        except KeyboardInterrupt:
            console.print(" [dim](interrupted)[/dim]", end="")
        except groq.AuthenticationError:
            console.print("\n[red]Groq rejected the API key. Check GROQ_API_KEY in .env.[/red]")
            convo.history.pop()
            continue
        except groq.RateLimitError:
            console.print("\n[yellow]Groq rate limit hit (free tier). Wait a moment and try again.[/yellow]")
            convo.history.pop()
            continue
        except groq.APIConnectionError:
            console.print("\n[red]Can't reach Groq. Check your internet connection.[/red]")
            convo.history.pop()
            continue
        except groq.APIStatusError as e:
            console.print(f"\n[red]Groq error {e.status_code}: {e.message}[/red]")
            convo.history.pop()
            continue

        full = "".join(reply).strip()
        if full:
            convo.add_assistant(full)
        else:
            convo.history.pop()  # nothing came back; don't leave a dangling user turn
        console.print()
        if settings.show_timing and first_token_at is not None:
            total = time.perf_counter() - start
            console.print(f"[dim]first word {first_token_at - start:.2f}s · total {total:.2f}s[/dim]")


if __name__ == "__main__":
    main()
