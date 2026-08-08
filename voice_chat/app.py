from __future__ import annotations

import queue
import sys
import threading
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from voice_chat.config import Config
from voice_chat.llm import LLMClient
from voice_chat.stt import SpeechToText
from voice_chat.tts import TextToSpeech

console = Console()


class VoiceChatApp:
    """Listen → transcribe → LLM → speak, continuously."""

    def __init__(self, config: Config):
        self.config = config
        self._turns: "queue.Queue[Optional[str]]" = queue.Queue()
        self._busy = threading.Event()
        self._stop = threading.Event()

        self.llm = LLMClient(config)
        self.tts = TextToSpeech(config)
        self.stt = SpeechToText(
            config,
            on_partial=self._on_partial,
            on_final=self._on_final,
        )

    def _on_partial(self, text: str) -> None:
        if self._busy.is_set():
            return
        console.print(f"[dim]… {text}[/dim]", end="\r")

    def _on_final(self, text: str) -> None:
        text = text.strip()
        if not text or self._busy.is_set() or self._stop.is_set():
            return
        # Mute immediately so TTS / ambient noise don't enqueue more turns
        self.stt.mute(True)
        self._turns.put(text)

    def load(self) -> None:
        console.print("[bold]Loading Moonshine STT (CPU)…[/bold]")
        self.stt.load()
        console.print("[bold]Loading CosyVoice2-0.5B TTS (GPU FP16)…[/bold]")
        console.print(
            "[dim]First run downloads ~2GB of weights; later starts are faster.[/dim]"
        )
        self.tts.load()
        console.print("[green]Ready.[/green]")

    def run(self) -> None:
        self.load()
        console.print(
            Panel.fit(
                "[bold]Voice chat[/bold]\n"
                f"STT: Moonshine [cyan]{self.config.moonshine_model_arch}[/cyan]\n"
                f"TTS: CosyVoice2-0.5B\n"
                f"LLM: [cyan]{self.config.llm_model}[/cyan] @ {self.config.llm_base_url}\n\n"
                "Speak naturally. Pause when finished — your question is sent automatically.\n"
                "Press [bold]Ctrl+C[/bold] to quit.",
                title="mic → stt → llm → tts",
                border_style="cyan",
            )
        )

        self.stt.start()
        console.print("[bold green]Listening…[/bold green]")

        try:
            while not self._stop.is_set():
                try:
                    user_text = self._turns.get(timeout=0.2)
                except queue.Empty:
                    continue
                if user_text is None:
                    break
                self._handle_turn(user_text)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping…[/yellow]")
        finally:
            self.shutdown()

    def _handle_turn(self, user_text: str) -> None:
        self._busy.set()
        try:
            console.print(f"\n[bold cyan]You:[/bold cyan] {user_text}")
            console.print("[dim]Thinking…[/dim]")
            try:
                reply = self.llm.chat(user_text)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]LLM error:[/red] {exc}")
                reply = "Sorry, I could not reach the language model."

            console.print(f"[bold magenta]Assistant:[/bold magenta] {reply}")
            console.print("[dim]Speaking…[/dim]")
            try:
                self.tts.speak(reply)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]TTS error:[/red] {exc}")
        finally:
            # Drain any speech that leaked while we were busy
            while True:
                try:
                    self._turns.get_nowait()
                except queue.Empty:
                    break
            self._busy.clear()
            self.stt.mute(False)
            console.print("[bold green]Listening…[/bold green]")

    def shutdown(self) -> None:
        self._stop.set()
        self._turns.put(None)
        self.stt.close()
        self.tts.unload()


def main(argv: Optional[list[str]] = None) -> int:
    _ = argv  # reserved for future CLI flags
    try:
        config = Config.load()
    except SystemExit as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    app = VoiceChatApp(config)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
