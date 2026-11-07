"""Speak text aloud, offline (sign-to-speech and exercise coach announcements).

Uses the voice that comes with the operating system, so nothing extra is installed:
Windows' System.Speech (through PowerShell), `say` on macOS, `espeak` on Linux if
present. Text is passed on stdin, never inside the command, and spoken by a
background thread one phrase at a time, so the video never waits for speech.
"""
import queue
import shutil
import subprocess
import sys
import threading

WINDOWS_SPEAK = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$s.Speak([Console]::In.ReadToEnd())"
)


def speech_command() -> list[str] | None:
    """The command that speaks its stdin on this system, or None if there isn't one."""
    if sys.platform == "win32" and shutil.which("powershell"):
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", WINDOWS_SPEAK]
    if sys.platform == "darwin" and shutil.which("say"):
        return ["say"]
    if shutil.which("espeak"):
        return ["espeak", "--stdin"]
    return None


class Speaker:
    def __init__(self, enabled: bool = True, command: list[str] | None = None):
        self.command = (command or speech_command()) if enabled else None
        self._queue: queue.Queue = queue.Queue()
        self._thread = None
        if self.command:
            self._thread = threading.Thread(target=self._speak_loop, name="speaker", daemon=True)
            self._thread.start()

    @property
    def available(self) -> bool:
        return self.command is not None

    def say(self, text: str) -> None:
        """Queue text to be spoken; returns immediately."""
        text = text.strip()
        if self.command and text:
            self._queue.put(text)

    def wait(self) -> None:
        """Block until everything queued has been spoken (used by tests)."""
        self._queue.join()

    def _speak_loop(self) -> None:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console window flashing up on Windows
        while True:
            text = self._queue.get()
            try:
                subprocess.run(self.command, input=text, text=True, timeout=30, creationflags=flags,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (OSError, subprocess.SubprocessError):
                pass  # speech is a nice extra; never let it break the app
            finally:
                self._queue.task_done()
