"""Run JARVIS in the background, starting automatically when you log in.

  uv run jarvis-service install     build JARVIS.app, start it now and at every login
  uv run jarvis-service status      is it running?
  uv run jarvis-service logs [-f]   show (or follow) its output
  uv run jarvis-service restart     restart it (e.g. after changing .env or the code)
  uv run jarvis-service stop        stop it until the next `start` (or the next login)
  uv run jarvis-service start
  uv run jarvis-service uninstall   stop it and remove JARVIS.app and the login item
"""

import argparse
import hashlib
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

LABEL = "com.jarvis.assistant"
PROJECT = Path(__file__).resolve().parent.parent
PYTHON = PROJECT / ".venv" / "bin" / "python"
APP = Path.home() / "Applications" / "JARVIS.app"
EXECUTABLE = APP / "Contents" / "MacOS" / "JARVIS"
LAUNCH_AGENT = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
SERVICE_LOG = PROJECT / "logs" / "service.log"
LAUNCHER_SOURCE = Path(__file__).with_name("service_launcher.c")
DOMAIN = f"gui/{os.getuid()}"

INFO_PLIST = {
    "CFBundleIdentifier": LABEL,
    "CFBundleName": "JARVIS",
    "CFBundleDisplayName": "JARVIS",
    "CFBundleExecutable": "JARVIS",
    "CFBundlePackageType": "APPL",
    "CFBundleVersion": "1",
    "CFBundleShortVersionString": "1.0",
    "LSUIElement": True,  # no Dock icon, no menu bar
    "LSMinimumSystemVersion": "13.0",
    "NSMicrophoneUsageDescription": "JARVIS listens for “Hey Jarvis” and your spoken requests.",
    "NSAppleEventsUsageDescription": "JARVIS controls apps like Spotify and Reminders when you ask it to.",
}


def _run(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _build_id() -> str:
    """Changes only if the launcher or paths change. Rebuilding makes macOS ask for permissions again."""
    h = hashlib.sha256(LAUNCHER_SOURCE.read_bytes())
    h.update(f"{PROJECT}|{PYTHON}|{sorted(INFO_PLIST.items())}".encode())
    return h.hexdigest()[:16]


def build_app() -> bool:
    """Create ~/Applications/JARVIS.app. Returns True if it was (re)built."""
    marker = APP / "Contents" / "Resources" / "build-id"
    if marker.exists() and marker.read_text() == _build_id() and EXECUTABLE.exists():
        return False
    if APP.exists():
        shutil.rmtree(APP)
    EXECUTABLE.parent.mkdir(parents=True)
    marker.parent.mkdir(parents=True)
    with open(APP / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump(INFO_PLIST, f)
    _run(
        "clang", "-O2", "-Wall", "-o", str(EXECUTABLE), str(LAUNCHER_SOURCE),
        f'-DJARVIS_DIR="{PROJECT}"', f'-DJARVIS_PYTHON="{PYTHON}"',
    )
    marker.write_text(_build_id())
    # Ad-hoc signature: gives the app a stable identity for macOS permission records.
    _run("codesign", "--force", "--sign", "-", "--identifier", LABEL, str(APP))
    return True


def write_launch_agent() -> None:
    SERVICE_LOG.parent.mkdir(exist_ok=True)
    LAUNCH_AGENT.parent.mkdir(parents=True, exist_ok=True)
    agent = {
        "Label": LABEL,
        "ProgramArguments": [str(EXECUTABLE)],
        "WorkingDirectory": str(PROJECT),
        "RunAtLoad": True,  # start at login
        "KeepAlive": {"SuccessfulExit": False},  # restart after a crash, not after a clean stop
        "ThrottleInterval": 20,
        "ProcessType": "Interactive",  # low-latency audio; don't throttle like a batch job
        "StandardOutPath": str(SERVICE_LOG),
        "StandardErrorPath": str(SERVICE_LOG),
    }
    with open(LAUNCH_AGENT, "wb") as f:
        plistlib.dump(agent, f)


def is_loaded() -> bool:
    return _run("launchctl", "print", f"{DOMAIN}/{LABEL}", check=False).returncode == 0


def running_pid() -> int | None:
    out = _run("launchctl", "print", f"{DOMAIN}/{LABEL}", check=False).stdout
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("pid = "):
            return int(line.split("=")[1])
    return None


def start() -> None:
    if not LAUNCH_AGENT.exists():
        sys.exit("JARVIS isn't installed as a service yet. Run: uv run jarvis-service install")
    if not is_loaded():
        _run("launchctl", "bootstrap", DOMAIN, str(LAUNCH_AGENT))
    else:
        _run("launchctl", "kickstart", f"{DOMAIN}/{LABEL}")
    print("JARVIS is starting in the background. Say “Hey Jarvis” in a few seconds.")


def stop() -> None:
    if is_loaded():
        _run("launchctl", "bootout", f"{DOMAIN}/{LABEL}", check=False)
    print("JARVIS stopped. It will start again at your next login (or run: uv run jarvis-service start).")


def install() -> None:
    if not PYTHON.exists():
        sys.exit("No .venv found. Run `uv sync` in the project folder first.")
    rebuilt = build_app()
    write_launch_agent()
    if is_loaded():
        _run("launchctl", "bootout", f"{DOMAIN}/{LABEL}", check=False)
    _run("launchctl", "bootstrap", DOMAIN, str(LAUNCH_AGENT))
    print(f"✓ {'Built' if rebuilt else 'Using existing'} {APP}")
    print(f"✓ Starts at login ({LAUNCH_AGENT.name})")
    print("✓ Running now")
    if rebuilt:
        print(
            "\nmacOS will ask: “JARVIS would like to access the microphone.” Click Allow.\n"
            "Then run:  uv run jarvis-service restart   (so it starts listening with permission)"
        )


def uninstall() -> None:
    if is_loaded():
        _run("launchctl", "bootout", f"{DOMAIN}/{LABEL}", check=False)
    LAUNCH_AGENT.unlink(missing_ok=True)
    if APP.exists():
        shutil.rmtree(APP)
    print("JARVIS background service removed. (You can still run it in a terminal: uv run jarvis)")


def status() -> None:
    if not LAUNCH_AGENT.exists():
        print("Not installed. Run: uv run jarvis-service install")
        return
    pid = running_pid()
    if pid:
        print(f"● JARVIS is running in the background (pid {pid}). Say “Hey Jarvis”.")
    elif is_loaded():
        print("○ JARVIS is installed but not running right now (it may be restarting). See: uv run jarvis-service logs")
    else:
        print("○ JARVIS is stopped. Start it with: uv run jarvis-service start")


def logs(follow: bool) -> None:
    if not SERVICE_LOG.exists():
        print("No service log yet.")
        return
    if follow:
        os.execvp("tail", ["tail", "-n", "40", "-f", str(SERVICE_LOG)])
    print("".join(SERVICE_LOG.read_text(errors="replace").splitlines(keepends=True)[-40:]), end="")


def main() -> None:
    parser = argparse.ArgumentParser(prog="jarvis-service", description="Run JARVIS in the background.")
    parser.add_argument("command", choices=["install", "uninstall", "start", "stop", "restart", "status", "logs"])
    parser.add_argument("-f", "--follow", action="store_true", help="with `logs`: keep following")
    args = parser.parse_args()
    if sys.platform != "darwin":
        sys.exit("The background service is macOS-only.")
    if args.command == "restart":
        if is_loaded():
            _run("launchctl", "kickstart", "-k", f"{DOMAIN}/{LABEL}")
            print("JARVIS restarted. Say “Hey Jarvis” in a few seconds.")
        else:
            start()
    elif args.command == "logs":
        logs(args.follow)
    else:
        {"install": install, "uninstall": uninstall, "start": start, "stop": stop, "status": status}[args.command]()


if __name__ == "__main__":
    main()
