"""Mac control: time, apps, volume, music."""

import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jarvis.tools.registry import Tool, applescript_string, osascript, params


def get_time(timezone: str | None = None) -> str:
    if timezone:
        try:
            now = datetime.now(ZoneInfo(timezone))
        except (ZoneInfoNotFoundError, ValueError):
            return f"Error: unknown timezone '{timezone}'. Use an IANA name like 'Asia/Tokyo'."
    else:
        now = datetime.now().astimezone()
    return now.strftime("%A %d %B %Y, %I:%M %p %Z")


def open_app(name: str) -> str:
    result = subprocess.run(["open", "-a", name], capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        return f"Error: couldn't find an app called '{name}' on this Mac."
    return f"Opened {name}."


def _is_running(name: str) -> bool:
    return osascript(f"application {applescript_string(name)} is running") == "true"


def quit_app(name: str) -> str:
    if not _is_running(name):
        return f"{name} isn't running."
    osascript(f"tell application {applescript_string(name)} to quit")
    return f"Quit {name}."


def set_volume(level: int | None = None, change: int | None = None) -> str:
    current = int(osascript("output volume of (get volume settings)") or 0)
    if level is None and change is None:
        return f"Volume is {current}%."
    target = level if level is not None else current + (change or 0)
    target = max(0, min(100, int(target)))
    osascript(f"set volume output volume {target}")
    return f"Volume set to {target}%."


_MEDIA_COMMANDS = {
    "play": "play",
    "pause": "pause",
    "toggle": "playpause",
    "next": "next track",
    "previous": "previous track",
}


def media_control(action: str, app: str = "Spotify") -> str:
    if action not in _MEDIA_COMMANDS:
        return f"Error: action must be one of {', '.join(_MEDIA_COMMANDS)}."
    if app not in ("Spotify", "Music"):
        return "Error: app must be 'Spotify' or 'Music'."
    target = applescript_string(app)
    osascript(f"tell application {target} to {_MEDIA_COMMANDS[action]}")
    if action in ("pause",):
        return f"Paused {app}."
    try:
        track = osascript(
            f"tell application {target} to (name of current track) & \" by \" & (artist of current track)"
        )
        return f"{app}: {action} done. Now playing: {track}."
    except RuntimeError:
        return f"{app}: {action} done."


TOOLS = [
    Tool(
        "get_time",
        "Current date and time. Omit timezone for the user's local time.",
        params(timezone={"type": "string", "description": "IANA timezone, e.g. 'Asia/Tokyo'"}),
        get_time,
    ),
    Tool(
        "open_app",
        "Open (launch or bring to front) a Mac application by name.",
        params(["name"], name={"type": "string", "description": "App name, e.g. 'Spotify', 'Safari'"}),
        open_app,
    ),
    Tool(
        "quit_app",
        "Quit a running Mac application.",
        params(["name"], name={"type": "string", "description": "App name"}),
        quit_app,
        confirm=lambda name: f"Quit {name}? Any unsaved work there could be lost.",
    ),
    Tool(
        "set_volume",
        "Get or change the Mac's output volume. No arguments = report current volume.",
        params(
            level={"type": "integer", "description": "Absolute volume 0-100"},
            change={"type": "integer", "description": "Relative change, e.g. 10 or -20"},
        ),
        set_volume,
    ),
    Tool(
        "media_control",
        "Control music playback in Spotify or Apple Music.",
        params(
            ["action"],
            action={"type": "string", "enum": list(_MEDIA_COMMANDS)},
            app={"type": "string", "enum": ["Spotify", "Music"], "description": "Default Spotify"},
        ),
        media_control,
    ),
]
