"""Apple Reminders (syncs to iPhone via iCloud)."""

from datetime import datetime

from jarvis.tools.registry import Tool, applescript_string, osascript, params


def reminder_script(title: str, due: datetime | None) -> str:
    lines = []
    props = f"name:{applescript_string(title)}"
    if due is not None:
        # Build the date from parts: AppleScript date strings are locale-dependent.
        # Set day to 1 first so changing the month can't overflow (e.g. 31 → February).
        lines += [
            "set d to current date",
            "set day of d to 1",
            f"set year of d to {due.year}",
            f"set month of d to {due.month}",
            f"set day of d to {due.day}",
            f"set time of d to {due.hour * 3600 + due.minute * 60}",
        ]
        props += ", remind me date:d"
    lines.append(f'tell application "Reminders" to make new reminder with properties {{{props}}}')
    return "\n".join(lines)


def create_reminder(title: str, due: str = "") -> str:
    title = title.strip()
    if not title:
        return "Error: the reminder needs a title."
    when = None
    if due.strip():
        try:
            when = datetime.fromisoformat(due.strip())
        except ValueError:
            return f"Error: couldn't understand due time '{due}'. Use ISO format like 2026-11-03T18:30."
        if when < datetime.now(when.tzinfo):
            return f"Error: {when:%d %B %I:%M %p} is in the past."
    osascript(reminder_script(title, when))
    return f"Reminder '{title}' created" + (f" for {when:%A %d %B at %I:%M %p}." if when else ".")


TOOLS = [
    Tool(
        "create_reminder",
        "Add a reminder to Apple Reminders, optionally with an alert time.",
        params(
            ["title"],
            title={"type": "string"},
            due={"type": "string", "description": "Local ISO datetime, e.g. 2026-11-03T18:30. Omit for no alert."},
        ),
        create_reminder,
    ),
]
