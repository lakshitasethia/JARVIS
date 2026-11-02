"""JARVIS's abilities. Add a new tool by writing a `Tool` and listing it in build_registry()."""

from jarvis.tools import reminders, system
from jarvis.tools.registry import Tool, ToolRegistry
from jarvis.tools.timers import TimerManager
from jarvis.tools.weather import make_weather_tool


def build_registry(timers: TimerManager, home_location: str) -> ToolRegistry:
    tools: list[Tool] = [
        *system.TOOLS,
        *timers.tools(),
        make_weather_tool(home_location),
        *reminders.TOOLS,
    ]
    return ToolRegistry(tools)
