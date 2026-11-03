"""JARVIS's abilities. Add a new tool by writing a `Tool` and listing it in build_registry()."""

from jarvis.memory import MemoryStore
from jarvis.tools import reminders, system
from jarvis.tools.memory_tools import memory_tools
from jarvis.tools.phone import AndroidPhone
from jarvis.tools.registry import Tool, ToolRegistry
from jarvis.tools.timers import TimerManager
from jarvis.tools.weather import make_weather_tool


def build_registry(
    timers: TimerManager, home_location: str, memory: MemoryStore, phone: AndroidPhone | None = None
) -> ToolRegistry:
    tools: list[Tool] = [
        *system.TOOLS,
        *timers.tools(),
        make_weather_tool(home_location),
        *reminders.TOOLS,
        *memory_tools(memory),
        *(phone.tools() if phone else []),
    ]
    return ToolRegistry(tools)
