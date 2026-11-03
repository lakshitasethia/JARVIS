import pytest

from jarvis.brain.persona import system_prompt
from jarvis.memory import MAX_MEMORY_CHARS, MemoryStore
from jarvis.tools.memory_tools import memory_tools
from jarvis.tools.registry import ToolRegistry


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory.db")


def test_add_get_and_duplicates(store):
    m, created = store.add("Takes coffee black.")
    assert created and m.id == 1 and m.text == "Takes coffee black."
    again, created = store.add("  takes COFFEE black ")
    assert not created and again.id == 1
    assert len(store.all()) == 1


def test_update_delete_clear(store):
    store.add("Lives in Delhi.")
    store.add("Sister is called Riya.")
    assert store.update(1, "Lives in Mumbai.").text == "Lives in Mumbai."
    assert store.update(99, "nope") is None
    assert store.delete(2).text == "Sister is called Riya."
    assert store.delete(2) is None
    assert store.clear() == 1
    assert store.all() == []


def test_validation(store):
    with pytest.raises(ValueError):
        store.add("   ")
    with pytest.raises(ValueError):
        store.add("x" * (MAX_MEMORY_CHARS + 1))


def test_memories_survive_restart(tmp_path):
    MemoryStore(tmp_path / "m.db").add("Vegetarian.")
    assert [m.text for m in MemoryStore(tmp_path / "m.db").all()] == ["Vegetarian."]


def test_prompt_section_newest_first_and_budgeted(store):
    assert "don't know anything" in store.prompt_section()
    for i in range(30):
        store.add(f"Fact number {i} about the user, padded to take some room.")
    section = store.prompt_section(budget=300)
    lines = section.splitlines()
    assert lines[0].startswith("[30] Fact number 29")
    assert lines[-1].startswith("(+") and "older memories not shown" in lines[-1]
    assert len(section) < 400


def test_memory_tools_round_trip(store):
    tools = ToolRegistry(memory_tools(store))
    no = lambda q: False
    assert tools.run("remember", '{"fact": "Takes coffee black."}', no) == ("Saved as memory [1].", True)
    assert tools.run("remember", '{"fact": "takes coffee black"}', no) == ("Already known (memory [1]).", True)
    assert tools.run("update_memory", '{"memory_id": 1, "fact": "Takes coffee with oat milk."}', no)[1]
    assert store.get(1).text == "Takes coffee with oat milk."
    assert tools.run("forget", '{"memory_id": 1}', no) == ("Forgot: Takes coffee with oat milk.", True)
    assert tools.run("forget", '{"memory_id": 1}', no)[1] is False
    assert tools.run("remember", '{"fact": ""}', no)[1] is False


def test_forget_everything_needs_a_yes(store):
    store.add("A.")
    store.add("B.")
    tools = ToolRegistry(memory_tools(store))
    asked = []
    out, ok = tools.run("forget_everything", "{}", lambda q: asked.append(q) or False)
    assert not ok and len(store.all()) == 2 and "can't be undone" in asked[0]
    assert tools.run("forget_everything", "{}", lambda q: True) == ("Forgot all 2 memories.", True)


def test_system_prompt_includes_memories():
    p = system_prompt("sir", memories="[1] Takes coffee black.")
    assert "[1] Takes coffee black." in p
    assert "remember" in p
