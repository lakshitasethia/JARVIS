from datetime import datetime

from jarvis.brain.conversation import Conversation
from jarvis.brain.persona import system_prompt


def test_messages_start_with_system_prompt():
    c = Conversation("SYS")
    c.add_user("hi")
    assert c.messages() == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
    ]


def test_history_is_trimmed_and_starts_on_user_turn():
    c = Conversation("SYS", max_history_messages=3)
    for i in range(3):
        c.add_user(f"q{i}")
        c.add_assistant(f"a{i}")
    msgs = c.messages()
    # last 3 = [a1, q2, a2] -> leading assistant dropped -> [q2, a2]
    assert [m["content"] for m in msgs] == ["SYS", "q2", "a2"]
    assert msgs[1]["role"] == "user"


def test_reset_clears_history():
    c = Conversation("SYS")
    c.add_user("hi")
    c.reset()
    assert c.messages() == [{"role": "system", "content": "SYS"}]


def test_persona_includes_name_and_date():
    p = system_prompt("Tony", now=datetime(2026, 11, 2))
    assert "Tony" in p
    assert "Monday, 02 November 2026" in p
