import threading
from datetime import datetime

from jarvis.brain.agent import Agent, CitationStripper, TextDelta, ToolFinished, ToolStarted, summarize_arguments
from jarvis.brain.conversation import Conversation
from jarvis.brain.llm import ToolCallRequest
from jarvis.listen.yesno import parse_yes_no
from jarvis.tools.registry import MAX_OUTPUT_CHARS, Tool, ToolRegistry, applescript_string, params
from jarvis.tools.reminders import create_reminder, reminder_script
from jarvis.tools.system import get_time
from jarvis.tools.timers import TimerManager, spoken_duration
from jarvis.tools.weather import describe, format_report

# --- registry -------------------------------------------------------------------------

def add(a: int, b: int) -> str:
    return str(a + b)


def make_registry(confirm=None):
    return ToolRegistry([
        Tool("add", "Add numbers", params(["a", "b"], a={"type": "integer"}, b={"type": "integer"}), add),
        Tool("danger", "Risky", params(["x"], x={"type": "string"}), lambda x: f"did {x}",
             confirm=lambda x: f"Really {x}?"),
        Tool("boom", "Fails", params(), lambda: 1 / 0),
        Tool("big", "Huge output", params(), lambda: "x" * 10_000),
    ])


def test_registry_runs_tools_and_reports_errors():
    r = make_registry()
    never = lambda q: False
    assert r.run("add", '{"a": 2, "b": 3}', never) == ("5", True)
    assert r.run("nope", "{}", never)[1] is False
    assert "invalid arguments" in r.run("add", "{not json", never)[0]
    assert "bad arguments" in r.run("add", '{"a": 1}', never)[0]
    out, ok = r.run("boom", "{}", never)
    assert not ok and "division" in out
    out, ok = r.run("big", "{}", never)
    assert ok and len(out) < MAX_OUTPUT_CHARS + 20 and out.endswith("(truncated)")


def test_sensitive_tool_needs_a_yes():
    r = make_registry()
    asked = []
    out, ok = r.run("danger", '{"x": "launch"}', lambda q: asked.append(q) or False)
    assert asked == ["Really launch?"] and not ok and "declined" in out
    assert r.run("danger", '{"x": "launch"}', lambda q: True) == ("did launch", True)


def test_schemas_are_openai_function_format():
    s = make_registry().schemas()[0]
    assert s["type"] == "function" and s["function"]["name"] == "add"
    assert s["function"]["parameters"]["required"] == ["a", "b"]


def test_applescript_string_escapes_quotes_and_backslashes():
    assert applescript_string('say "hi" \\ bye') == '"say \\"hi\\" \\\\ bye"'


# --- yes / no ------------------------------------------------------------------------------

def test_parse_yes_no():
    for t in ["Yes.", "yeah go ahead", "Sure, do it", "OK", "y"]:
        assert parse_yes_no(t) is True, t
    for t in ["No.", "nope", "don't", "Cancel that", "wait", "yes, no actually", "n o"]:
        assert parse_yes_no(t) in (False, None), t
    assert parse_yes_no("no") is False
    assert parse_yes_no("hmm, purple") is None


# --- timers ----------------------------------------------------------------------------------

def test_spoken_duration():
    assert spoken_duration(600) == "10 minutes"
    assert spoken_duration(3725) == "1 hour 2 minutes 5 seconds"
    assert spoken_duration(1) == "1 second"
    assert spoken_duration(0) == "0 seconds"


def test_timer_fires_and_announces():
    done = threading.Event()
    messages = []
    tm = TimerManager(lambda m: (messages.append(m), done.set()))
    assert "set for 1 second" in tm.set_timer(1, "tea")
    assert "'tea'" in tm.list_timers()
    assert done.wait(3)
    assert messages == ["Your tea timer is done."]
    assert tm.list_timers() == "No timers running."


def test_timer_cancel_rules():
    tm = TimerManager(lambda m: None)
    assert tm.set_timer(0).startswith("Error")
    tm.set_timer(600, "pasta")
    tm.set_timer(300, "eggs")
    assert "Which one" in tm.cancel_timer()
    assert "No timer matching" in tm.cancel_timer("rice")
    assert tm.cancel_timer("pas") == "Cancelled 'pasta'."
    assert tm.cancel_timer() == "Cancelled 'eggs'."
    assert tm.cancel_timer() == "No timers running."


# --- other tools ------------------------------------------------------------------------------

def test_get_time_timezones():
    assert "Error" in get_time("Mars/Olympus")
    assert get_time("Asia/Tokyo").endswith("JST")
    assert get_time()  # local time


def test_reminder_script_builds_date_from_parts():
    s = reminder_script('Call "Mum"', datetime(2027, 2, 28, 18, 30))
    assert "set day of d to 1" in s.splitlines()[1]
    assert "set month of d to 2" in s and "set day of d to 28" in s
    assert f"set time of d to {18 * 3600 + 30 * 60}" in s
    assert 'name:"Call \\"Mum\\"", remind me date:d' in s
    assert "remind me date" not in reminder_script("x", None)


def test_create_reminder_validates_without_touching_reminders_app():
    assert create_reminder("  ").startswith("Error")
    assert "couldn't understand" in create_reminder("x", "next tuesday-ish")
    assert "in the past" in create_reminder("x", "2001-01-01T09:00")


def test_weather_report_format():
    place = {"name": "London", "admin1": "England", "country": "United Kingdom"}
    data = {
        "current": {"temperature_2m": 9.4, "apparent_temperature": 7.1, "relative_humidity_2m": 81,
                    "weather_code": 61, "wind_speed_10m": 14.2},
        "daily": {"time": ["2026-11-03", "2026-11-04", "2026-11-05"], "weather_code": [61, 3, 0],
                  "temperature_2m_min": [6, 5, 4], "temperature_2m_max": [11, 10, 9],
                  "precipitation_probability_max": [80, 20, 5]},
    }
    text = format_report(place, data)
    assert text.startswith("London, England, United Kingdom now: light rain, 9°C (feels like 7°C)")
    assert "Tomorrow: overcast, 5 to 10°C, 20% chance of rain." in text
    assert "2026-11-05: clear sky" in text
    assert describe(None) == describe(12345) == "unknown conditions"


# --- agent loop -------------------------------------------------------------------------------

class ScriptedLLM:
    """Plays back canned model turns; records what it was sent."""

    model = "fake"

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def stream(self, messages, tools=None):
        self.seen.append([dict(m) for m in messages])
        yield from self.turns.pop(0)


def test_agent_runs_tool_then_answers_and_records_history():
    llm = ScriptedLLM([
        ["Let me add that. ", [ToolCallRequest("call_1", "add", '{"a": 2, "b": 2}')]],
        ["It's ", "4."],
    ])
    convo = Conversation("SYS")
    convo.add_user("what's 2+2")
    events = list(Agent(llm, make_registry()).run(convo, lambda q: False))

    assert [type(e) for e in events] == [TextDelta, ToolStarted, ToolFinished, TextDelta, TextDelta]
    assert events[2].output == "4" and events[2].ok
    roles = [m["role"] for m in convo.history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert convo.history[1]["tool_calls"][0]["function"]["name"] == "add"
    assert convo.history[2] == {"role": "tool", "tool_call_id": "call_1", "name": "add", "content": "4"}
    assert convo.history[3]["content"] == "It's 4."
    # second model call saw the tool result
    assert llm.seen[1][-1]["role"] == "tool"


def test_agent_gives_up_after_too_many_rounds():
    loop_forever = [[[ToolCallRequest(f"c{i}", "add", '{"a": 1, "b": 1}')]] for i in range(10)]
    convo = Conversation("SYS")
    convo.add_user("hi")
    events = list(Agent(ScriptedLLM(loop_forever), make_registry()).run(convo, lambda q: False))
    assert isinstance(events[-1], TextDelta) and "tangled" in events[-1].text
    assert convo.history[-1]["role"] == "assistant"


def test_citation_stripper_handles_split_markers():
    s = CitationStripper()
    out = s.feed("Verstappen won【1†L9") + s.feed("-L17】 the race.")
    assert out == "Verstappen won the race."


def test_summarize_arguments():
    assert summarize_arguments('{"name": "Spotify"}') == "name='Spotify'"
    assert summarize_arguments("{}") == ""
    assert summarize_arguments("{bad") == "{bad"
    assert len(summarize_arguments('{"q": "' + "x" * 200 + '"}')) == 80


def test_history_trim_never_orphans_tool_results():
    convo = Conversation("SYS", max_history_messages=3)
    convo.add_user("q1")
    convo.add_assistant("", [ToolCallRequest("c1", "add", "{}")])
    convo.add_tool_result("c1", "add", "2")
    convo.add_assistant("done")
    convo.add_user("q2")
    convo.add_assistant("a2")
    msgs = convo.messages()
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[1]["content"] == "q2"


# --- robustness ---------------------------------------------------------------------------

from jarvis.brain.agent import FUMBLE, RepeatGuard
from jarvis.brain.llm import ToolCallRejected


def guard_all(chunks):
    g = RepeatGuard()
    return "".join(g.feed(c) for c in chunks), g.tripped


def test_repeat_guard_cuts_glued_second_answer():
    # exactly how gpt-oss streamed it in the live test
    chunks = ["London", " is", " currently", " mainly", " clear", ".", "London", " is", " currently"]
    assert guard_all(chunks) == ("London is currently mainly clear.", True)
    assert guard_all(["Your volume is 75 percent.Your current"]) == ("Your volume is 75 percent.", True)


def test_repeat_guard_leaves_normal_text_alone():
    for text in ["It is 21.5 degrees. Nice.", "The U.S. Open starts today. Exciting!", "Done.", "Hi.There"]:
        assert guard_all([text]) == (text, False), text


class FlakyLLM(ScriptedLLM):
    """Like ScriptedLLM, but a turn given as an Exception raises it."""

    def stream(self, messages, tools=None):
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        yield from turn


def test_bogus_tool_call_is_retried_silently():
    llm = FlakyLLM([ToolCallRejected("called get_volume"), ["Volume is 75 percent."]])
    convo = Conversation("SYS")
    convo.add_user("volume?")
    events = list(Agent(llm, make_registry()).run(convo, lambda q: False))
    assert [e.text for e in events] == ["Volume is 75 percent."]
    assert convo.history[-1] == {"role": "assistant", "content": "Volume is 75 percent."}


def test_repeated_failures_apologise_instead_of_going_silent():
    for turns in ([ToolCallRejected("x"), ToolCallRejected("y")], [[], []]):
        convo = Conversation("SYS")
        convo.add_user("volume?")
        events = list(Agent(FlakyLLM(list(turns)), make_registry()).run(convo, lambda q: False))
        assert [e.text for e in events] == [FUMBLE]
        assert convo.history[-1]["content"] == FUMBLE  # never leaves the user message dangling


def test_empty_reply_is_retried_once():
    llm = FlakyLLM([[], ["Here you go."]])
    convo = Conversation("SYS")
    convo.add_user("hi")
    events = list(Agent(llm, make_registry()).run(convo, lambda q: False))
    assert [e.text for e in events] == ["Here you go."]


def test_error_text_from_a_tool_counts_as_failure():
    r = ToolRegistry([Tool("t", "d", params(), lambda: "Error: couldn't find that app.")])
    assert r.run("t", "{}", lambda q: False) == ("Error: couldn't find that app.", False)


# --- glitch safeguards (Phase 4) ------------------------------------------------------------

from jarvis.brain.agent import SCRAMBLED, JunkLineFilter, Scrambled
from jarvis.listen.stt import is_junk_transcript
from jarvis.speak.sentences import clean_for_speech


def junk_filter_all(chunks):
    f = JunkLineFilter()
    return "".join(f.feed(c) for c in chunks)


def test_junk_lines_are_hidden_and_normal_text_streams():
    assert junk_filter_all(["...\n", "It's ", "sunny."]) == "It's sunny."
    assert junk_filter_all(["Hello", " there.\n\nBye."]) == "Hello there.\nBye."
    assert junk_filter_all(["... well, ", "maybe."]) == "... well, maybe."  # punctuation then words is fine


def test_degenerate_output_is_detected():
    # what the user saw: dots on their own lines with blank lines between
    import pytest

    with pytest.raises(Scrambled):
        junk_filter_all(["...\n\n", "...\n\n", "I'm sorry\n\n", "...\n"])
    with pytest.raises(Scrambled):
        junk_filter_all(["Hi\n\n\n\nthere"])


def test_scrambled_reply_is_retried_then_apologised_for():
    convo = Conversation("SYS")
    convo.add_user("hi")
    events = list(Agent(FlakyLLM([["...\n\n...\n\n"], ["Hello, sir."]]), make_registry()).run(convo, lambda q: False))
    assert [e.text for e in events] == ["Hello, sir."]

    convo = Conversation("SYS")
    convo.add_user("hi")
    events = list(Agent(FlakyLLM([["...\n\n...\n"], ["…\n\n…\n"]]), make_registry()).run(convo, lambda q: False))
    assert [e.text for e in events] == [SCRAMBLED]
    assert convo.history[-1]["content"] == SCRAMBLED


def test_junk_transcripts():
    for t in ["...", ".", " ♪ ", "", "Thanks for watching!", "you"]:
        assert is_junk_transcript(t), t
    for t in ["Yes.", "No", "What time is it?", "Thank you, Jarvis."]:
        assert not is_junk_transcript(t), t


def test_punctuation_only_is_never_spoken():
    assert clean_for_speech("...") == ""
    assert clean_for_speech("…… —") == ""
    assert clean_for_speech("... Right.") == "... Right."
