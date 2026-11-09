import sys
import time
from contextlib import nullcontext

import pytest

import agent

PY = f'"{sys.executable}"'


def run(cmd):
    return agent.bash.invoke({"command": cmd})


def test_bash_returns_output():
    assert "hi" in run("echo hi")


def test_bash_no_output():
    assert run(f'{PY} -c "pass"') == "(no output)"


def test_bash_exit_code():
    assert "[exit 3]" in run("exit 3")


def test_bash_truncates():
    out = run(f"{PY} -c \"print('x'*20000)\"")
    assert out.endswith("[truncated]")
    assert len(out) < agent.LIMIT + 50


def test_bash_timeout_returns_promptly(monkeypatch):
    monkeypatch.setattr(agent, "TIMEOUT", 1)
    t = time.time()
    out = run(f'{PY} -c "import time; time.sleep(10)"')
    assert "timed out" in out and time.time() - t < 4


def test_bash_survives_undecodable_bytes():
    out = run(f"{PY} -c \"import sys; sys.stdout.buffer.write(b'\x81ok')\"")
    assert "ok" in out


def test_system_prompt_explains_shell():
    assert "fresh shell" in agent.SYSTEM
    assert sys.platform != "win32" or "cmd.exe" in agent.SYSTEM


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


class Fake:
    def __init__(self, *replies):
        self.replies = list(replies)

    def invoke(self, messages):
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def call(cmd=None, id="1"):
    args = {} if cmd is None else {"command": cmd}
    return {"name": "bash", "args": args, "id": id}


def turn(llm, ask=lambda c: True, line="go"):
    messages, events = [SystemMessage("sys")], []
    agent.run_turn(llm, messages, line, ask, lambda k, d: events.append((k, d)), nullcontext)
    return messages, events


def test_plain_reply():
    messages, events = turn(Fake(AIMessage("hello")))
    assert events == [("text", "hello")]
    assert [type(m) for m in messages] == [SystemMessage, HumanMessage, AIMessage]


def test_tool_call_then_reply():
    llm = Fake(AIMessage("", tool_calls=[call("echo hi")]), AIMessage("done"))
    messages, events = turn(llm)
    assert events[0] == ("tool", "echo hi")
    assert events[1][0] == "result" and "hi" in events[1][1]
    assert events[-1] == ("text", "done")
    assert isinstance(messages[3], ToolMessage) and messages[3].tool_call_id == "1"


def test_denied_command_not_run():
    llm = Fake(AIMessage("", tool_calls=[call("echo hi")]), AIMessage("ok"))
    messages, _ = turn(llm, ask=lambda c: False)
    assert messages[3].content == "User denied this command."


def test_missing_command_arg_is_error_result():
    llm = Fake(AIMessage("", tool_calls=[call(None)]), AIMessage("ok"))
    messages, events = turn(llm)
    assert messages[3].content.startswith("error:")
    assert events[-1] == ("text", "ok")


def test_invalid_tool_calls_get_error_message():
    bad = AIMessage("", invalid_tool_calls=[
        {"name": "bash", "args": "{bad", "id": "x", "error": "boom", "type": "invalid_tool_call"}])
    messages, _ = turn(Fake(bad, AIMessage("ok")))
    assert messages[3].tool_call_id == "x" and messages[3].content.startswith("error:")


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), RuntimeError("api down")])
def test_failed_turn_rolls_back_history(exc):
    llm = Fake(AIMessage("", tool_calls=[call("echo hi")]), exc)
    messages = [SystemMessage("sys")]
    with pytest.raises(type(exc)):
        agent.run_turn(llm, messages, "go", lambda c: True, lambda k, d: None, nullcontext)
    assert len(messages) == 1


class FakeConsole:
    def __init__(self, *answers):
        self.answers = list(answers)

    def input(self, prompt):
        return self.answers.pop(0)


def test_ask_yes_no_always():
    ask = agent.make_ask(FakeConsole("y", "n", "a"), yolo=False)
    assert [ask("c"), ask("c"), ask("c"), ask("c")] == [True, False, True, True]


def test_ask_yolo_never_prompts():
    assert agent.make_ask(FakeConsole(), yolo=True)("c") is True
import io

from rich.console import Console


def test_emit_prints_markup_literally():
    buf = io.StringIO()
    emit = agent.make_emit(Console(file=buf, width=80, color_system=None))
    emit("tool", "echo [/oops]")
    emit("result", "[/bold] [red]x[/red]")
    out = buf.getvalue()
    assert "echo [/oops]" in out and "[/bold] [red]x[/red]" in out


def test_emit_renders_text_as_markdown():
    buf = io.StringIO()
    agent.make_emit(Console(file=buf, width=80, color_system=None))("text", "**hi**")
    assert "hi" in buf.getvalue() and "**" not in buf.getvalue()


def test_ask_prompt_shows_choices(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    agent.make_ask(Console(file=buf, color_system=None), yolo=False)("c")
    assert "[y/N/a]" in buf.getvalue()
