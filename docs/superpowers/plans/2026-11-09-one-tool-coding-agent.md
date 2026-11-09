# One-File, One-Tool Coding Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `agent.py`, a single-file LangChain coding agent with one `bash` tool, an explicit agent loop, and a `rich` terminal UI.

**Architecture:** `bash` is a LangChain `@tool`. `run_turn` is a UI-free agent core: it calls `ChatOpenAI.bind_tools` in a `while` loop and talks to the UI only through callbacks (`ask`, `emit`, `busy`). `main()` is the `rich` shell, the only part a later `textual`/`prompt_toolkit` UI would replace.

**Tech Stack:** Python 3.10+, `langchain-openai`, `langchain-core`, `rich`, `pytest` (tests only), `uv` for running.

**Spec:** `docs/superpowers/specs/2026-11-09-one-tool-coding-agent-design.md`

## Global Constraints

- Everything lives in one file, `agent.py` (~80-100 lines target, hard ceiling 120 lines).
- Runtime deps are only `langchain-openai` and `rich`, declared in a PEP 723 header so `uv run agent.py` works.
- Exactly one tool, named `bash`, run via `subprocess.run(shell=True)`; timeout 120s; output truncated to ~10k chars with a `[truncated]` marker.
- Config via env vars only: `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `MODEL` (default `gpt-4.1`).
- Loop is an explicit `while` over `ChatOpenAI.bind_tools` (no `create_agent`/LangGraph).
- Confirm each command with `y/N/a`; `--yolo` skips the prompt.
- No streaming, persistence, planning, sub-agents or sandboxing.
- `run_turn` must not print; all output goes through callbacks.
- Tests live in a separate file, `tests/test_agent.py`.
- Run tests with: `uv run --no-project --with pytest --with langchain-openai --with rich python -m pytest tests -v` from the project directory (`python -m pytest` puts the project root on `sys.path`).
- Commit only this project's files, using explicit paths (the parent repo has a lot of unrelated untracked content): `git add agent.py tests/test_agent.py`.

## Review Focus

Spec-implied failure modes, most likely first. Each has a pinned test in the task that owns the code.

1. Ctrl-C or an API error mid-turn leaves a dangling `tool_call` in history, so the next request fails. Expected: the turn rolls back and the session continues. (Task 2)
2. The model emits a tool call with no `command` argument. Expected: an error result goes back to the model and the loop continues. (Task 2)
3. The model emits malformed tool-call JSON (`invalid_tool_calls`). Expected: each one gets an error `ToolMessage`. (Task 2)
4. A command prints nothing, a huge amount, or hangs. Expected: `(no output)`, truncation, and a timeout error string. (Task 1)
5. Command output containing rich markup such as `[/oops]` must not crash rendering. Expected: printed literally. (Task 3)

---

### Task 1: The `bash` tool

**Files:**
- Create: `agent.py`
- Create: `tests/test_agent.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `agent.bash` (LangChain tool, `bash.invoke({"command": str}) -> str`), module constants `TIMEOUT = 120` and `LIMIT = 10000`, and the full import block used by later tasks.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent.py`:

```python
import sys
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


def test_bash_timeout(monkeypatch):
    monkeypatch.setattr(agent, "TIMEOUT", 1)
    assert "timed out" in run(f'{PY} -c "import time; time.sleep(3)"')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-project --with pytest --with langchain-openai --with rich python -m pytest tests -v`
Expected: collection error, `ModuleNotFoundError: No module named 'agent'`.

- [ ] **Step 3: Write minimal implementation**

Create `agent.py`:

```python
# /// script
# dependencies = ["langchain-openai", "rich"]
# ///
"""One-file, one-tool coding agent.  uv run agent.py [--yolo]"""
import os, platform, subprocess, sys

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markdown import Markdown

TIMEOUT, LIMIT = 120, 10000
SYSTEM = (f"You are a coding agent. cwd: {os.getcwd()}. OS: {platform.system()}. "
          "Use the bash tool for all file and system work. Be concise.")


@tool
def bash(command: str) -> str:
    """Run a shell command and return its combined stdout and stderr."""
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {TIMEOUT}s"
    out = r.stdout + r.stderr
    if r.returncode:
        out += f"\n[exit {r.returncode}]"
    if len(out) > LIMIT:
        out = out[:LIMIT] + "\n[truncated]"
    return out or "(no output)"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-project --with pytest --with langchain-openai --with rich python -m pytest tests -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: add bash tool"
```

---

### Task 2: Agent core (`run_turn`) and confirmation (`make_ask`)

**Files:**
- Modify: `agent.py` (append after `bash`)
- Modify: `tests/test_agent.py` (append)

**Interfaces:**
- Consumes: `agent.bash`, the imports and constants from Task 1.
- Produces:
  - `run_turn(llm, messages: list, line: str, ask, emit, busy) -> None`
    - `llm` has `.invoke(messages) -> AIMessage`.
    - `ask(cmd: str) -> bool`.
    - `emit(kind: str, data: str)`, where `kind` is `"text"`, `"tool"` or `"result"`.
    - `busy` is a zero-argument callable returning a context manager.
    - It appends `HumanMessage(line)` itself. On any exception it truncates `messages` back to its length before the call and re-raises.
  - `make_ask(console, yolo: bool) -> ask`, where `console` needs only `.input(prompt) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent.py`:

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


class Fake:
    def __init__(self, *replies):
        self.replies = list(replies)

    def invoke(self, messages):
        r = self.replies.pop(0)
        if isinstance(r, Exception):
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-project --with pytest --with langchain-openai --with rich python -m pytest tests -v`
Expected: the new tests FAIL with `AttributeError: module 'agent' has no attribute 'run_turn'` (and `make_ask`); Task 1 tests still pass.

- [ ] **Step 3: Write minimal implementation**

Append to `agent.py`:

```python


def run_turn(llm, messages, line, ask, emit, busy):
    """One user turn: loop model <-> bash until the model stops calling tools."""
    n = len(messages)
    messages.append(HumanMessage(line))
    try:
        while True:
            with busy():
                ai = llm.invoke(messages)
            messages.append(ai)
            if ai.content:
                emit("text", ai.content)
            if not (ai.tool_calls or ai.invalid_tool_calls):
                return
            for c in ai.tool_calls:
                cmd = c["args"].get("command", "")
                emit("tool", cmd)
                try:
                    msg = bash.invoke(c) if ask(cmd) else ToolMessage(
                        "User denied this command.", tool_call_id=c["id"])
                except Exception as e:
                    msg = ToolMessage(f"error: {e}", tool_call_id=c["id"])
                emit("result", msg.content)
                messages.append(msg)
            for c in ai.invalid_tool_calls:
                messages.append(ToolMessage(f"error: {c['error']}", tool_call_id=c["id"]))
    except BaseException:
        del messages[n:]
        raise


def make_ask(console, yolo):
    state = {"all": yolo}

    def ask(cmd):
        if state["all"]:
            return True
        a = console.input("[yellow]run? [y/N/a] [/]").strip().lower()
        state["all"] = a == "a"
        return a in ("y", "a")

    return ask
```

Note: if a bad tool call raises inside `ask` (for example `EOFError` at the prompt), the `except Exception` turns it into an error result. `KeyboardInterrupt` is not an `Exception`, so it still propagates and triggers the rollback.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-project --with pytest --with langchain-openai --with rich python -m pytest tests -v`
Expected: all tests pass (5 from Task 1 plus 10 new).

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: add agent loop and command confirmation"
```

---

### Task 3: `rich` UI shell (`make_emit`, `main`)

**Files:**
- Modify: `agent.py` (append)
- Modify: `tests/test_agent.py` (append)

**Interfaces:**
- Consumes: `run_turn`, `make_ask`, `SYSTEM`, `bash` from earlier tasks.
- Produces:
  - `make_emit(console) -> emit(kind, data)`, where `console` is a `rich.console.Console`.
  - `main() -> None`, the CLI entry point. It reads `--yolo` from `sys.argv`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_agent.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-project --with pytest --with langchain-openai --with rich python -m pytest tests -v`
Expected: the two new tests FAIL with `AttributeError: module 'agent' has no attribute 'make_emit'`.

- [ ] **Step 3: Write minimal implementation**

Append to `agent.py`:

```python


def make_emit(console):
    def emit(kind, data):
        if kind == "text":
            console.print(Markdown(data))
        elif kind == "tool":
            console.print(f"$ {data}", style="cyan", markup=False)
        else:
            console.print(data, style="dim", markup=False)

    return emit


def main():
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set")
    console = Console()
    llm = ChatOpenAI(model=os.getenv("MODEL", "gpt-4.1")).bind_tools([bash])
    ask, emit = make_ask(console, "--yolo" in sys.argv), make_emit(console)
    messages = [SystemMessage(SYSTEM)]
    while True:
        try:
            line = console.input("[bold green]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in ("/q", "/quit"):
            break
        if line == "/c":
            messages[:] = messages[:1]
            console.print("cleared", style="dim")
        elif line:
            try:
                run_turn(llm, messages, line, ask, emit, lambda: console.status("thinking..."))
            except (Exception, KeyboardInterrupt) as e:
                console.print(f"cancelled: {e!r}", style="red", markup=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass, and check size**

Run: `uv run --no-project --with pytest --with langchain-openai --with rich python -m pytest tests -v`
Expected: all tests pass (17 total).

Run: `wc -l agent.py`
Expected: at most 120 lines (target ~100). If it is over 120, trim before committing.

- [ ] **Step 5: Manual smoke test with a real key**

Run (PowerShell): `$env:OPENAI_API_KEY="sk-..."; uv run agent.py`
Check each of these:
1. Type `say hi`. A markdown reply appears under a "thinking..." spinner.
2. Type `list the files here`. A `$ dir` or `$ ls` line appears and the `run? [y/N/a]` prompt shows. Answer `y` and a dim result appears, then a summary.
3. Repeat and answer `n`. The model acknowledges the denial.
4. Type `/c`, then `/q`. The agent clears and exits.
5. Run `uv run agent.py --yolo` and give a multi-step task such as "create hello.txt containing hi, then print it". Commands run with no prompts. Delete `hello.txt` afterwards.
6. Press Ctrl-C during a turn. The agent prints `cancelled`, stays alive, and the next message works.

Expected: all six behave as described. Report any that don't instead of editing around them.

- [ ] **Step 6: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: add rich UI shell and main loop"
```

---

## Self-Review

- **Spec coverage:**
  - Explicit loop, tool, truncation and timeout, confirm with `--yolo`, env config: Tasks 1-3.
  - Platform shell with the OS in the system prompt: `SYSTEM` in Task 1.
  - `/c`, `/q`, Ctrl-D, Ctrl-C, and the missing-key exit: `main` in Task 3.
  - PEP 723 header: Task 1.
  - Testing with a fake model: Task 2.
  - Deviation from the spec: `run_turn` takes the user `line` and a `busy` context-manager callback. The spec listed only `ask` and `on_event`, and `line` is there so rollback also removes the `HumanMessage`. `busy` keeps the spinner out of the UI-free core. `on_event` is named `emit`.
- **Placeholder scan:** none. Every step has complete code or an exact command.
- **Type consistency:** `run_turn(llm, messages, line, ask, emit, busy)` is the same in the Task 2 tests and implementation and in `main`. `make_ask(console, yolo)`, `make_emit(console)`, `bash`, `TIMEOUT` and `LIMIT` are consistent across tasks.
- **Review Focus:** all five items have a pinned test (items 1-3 and 5 in Tasks 2-3, item 4 in Task 1).
