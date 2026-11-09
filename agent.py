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
SHELL = "cmd.exe" if os.name == "nt" else "sh"
SYSTEM = (f"You are a coding agent. cwd: {os.getcwd()}. OS: {platform.system()}. "
          f"The bash tool runs each command in a fresh shell ({SHELL}), so cd and env vars do "
          f"not persist and commands must use {SHELL} syntax. Use it for all file and system work. "
          "Be concise.")


@tool
def bash(command: str) -> str:
    """Run a shell command and return its combined stdout and stderr."""
    p = subprocess.Popen(command, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
                         start_new_session=os.name != "nt")
    try:
        out, _ = p.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        if os.name == "nt":  # kill the whole tree, or communicate() waits for grandchildren
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        else:
            os.killpg(p.pid, 9)
        p.communicate()
        return f"error: command timed out after {TIMEOUT}s"
    if p.returncode:
        out += f"\n[exit {p.returncode}]"
    if len(out) > LIMIT:
        out = out[:LIMIT] + "\n[truncated]"
    return out or "(no output)"


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
        a = console.input("[yellow]run? \\[y/N/a] [/]").strip().lower()
        state["all"] = a == "a"
        return a in ("y", "a")

    return ask


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
    llm = ChatOpenAI(model=os.getenv("MODEL", "gpt-4.1"),
                     base_url=os.getenv("OPENAI_BASE_URL")).bind_tools([bash])
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
