# One-File, One-Tool Coding Agent — Design

## Goal
A single-file Python coding agent, smaller than nanocode (target ~80-100 lines), built on LangChain with an OpenAI-compatible LLM. One tool (`bash`), run in an agent loop, with a `rich` terminal UI. Optimised for minimal size and readability, not polish.

## Decisions
| Topic | Decision |
|---|---|
| Loop | Explicit `while` loop over `ChatOpenAI.bind_tools` (not `create_agent`/LangGraph) |
| UI | `rich` now (Markdown replies, spinner); structured so `textual`/`prompt_toolkit` can replace the shell later |
| Safety | Confirm each command (`y/N/a`); `--yolo` skips the prompt |
| Shell | Platform shell via `subprocess.run(shell=True)`; system prompt states cwd and `platform.system()` |
| Config | Env vars: `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `MODEL` (default `gpt-4.1`) |
| Streaming | None initially; each reply rendered once under a spinner |
| History | In-memory message list; `/c` clears, `/q` or Ctrl-D quits |
| Deps | `langchain-openai`, `rich` via PEP 723 header (`uv run agent.py`) |

## Structure (`agent.py`)
```
# /// script (langchain-openai, rich)
imports, config (MODEL, SYSTEM)
bash tool                          -> run command, return truncated output
run_turn(messages, ask, on_event)  -> agent core, no printing
main()                             -> rich UI shell: input loop, rendering, /c /q, --yolo
```
`run_turn` is the only code that talks to the model. Callbacks: `ask(cmd) -> bool` (confirmation) and `on_event(kind, data)` (output). A later TUI replaces `main()` and the callbacks only.

## Data flow
1. User line -> `HumanMessage` appended.
2. `run_turn`: `llm.invoke(messages)` under a spinner; append `AIMessage`.
3. Emit `("text", ...)`; `main` renders as `rich.Markdown`.
4. No `tool_calls` -> turn ends.
5. For each tool call: emit `("tool", command)`, then `ask(cmd)` (always True under `--yolo`; `a` sets a session-wide allow flag).
   - Approved: `bash.invoke(call)`, emit result, append returned `ToolMessage`.
   - Denied: append `ToolMessage("User denied this command.")` so the model can adapt.
6. Loop to step 2.

## The `bash` tool and errors
- `@tool def bash(command: str) -> str`: `subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)`.
- Output: stdout + stderr merged, exit code appended when non-zero, truncated to ~10k chars with a `[truncated]` marker.
- Timeouts and exceptions are returned as error strings; they never crash the loop.
- System prompt: cwd, `platform.system()`, and an instruction to use the single shell tool for all file work.
- Ctrl-C during a turn cancels the turn and keeps the session; Ctrl-D or `/q` quits.
- Missing `OPENAI_API_KEY` exits immediately with a clear message.

## Testing
Separate small test file (not part of the one-file agent) using a fake chat model (`GenericFakeChatModel` or an `invoke` stub), covering: plain reply with no tool call; one tool call then a reply; a denied command. Plus a manual smoke test with a real key (`echo hi`, then a multi-step task).

## Out of scope
Persistence, planning, sub-agents, token streaming, multiple tools, command sandboxing, full-screen TUI (planned later via `textual`/`prompt_toolkit`).
