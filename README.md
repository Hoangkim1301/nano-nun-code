# nano-nun-code

A coding agent in a single Python file, built on [LangChain](https://python.langchain.com) with one tool: `bash`.

```
> create hello.py that prints the first 10 primes and run it
$ python -c "..." > hello.py
run? [y/N/a] y
...
```

`nano-nun-code` is a minimal take on projects like [nanocode](https://github.com/1rgs/nanocode). The whole agent is one readable file (~120 lines): a model, one shell tool, a loop, and a small terminal UI.

## Features

- **One file, one tool.** Reading, writing, editing and searching all go through the shell.
- **Explicit agent loop.** Plain Python on top of `ChatOpenAI.bind_tools`, with no hidden framework loop.
- **OpenAI-compatible.** Works with OpenAI or any endpoint that speaks the Chat Completions API with tool calling.
- **Confirm before running.** Every command asks `y/N/a` first. Pass `--yolo` to skip the prompt.
- **Terminal UI with [rich](https://github.com/Textualize/rich).** Markdown replies, a spinner while the model thinks, and dimmed command output.
- **Zero setup.** Dependencies are declared inline (PEP 723), so `uv run` handles everything.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) and an API key for an OpenAI-compatible endpoint.

```bash
git clone https://github.com/Hoangkim1301/nano-nun-code.git
cd nano-nun-code

cp .env.example .env        # then edit .env with your key
uv run --env-file .env agent.py
```

The first run installs dependencies into a cached environment; later runs start quickly.

## Configuration

Settings are read from environment variables. `.env` is git-ignored; `.env.example` is the template.

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | none | API key. Use any non-empty value if your server ignores it. |
| `OPENAI_BASE_URL` | no | OpenAI | Base URL of an OpenAI-compatible endpoint, e.g. `https://host/v1`. |
| `MODEL` | no | `gpt-4.1` | Model name sent to the endpoint. |

You can also set them without a `.env` file:

```powershell
# PowerShell (current session)
$env:OPENAI_API_KEY = "your-key"
uv run agent.py
```

```bash
# bash / zsh
OPENAI_API_KEY=your-key uv run agent.py
```

## Usage

Start the agent and type a request:

```
> find all TODO comments in this folder and summarize them
```

| Input | Effect |
|---|---|
| any text | Send a message to the agent |
| `/c` | Clear the conversation |
| `/q` or `/quit` | Quit (Ctrl-D also works on Linux and macOS) |
| Ctrl-C | Cancel the current turn; the session keeps running |

When the agent wants to run a command you see it prefixed with `$` and get a prompt:

| Answer | Meaning |
|---|---|
| `y` | Run this command |
| `n` or Enter | Deny it; the model is told and can adapt |
| `a` | Run this and every later command this session |

Run with `--yolo` to approve everything automatically:

```bash
uv run --env-file .env agent.py --yolo
```

> **Warning:** the agent executes shell commands as your user. There is no sandbox. Use `--yolo` only in a directory and environment where you accept that risk.

## How it works

The whole agent is this loop (`run_turn` in `agent.py`):

```python
while True:
    ai = llm.invoke(messages)                 # model answers or requests the tool
    messages.append(ai)
    if not ai.tool_calls:
        return                                # no tool call: the turn is done
    for call in ai.tool_calls:
        messages.append(bash.invoke(call))    # run it, feed the result back
```

The pieces:

- `bash`: a LangChain `@tool` that runs a command via the platform shell (`cmd.exe` on Windows, `sh` elsewhere) with a 120s timeout and output truncated to 10,000 characters.
- `run_turn`: the agent core. It never prints; it talks to the UI through callbacks (`ask`, `emit`, `busy`), so the `rich` shell can be swapped for another UI later.
- `main`: the `rich` input loop and rendering.

If a turn fails or is cancelled, the history rolls back to before the turn, so the next message is not sent with a dangling tool call.

## Development

```bash
uv run --no-project --with pytest --with langchain-openai --with rich python -m pytest tests -v
```

Tests use a fake model, so they need no API key or network.

## Known limitations

- No streaming: each reply appears once it is complete.
- On Windows each command runs in a fresh `cmd.exe`, so `cd` and environment changes do not persist between calls. The model is told this in its system prompt.
- No iteration cap on the agent loop and no request timeout on the model call.
- No persistence across sessions.
- No thinking or reasoning display. `ChatOpenAI` on the Chat Completions API does not expose reasoning text.

## Roadmap

- A fuller terminal UI with [Textual](https://textual.textualize.io) or [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io).
- Token streaming.

## Project docs

- Design: [`docs/superpowers/specs/2026-11-09-one-tool-coding-agent-design.md`](docs/superpowers/specs/2026-11-09-one-tool-coding-agent-design.md)
- Implementation plan: [`docs/superpowers/plans/2026-11-09-one-tool-coding-agent.md`](docs/superpowers/plans/2026-11-09-one-tool-coding-agent.md)

## License

[MIT](LICENSE)
