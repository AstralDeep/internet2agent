# internet2agent

Python interface and LLM agent for the **Internet2 Periscope Looking Glass MCP
server** (`https://periscope.ns.internet2.edu/mcp`) - the service announced in
[Internet2's MCP server post](https://internet2.edu/new-mcp-server-lets-re-community-connect-their-ai-agent/),
documented in the [Console docs](https://console.internet2.edu/docs/looking-glass.html#mcp-server).

Two ways in, no GUI:

1. **Direct client** (`PeriscopeClient` / `AsyncPeriscopeClient`) - typed Python
   access to the five Looking Glass tools. **Works today with zero
   credentials** (the server is currently open; verified live).
2. **LLM agent** (`Internet2Agent`) - natural-language questions answered by
   any **OpenAI-compatible** model (OpenAI, Ollama, vLLM, LM Studio,
   OpenRouter, ...). The agent pulls the tool schemas from the MCP server,
   hands them to the model as function tools, executes the model's tool calls
   against Periscope, and loops until it has an answer.

## Setup

```powershell
cd Y:\WORK\MCP\internet2agent
.venv\Scripts\activate
pip install -e ".[dev]"
```

Then configure your LLM in [.env](.env) (gitignored; template in
[.env.example](.env.example)):

| Variable | Needed for | Notes |
|---|---|---|
| `OPENAI_BASE_URL` | `ask` / `chat` / `Internet2Agent` | unset = api.openai.com; Ollama: `http://localhost:11434/v1`; LM Studio: `http://localhost:1234/v1` |
| `OPENAI_API_KEY` | same | optional for local endpoints that don't check keys |
| `I2A_MODEL` | same | model name to request (default `gpt-4o` - set to what your endpoint serves) |
| `PERISCOPE_MCP_URL` | optional | defaults to the public endpoint |
| `PERISCOPE_AUTH_TOKEN` | **not yet** | future Internet2 credential; sent as `Authorization: Bearer ...` once set |

## CLI

Direct Looking Glass (no credentials):

```powershell
internet2agent info                # service limits (rate limit, max targets)
internet2agent devices             # device inventory (name, location, platform)
internet2agent commands            # supported commands per platform
internet2agent filters             # output filters (include/exclude + regex)
internet2agent exec "show bgp" -t rtr1 rtr2 -p summary -f "include Established"
```

Add `--json` to any of the above for raw JSON. `i2a` is a short alias for
`internet2agent`.

LLM agent:

```powershell
internet2agent ask "Is BGP healthy on the Chicago routers?"
internet2agent chat                # interactive multi-turn session
```

If no LLM endpoint is configured yet, `ask`/`chat` walk you through a one-time
setup (base URL, API key, model) and offer to save it to `.env`.

Tool calls are echoed as `[lg_execute {...}]` lines while the agent works.

## Python API

```python
from internet2agent import PeriscopeClient

with PeriscopeClient() as lg:
    devices = lg.devices()                       # [{"name": ..., "platform": ...}, ...]
    result = lg.execute("show bgp", ["rtr1"], parameter="summary")
```

Async variant:

```python
from internet2agent import AsyncPeriscopeClient

async with AsyncPeriscopeClient() as lg:
    print(await lg.config())
```

Agent:

```python
from internet2agent import Internet2Agent

with Internet2Agent() as agent:                   # reads .env
    print(agent.ask("Which devices are in Seattle?"))
    print(agent.ask("Run a traceroute from one of them to 8.8.8.8"))  # follow-ups keep context
```

## Service constraints (from the server)

- Max **10 target devices** per `lg_execute`; commands must match documented
  syntax exactly (no abbreviations).
- Rate limit: **60 requests/min** (check live with `internet2agent info`).
- `parameter` is appended to the command (`show route` + `10.0.0.0/8`);
  `filter` is a filter name plus case-sensitive regex (`include bgp`,
  `exclude ^$`).
- Commands/filters are platform-specific - the agent (and you) should check
  `commands`/`filters` against each device's `platform` before executing.

## Tests

```powershell
pytest              # unit tests (offline, mocked)
pytest -m network   # live smoke tests against the real Periscope server
```

## Notes

- The agent requires an endpoint that supports OpenAI-style **function/tool
  calling**; pick a tool-capable model (most current ones are).
- The agent caps each question at 25 LLM round-trips as a runaway guard
  (`Internet2Agent(max_steps=...)` to change).
- Tool errors (unknown device, bad syntax, timeouts) are fed back to the model
  as `ERROR:` tool results so it can correct itself instead of crashing.
