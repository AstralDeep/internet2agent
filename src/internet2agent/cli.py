"""Command-line interface for internet2agent.

Direct Looking Glass access (no credentials needed):

    internet2agent info
    internet2agent devices
    internet2agent commands
    internet2agent filters
    internet2agent exec "show bgp" -t rtr1 rtr2 -p summary -f "include Established"

Natural-language agent (configure your OpenAI-compatible endpoint in .env):

    internet2agent ask "Is BGP healthy between Chicago and Seattle?"
    internet2agent chat
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from typing import Any

import openai

from .agent import Internet2Agent
from .config import DEFAULT_MODEL, Settings, save_env_values
from .periscope import PeriscopeClient, PeriscopeError

_DEVICE_COLUMNS = ["name", "location", "type", "platform"]
_COMMAND_COLUMNS = ["name", "platforms", "help"]


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2))


def _print_table(rows: list[dict[str, Any]], preferred: list[str]) -> None:
    if not rows:
        print("(no results)")
        return
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            seen[key] = None
    columns = [c for c in preferred if c in seen] + [c for c in seen if c not in preferred]

    def cell(row: dict[str, Any], col: str) -> str:
        value = row.get(col, "")
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        return "" if value is None else str(value)

    widths = {c: max(len(c), *(len(cell(r, c)) for r in rows)) for c in columns}
    print("  ".join(c.upper().ljust(widths[c]) for c in columns))
    print("  ".join("-" * widths[c] for c in columns))
    for row in rows:
        print("  ".join(cell(row, c).ljust(widths[c]) for c in columns))


def _rows(data: Any, *keys: str) -> list[dict[str, Any]] | None:
    """Coerce a tool result into a list of dicts, unwrapping one level if needed."""
    if isinstance(data, dict):
        for key in keys:
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return data
    return None


def _show(data: Any, as_json: bool, preferred: list[str], *unwrap_keys: str) -> None:
    rows = None if as_json else _rows(data, *unwrap_keys)
    if rows is None:
        _print_json(data)
    else:
        _print_table(rows, preferred)


def _cmd_info(args: argparse.Namespace) -> None:
    with PeriscopeClient() as lg:
        _print_json(lg.config())


def _cmd_devices(args: argparse.Namespace) -> None:
    with PeriscopeClient() as lg:
        _show(lg.devices(), args.json, _DEVICE_COLUMNS, "devices")


def _cmd_commands(args: argparse.Namespace) -> None:
    with PeriscopeClient() as lg:
        _show(lg.commands(), args.json, _COMMAND_COLUMNS, "commands")


def _cmd_filters(args: argparse.Namespace) -> None:
    with PeriscopeClient() as lg:
        _show(lg.filters(), args.json, ["name", "platforms", "help"], "filters")


def _cmd_exec(args: argparse.Namespace) -> None:
    with PeriscopeClient() as lg:
        result = lg.execute(
            args.command,
            args.targets,
            parameter=args.parameter,
            filter=args.filter,
        )
    if args.json or not isinstance(result, dict) or "outputs" not in result:
        _print_json(result)
        return
    for output in result["outputs"]:
        if isinstance(output, dict):
            target = output.get("target") or output.get("device") or "?"
            body = output.get("output")
            if not isinstance(body, str):
                body = json.dumps(output, indent=2)
            status = output.get("status")
        else:
            target, body, status = "?", str(output), None
        suffix = f"  [status: {status}]" if status not in (None, "ok") else ""
        print(f"=== {target}{suffix} ===")
        print(body.rstrip("\n"))
        print()


def _show_tool_call(name: str, raw_arguments: str) -> None:
    print(f"  [{name} {raw_arguments}]", flush=True)


def _onboard_llm() -> Settings:
    """First-run setup: collect the OpenAI-compatible endpoint interactively."""
    print("No LLM endpoint is configured yet (OPENAI_BASE_URL / OPENAI_API_KEY are both unset).")
    print("The agent works with any OpenAI-compatible endpoint, for example:")
    print("  OpenAI:    leave the base URL blank and paste an API key")
    print("  Ollama:    http://localhost:11434/v1   (no key needed)")
    print("  LM Studio: http://localhost:1234/v1    (no key needed)")
    if not sys.stdin.isatty():
        print(
            "error: non-interactive session - set OPENAI_BASE_URL / OPENAI_API_KEY / "
            "I2A_MODEL in .env or the environment.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print()
    base_url = input("LLM base URL [blank = api.openai.com]: ").strip()
    api_key = getpass.getpass("API key (hidden) [blank if the endpoint needs none]: ").strip()
    if not base_url and not api_key:
        print("error: nothing entered - the agent needs at least a base URL or an API key.",
              file=sys.stderr)
        raise SystemExit(2)
    model = input(f"Model name [{DEFAULT_MODEL}]: ").strip() or DEFAULT_MODEL

    values = {"I2A_MODEL": model}
    if base_url:
        values["OPENAI_BASE_URL"] = base_url
    if api_key:
        values["OPENAI_API_KEY"] = api_key
    os.environ.update(values)  # takes effect for this run

    save = input("Save to .env for future runs? [Y/n]: ").strip().lower()
    if save in ("", "y", "yes"):
        print(f"(saved to {save_env_values(values)})")
    print()
    return Settings.from_env()


def _make_agent() -> Internet2Agent:
    settings = Settings.from_env()
    if not (settings.openai_api_key or settings.openai_base_url):
        settings = _onboard_llm()
    return Internet2Agent(settings=settings)


def _cmd_ask(args: argparse.Namespace) -> None:
    with _make_agent() as agent:
        print(agent.ask(" ".join(args.question), on_tool=_show_tool_call))


def _cmd_chat(args: argparse.Namespace) -> None:
    with _make_agent() as agent:
        print("Internet2 Periscope agent. Type 'exit' to quit, 'reset' to clear history.")
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not line:
                continue
            if line.lower() in {"exit", "quit"}:
                return
            if line.lower() == "reset":
                agent.reset()
                print("(history cleared)")
                continue
            try:
                reply = agent.ask(line, on_tool=_show_tool_call)
            except KeyboardInterrupt:
                print("\n(interrupted - session kept, ask again or 'exit')")
                continue
            print(f"agent> {reply}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="internet2agent",
        description="Internet2 Periscope Looking Glass - direct client and Claude agent",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    def lg_sub(name: str, help_text: str, func: Any) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--json", action="store_true", help="print raw JSON")
        p.set_defaults(func=func)
        return p

    lg_sub("info", "show service limits (rate limit, max targets)", _cmd_info)
    lg_sub("devices", "list available devices", _cmd_devices)
    lg_sub("commands", "list supported commands", _cmd_commands)
    lg_sub("filters", "list output filters", _cmd_filters)

    p_exec = lg_sub("exec", "run a command on one or more devices", _cmd_exec)
    p_exec.add_argument("command", help='command name, e.g. "show bgp" or "traceroute"')
    p_exec.add_argument(
        "-t", "--targets", nargs="+", required=True, metavar="DEVICE",
        help="device IDs from 'devices' (max 10)",
    )
    p_exec.add_argument("-p", "--parameter", help='appended argument, e.g. "10.0.0.0/8" or "summary"')
    p_exec.add_argument("-f", "--filter", help='output filter, e.g. "include bgp" or "exclude ^$"')

    p_ask = sub.add_parser("ask", help="ask the LLM agent one question (configure LLM in .env)")
    p_ask.add_argument("question", nargs="+", help="the question to ask")
    p_ask.set_defaults(func=_cmd_ask)

    p_chat = sub.add_parser("chat", help="interactive agent session (configure LLM in .env)")
    p_chat.set_defaults(func=_cmd_chat)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except PeriscopeError as exc:
        print(f"periscope error: {exc}", file=sys.stderr)
        return 1
    except openai.AuthenticationError:
        print(
            "error: the LLM endpoint rejected the API key.\n"
            "Set OPENAI_API_KEY (and OPENAI_BASE_URL if not using OpenAI) in .env.",
            file=sys.stderr,
        )
        return 2
    except openai.NotFoundError as exc:
        print(
            f"error: the LLM endpoint returned 404: {exc}\n"
            "Check I2A_MODEL (and OPENAI_BASE_URL) in .env - the endpoint may not serve that model.",
            file=sys.stderr,
        )
        return 2
    except openai.RateLimitError:
        print("error: LLM endpoint rate limit hit - try again shortly.", file=sys.stderr)
        return 2
    except openai.APIStatusError as exc:
        print(f"error: LLM endpoint returned {exc.status_code}: {exc.message}", file=sys.stderr)
        return 2
    except openai.APIConnectionError:
        print(
            "error: could not reach the LLM endpoint - check OPENAI_BASE_URL in .env "
            "and that the server is running.",
            file=sys.stderr,
        )
        return 2
    except openai.OpenAIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except TimeoutError:
        print("error: Periscope MCP call timed out.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
