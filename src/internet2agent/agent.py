"""Natural-language agent for the Internet2 Periscope service.

Works with any OpenAI-compatible chat-completions endpoint (OpenAI, Ollama,
vLLM, LM Studio, OpenRouter, ...). The agent fetches the Looking Glass tool
schemas from the Periscope MCP server, exposes them to the model as function
tools, executes the model's tool calls through :class:`PeriscopeClient`, and
loops until the model produces a final answer.

Configuration (``.env`` / environment):

- ``OPENAI_BASE_URL`` - endpoint base URL (unset = api.openai.com)
- ``OPENAI_API_KEY``  - API key (optional for local endpoints)
- ``I2A_MODEL``       - model name to request
"""

from __future__ import annotations

import json
from typing import Any, Callable

from openai import OpenAI

from .config import Settings
from .periscope import PeriscopeClient, PeriscopeError

SYSTEM_PROMPT = """\
You are a network operations assistant for the Internet2 research & education
backbone. You answer questions by running diagnostics through Periscope, the
Internet2 Looking Glass MCP service, using the tools provided.

Workflow for every diagnostic question:
1. Call lg_devices to discover device IDs and platforms. Device IDs passed to
   lg_execute must exactly match the `name` field from lg_devices.
2. Call lg_commands to confirm the command is supported on the target
   platforms; call lg_filters if you want to narrow the output.
3. Call lg_execute with at most 10 targets. Commands must match documented
   syntax exactly - no abbreviations. When targeting mixed platforms, the
   command and filter must be supported on every target platform.

Report findings concisely. Quote the raw output lines that support your
conclusion and name the device each line came from. If a command fails or a
device is unknown, say so plainly and suggest the closest valid option.
"""

_MAX_STEPS = 25
_MAX_RESULT_CHARS = 60_000  # keep one giant router dump from blowing the context window


class Internet2Agent:
    """Multi-turn natural-language interface to Periscope.

    >>> with Internet2Agent() as agent:
    ...     print(agent.ask("Which devices are in Chicago?"))

    History is kept on the instance so follow-up questions work; call
    :meth:`reset` to start fresh. Use as a context manager (or call
    :meth:`close`) to release the underlying MCP session.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: OpenAI | None = None,
        periscope: PeriscopeClient | None = None,
        model: str | None = None,
        max_steps: int = _MAX_STEPS,
        max_result_chars: int = _MAX_RESULT_CHARS,
    ) -> None:
        self.settings = settings or Settings.from_env()
        if client is None:
            client = OpenAI(
                base_url=self.settings.openai_base_url,
                # Local OpenAI-compatible servers usually ignore the key, but
                # the SDK requires one - send a placeholder when unset.
                api_key=self.settings.openai_api_key or "not-needed",
            )
        self.client = client
        self.model = model or self.settings.model
        self.max_steps = max_steps
        self.max_result_chars = max_result_chars
        self.periscope = periscope or PeriscopeClient(settings=self.settings)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._tools: list[dict[str, Any]] | None = None

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        self.periscope.close()

    def __enter__(self) -> "Internet2Agent":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # -- tool plumbing --------------------------------------------------------

    def _openai_tools(self) -> list[dict[str, Any]]:
        """Looking Glass tools converted to OpenAI function-tool schemas."""
        if self._tools is None:
            self._tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description") or "",
                        "parameters": tool.get("input_schema")
                        or {"type": "object", "properties": {}},
                    },
                }
                for tool in self.periscope.tools()
            ]
        return self._tools

    def _run_tool_call(self, name: Any, raw_arguments: str) -> str:
        if not isinstance(name, str) or not name:
            return "ERROR: tool call is missing a valid tool name"
        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except ValueError:
            return f"ERROR: tool arguments were not valid JSON: {raw_arguments!r}"
        if not isinstance(arguments, dict):
            return f"ERROR: tool arguments must be a JSON object, got: {raw_arguments!r}"
        try:
            result = self.periscope.call(name, arguments)
        except (PeriscopeError, TimeoutError) as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # transport/protocol failures - let the model adapt
            return f"ERROR: {type(exc).__name__}: {exc}"
        content = result if isinstance(result, str) else json.dumps(result, default=str)
        if len(content) > self.max_result_chars:
            dropped = len(content) - self.max_result_chars
            content = content[: self.max_result_chars] + f"\n...[truncated {dropped} characters]"
        return content

    # -- conversation ---------------------------------------------------------

    def ask(
        self,
        question: str,
        on_tool: Callable[[str, str], None] | None = None,
    ) -> str:
        """Ask a question and return the assistant's final text.

        ``on_tool`` (if given) is called with ``(tool_name, raw_arguments)``
        before each tool execution - useful for progress display.
        """
        self.messages.append({"role": "user", "content": question})
        for _ in range(self.max_steps):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self._openai_tools(),
            )
            message = response.choices[0].message

            # Normalize tool calls defensively: lenient OpenAI-compatible servers
            # can yield a missing name or dict-typed arguments.
            tool_calls = []
            for call in message.tool_calls or []:
                function = getattr(call, "function", None)
                raw = getattr(function, "arguments", None)
                if isinstance(raw, dict):
                    raw = json.dumps(raw)
                elif not isinstance(raw, str):
                    raw = ""
                tool_calls.append(
                    {
                        "id": getattr(call, "id", None) or "",
                        "name": getattr(function, "name", None),
                        "arguments": raw,
                    }
                )

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.content}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"] or "",
                            "arguments": call["arguments"],
                        },
                    }
                    for call in tool_calls
                ]
            self.messages.append(assistant_msg)

            if not tool_calls:
                return message.content or ""

            for call in tool_calls:
                if on_tool is not None:
                    on_tool(call["name"] or "?", call["arguments"])
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": self._run_tool_call(call["name"], call["arguments"]),
                    }
                )
        return "[stopped: the model kept calling tools past the step limit]"

    def reset(self) -> None:
        """Clear the conversation history (keeps the system prompt)."""
        del self.messages[1:]
