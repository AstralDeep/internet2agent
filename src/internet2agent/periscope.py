"""Direct Python client for the Internet2 Periscope Looking Glass MCP server.

The server exposes five tools:

- ``lg_config``   - service limits (rate limit, max targets per request)
- ``lg_devices``  - available devices (name/ID, location, type, platform)
- ``lg_commands`` - supported commands and the platforms they run on
- ``lg_filters``  - output filters (``include``/``exclude`` + regex pattern)
- ``lg_execute``  - run a command on up to 10 devices

``AsyncPeriscopeClient`` is the asyncio interface; ``PeriscopeClient`` is a
synchronous facade that runs the MCP session on a background event-loop
thread so one session can be reused across calls.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterable

from mcp import Client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

try:
    from mcp.shared.exceptions import MCPError
except ImportError:  # pragma: no cover - name used by pre-2.1 releases
    from mcp.shared.exceptions import McpError as MCPError

from .config import Settings

_CALL_TIMEOUT = 180.0
_CONNECT_TIMEOUT = 30.0
_CLOSE_GRACE = 15.0

_LIST_TOOLS = object()  # queue sentinel: "run list_tools" (None means "shut down")


class PeriscopeError(RuntimeError):
    """Raised when the Periscope MCP server rejects or fails a request."""


class _SessionClosedError(PeriscopeError):
    """A queued call was abandoned because its session went away."""


@asynccontextmanager
async def _transport_with_headers(url: str, headers: dict[str, str]) -> AsyncIterator[Any]:
    # mcp's Client(url) offers no header hook, so build the transport ourselves.
    async with create_mcp_http_client(headers=headers) as http_client:
        async with streamable_http_client(url, http_client=http_client) as streams:
            yield streams


def _decode(result: Any) -> Any:
    """Decode a CallToolResult: structured JSON if present, else (JSON) text."""
    structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
    if structured:
        return structured
    texts = [b.text for b in (result.content or []) if getattr(b, "type", None) == "text"]
    text = "\n".join(texts)
    try:
        return json.loads(text)
    except ValueError:
        return text


def _is_error(result: Any) -> bool:
    return bool(getattr(result, "isError", False) or getattr(result, "is_error", False))


class AsyncPeriscopeClient:
    """Async MCP client for Periscope. Use as an async context manager.

    ``url`` and ``auth_token`` default to the environment (``PERISCOPE_MCP_URL``,
    ``PERISCOPE_AUTH_TOKEN``); the server currently requires no token.
    """

    def __init__(
        self,
        url: str | None = None,
        auth_token: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        if url is None or auth_token is None:
            settings = settings or Settings.from_env()
        self.url = url or settings.mcp_url
        self.auth_token = auth_token if auth_token is not None else settings.periscope_token
        self._client: Client | None = None

    async def __aenter__(self) -> "AsyncPeriscopeClient":
        if self.auth_token:
            target: Any = _transport_with_headers(
                self.url, {"Authorization": f"Bearer {self.auth_token}"}
            )
        else:
            target = self.url
        client = Client(target)
        await client.__aenter__()
        self._client = client
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.__aexit__(exc_type, exc, tb)

    async def call(self, tool: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call one MCP tool and return its decoded result."""
        if self._client is None:
            raise RuntimeError("not connected - use 'async with AsyncPeriscopeClient() as lg:'")
        try:
            result = await self._client.call_tool(tool, arguments or {})
        except MCPError as exc:
            # Protocol-level rejections (unknown device, invalid characters, rate
            # limit) arrive as JSON-RPC errors, not isError results.
            raise PeriscopeError(f"{tool} failed: {exc}") from exc
        payload = _decode(result)
        if _is_error(result):
            raise PeriscopeError(f"{tool} failed: {payload}")
        return payload

    async def tools(self) -> list[dict[str, Any]]:
        """The server's tool inventory: name, description, and input schema."""
        if self._client is None:
            raise RuntimeError("not connected - use 'async with AsyncPeriscopeClient() as lg:'")
        result = await self._client.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": getattr(tool, "input_schema", None)
                or getattr(tool, "inputSchema", None),
            }
            for tool in result.tools
        ]

    async def config(self) -> Any:
        """Service limits: rate limit and max targets per request."""
        return await self.call("lg_config")

    async def devices(self) -> Any:
        """All available devices with name (ID), location, type, and platform."""
        return await self.call("lg_devices")

    async def commands(self) -> Any:
        """Supported commands with help text and supported platforms."""
        return await self.call("lg_commands")

    async def filters(self) -> Any:
        """Available output filters and the platforms they support."""
        return await self.call("lg_filters")

    async def execute(
        self,
        command: str,
        targets: Iterable[str],
        parameter: str | None = None,
        filter: str | None = None,
    ) -> Any:
        """Run ``command`` on up to 10 devices.

        ``targets`` must exactly match ``name`` values from :meth:`devices`.
        ``parameter`` is appended to the command (e.g. a prefix for
        ``show route``); ``filter`` is a filter name plus regex pattern,
        e.g. ``"include bgp"`` or ``"exclude ^$"``.
        """
        arguments: dict[str, Any] = {"command": command, "targets": list(targets)}
        if parameter is not None:
            arguments["parameter"] = parameter
        if filter is not None:
            arguments["filter"] = filter
        return await self.call("lg_execute", arguments)


class PeriscopeClient:
    """Synchronous facade over :class:`AsyncPeriscopeClient`.

    The MCP session lives inside a single task on a background event loop
    (mcp's transports use anyio cancel scopes, which must be entered and
    exited by the same task). Calls are handed to that task over a queue.

    Connects lazily on first use and reconnects automatically if the session
    dies (server restart, dropped connection). Call :meth:`close` (or use it
    as a context manager) to disconnect. Instances are not thread-safe.
    """

    def __init__(
        self,
        url: str | None = None,
        auth_token: str | None = None,
        settings: Settings | None = None,
        call_timeout: float = _CALL_TIMEOUT,
        connect_timeout: float = _CONNECT_TIMEOUT,
    ) -> None:
        self._client_args = (url, auth_token, settings)
        self._call_timeout = call_timeout
        self._connect_timeout = connect_timeout
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._queue: asyncio.Queue[Any] | None = None
        self._worker: "concurrent.futures.Future[None] | None" = None
        self._worker_exited: threading.Event | None = None

    # -- session lifecycle ----------------------------------------------------

    def _ensure_connected(self) -> None:
        with self._lock:
            if self._loop is not None:
                return
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever, name="periscope-mcp", daemon=True)
            thread.start()

            queue: asyncio.Queue[Any] = asyncio.Queue()
            ready: "concurrent.futures.Future[None]" = concurrent.futures.Future()
            exited = threading.Event()
            worker = asyncio.run_coroutine_threadsafe(
                self._session_worker(queue, ready, exited), loop
            )
            try:
                ready.result(self._connect_timeout + 10)
            except BaseException:
                worker.cancel()
                exited.wait(5)
                loop.call_soon_threadsafe(loop.stop)
                thread.join(timeout=5)
                loop.close()
                raise
            self._loop = loop
            self._thread = thread
            self._queue = queue
            self._worker = worker
            self._worker_exited = exited

    async def _session_worker(
        self,
        queue: "asyncio.Queue[Any]",
        ready: "concurrent.futures.Future[None]",
        exited: threading.Event,
    ) -> None:
        """Own the MCP session for its whole lifetime and serve queued calls."""
        try:
            client = AsyncPeriscopeClient(*self._client_args)
            try:
                await asyncio.wait_for(client.__aenter__(), self._connect_timeout)
            except BaseException as exc:
                if not ready.done():
                    if isinstance(exc, Exception):
                        ready.set_exception(exc)
                    else:
                        ready.set_exception(PeriscopeError(f"connect failed: {exc!r}"))
                return
            try:
                ready.set_result(None)
                while True:
                    item = await queue.get()
                    if item is None:
                        return
                    tool, arguments, future = item
                    try:
                        if tool is _LIST_TOOLS:
                            coro = client.tools()
                        else:
                            coro = client.call(tool, arguments)
                        result = await asyncio.wait_for(coro, self._call_timeout)
                    except asyncio.CancelledError:
                        # close() gave up waiting - fail the caller with a normal
                        # exception and let the cancellation complete.
                        if not future.done():
                            future.set_exception(
                                PeriscopeError("client closed while a call was in flight")
                            )
                        raise
                    except asyncio.TimeoutError:
                        if not future.done():
                            future.set_exception(
                                TimeoutError(
                                    f"Periscope call timed out after {self._call_timeout:g}s"
                                )
                            )
                    except Exception as exc:  # per-call errors keep the session alive
                        if not future.done():
                            future.set_exception(exc)
                    except BaseException as exc:
                        if not future.done():
                            future.set_exception(exc)
                        raise
                    else:
                        if not future.done():
                            future.set_result(result)
            finally:
                try:
                    await asyncio.wait_for(client.__aexit__(None, None, None), 15)
                except BaseException:
                    pass  # best effort - the transport may already be gone
        finally:
            # Fail anything still queued so callers don't wait out their timeout,
            # then signal the sync side that this session is over.
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is not None and not item[2].done():
                    item[2].set_exception(_SessionClosedError("Periscope session closed"))
            exited.set()

    def _call(self, tool: Any, arguments: dict[str, Any] | None = None, *, _retry: bool = True) -> Any:
        self._ensure_connected()
        exited = self._worker_exited
        if exited is not None and exited.is_set():
            # The session died since the last call (server restart, dropped
            # connection). Reap it and reconnect once.
            self.close()
            self._ensure_connected()

        loop, queue = self._loop, self._queue
        if loop is None or queue is None:
            raise PeriscopeError("client is closed")
        future: "concurrent.futures.Future[Any]" = concurrent.futures.Future()
        loop.call_soon_threadsafe(queue.put_nowait, (tool, arguments, future))
        try:
            # The worker enforces the real per-call timeout; this is a backstop.
            return future.result(self._call_timeout + 15)
        except _SessionClosedError:
            # Raced a dying session: the call was queued while the worker was
            # unwinding. Reap it and retry once on a fresh session.
            if not _retry:
                raise
            self.close()
            return self._call(tool, arguments, _retry=False)
        except (TimeoutError, concurrent.futures.TimeoutError) as exc:
            if not future.done():
                future.cancel()
            # Normalize to builtin TimeoutError (they differ on Python 3.10).
            raise TimeoutError(
                str(exc) or f"Periscope call timed out after {self._call_timeout:g}s"
            ) from None

    def close(self) -> None:
        """Disconnect and stop the background loop. Safe to call twice."""
        with self._lock:
            loop, thread, queue, worker, exited = (
                self._loop,
                self._thread,
                self._queue,
                self._worker,
                self._worker_exited,
            )
            self._loop = self._thread = self._queue = self._worker = self._worker_exited = None
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, None)
            if not exited.wait(_CLOSE_GRACE):
                worker.cancel()  # a call is stuck in flight - force the session down
                exited.wait(10)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            loop.close()

    def __enter__(self) -> "PeriscopeClient":
        self._ensure_connected()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # -- Looking Glass tools --------------------------------------------------

    def call(self, tool: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call an arbitrary Periscope MCP tool and return its decoded result."""
        if not isinstance(tool, str) or not tool:
            raise PeriscopeError(f"invalid tool name: {tool!r}")
        return self._call(tool, arguments)

    def tools(self) -> list[dict[str, Any]]:
        """The server's tool inventory: name, description, and input schema."""
        return self._call(_LIST_TOOLS)

    def config(self) -> Any:
        """Service limits: rate limit and max targets per request."""
        return self.call("lg_config")

    def devices(self) -> Any:
        """All available devices with name (ID), location, type, and platform."""
        return self.call("lg_devices")

    def commands(self) -> Any:
        """Supported commands with help text and supported platforms."""
        return self.call("lg_commands")

    def filters(self) -> Any:
        """Available output filters and the platforms they support."""
        return self.call("lg_filters")

    def execute(
        self,
        command: str,
        targets: Iterable[str],
        parameter: str | None = None,
        filter: str | None = None,
    ) -> Any:
        """Run ``command`` on up to 10 devices. See AsyncPeriscopeClient.execute."""
        arguments: dict[str, Any] = {"command": command, "targets": list(targets)}
        if parameter is not None:
            arguments["parameter"] = parameter
        if filter is not None:
            arguments["filter"] = filter
        return self.call("lg_execute", arguments)
