import json
from types import SimpleNamespace

import pytest

import internet2agent.periscope as periscope
from internet2agent.periscope import PeriscopeClient, PeriscopeError, _decode, _is_error


def _text_result(payload, is_error=False):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        structuredContent=None,
        isError=is_error,
    )


class TestDecode:
    def test_prefers_structured_content(self):
        result = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ignored")],
            structuredContent={"a": 1},
            isError=False,
        )
        assert _decode(result) == {"a": 1}

    def test_snake_case_structured_content(self):
        result = SimpleNamespace(content=[], structured_content={"b": 2}, isError=False)
        assert _decode(result) == {"b": 2}

    def test_parses_json_text(self):
        assert _decode(_text_result({"rate_limit": 60})) == {"rate_limit": 60}

    def test_falls_back_to_raw_text(self):
        assert _decode(_text_result("not json")) == "not json"

    def test_empty_content(self):
        assert _decode(SimpleNamespace(content=[], structuredContent=None)) == ""

    def test_is_error_both_spellings(self):
        assert _is_error(SimpleNamespace(isError=True))
        assert _is_error(SimpleNamespace(is_error=True))
        assert not _is_error(SimpleNamespace(isError=False))


class FakeCrash(BaseException):
    """Non-Exception failure that kills the session worker."""


class FakeAsyncClient:
    """Stands in for AsyncPeriscopeClient in sync-facade tests."""

    instances: list["FakeAsyncClient"] = []

    def __init__(self, url=None, auth_token=None, settings=None):
        self.calls = []
        self.entered = False
        self.exited = False
        FakeAsyncClient.instances.append(self)

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True

    async def call(self, tool, arguments=None):
        self.calls.append((tool, arguments))
        if tool == "boom":
            raise PeriscopeError("boom")
        if tool == "crash":
            raise FakeCrash("worker down")
        return {"tool": tool, "arguments": arguments}

    async def tools(self):
        return [{"name": "lg_config", "description": "", "input_schema": {}}]


@pytest.fixture
def fake_async(monkeypatch):
    FakeAsyncClient.instances.clear()
    monkeypatch.setattr(periscope, "AsyncPeriscopeClient", FakeAsyncClient)
    yield FakeAsyncClient


class TestSyncFacade:
    def test_calls_round_trip(self, fake_async):
        with PeriscopeClient() as lg:
            assert lg.config() == {"tool": "lg_config", "arguments": None}
            assert lg.devices()["tool"] == "lg_devices"
            result = lg.execute("show bgp", ["rtr1"], parameter="summary", filter="include x")
        assert result["arguments"] == {
            "command": "show bgp",
            "targets": ["rtr1"],
            "parameter": "summary",
            "filter": "include x",
        }
        (instance,) = fake_async.instances
        assert instance.entered and instance.exited

    def test_error_keeps_session_alive(self, fake_async):
        with PeriscopeClient() as lg:
            with pytest.raises(PeriscopeError):
                lg.call("boom")
            assert lg.config()["tool"] == "lg_config"

    def test_close_is_idempotent(self, fake_async):
        lg = PeriscopeClient()
        lg.config()
        lg.close()
        lg.close()
        (instance,) = fake_async.instances
        assert instance.exited

    def test_lazy_connect(self, fake_async):
        lg = PeriscopeClient()
        assert fake_async.instances == []
        lg.devices()
        assert len(fake_async.instances) == 1
        lg.close()

    def test_reuse_after_close_reconnects(self, fake_async):
        lg = PeriscopeClient()
        lg.config()
        lg.close()
        assert lg.config()["tool"] == "lg_config"
        assert len(fake_async.instances) == 2
        lg.close()

    def test_dead_worker_reconnects_on_next_call(self, fake_async):
        lg = PeriscopeClient()
        with pytest.raises(FakeCrash):
            lg.call("crash")
        # The worker died; the next call must reap it and reconnect.
        assert lg.config()["tool"] == "lg_config"
        assert len(fake_async.instances) == 2
        assert fake_async.instances[0].exited
        lg.close()

    def test_list_tools(self, fake_async):
        with PeriscopeClient() as lg:
            tools = lg.tools()
        assert tools[0]["name"] == "lg_config"

    def test_invalid_tool_name_rejected_without_call(self, fake_async):
        with PeriscopeClient() as lg:
            with pytest.raises(PeriscopeError):
                lg.call(None)
            with pytest.raises(PeriscopeError):
                lg.call("")
        assert all(instance.calls == [] for instance in fake_async.instances)


class TestExecuteArguments:
    def test_optional_arguments_omitted(self, fake_async):
        with PeriscopeClient() as lg:
            result = lg.execute("ping", ("rtr1", "rtr2"))
        assert result["arguments"] == {"command": "ping", "targets": ["rtr1", "rtr2"]}
