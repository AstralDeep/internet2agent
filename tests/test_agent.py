import json
from types import SimpleNamespace

from internet2agent.agent import SYSTEM_PROMPT, Internet2Agent
from internet2agent.config import Settings
from internet2agent.periscope import PeriscopeError

FAKE_TOOLS = [
    {"name": "lg_devices", "description": "List devices", "input_schema": {"type": "object", "properties": {}}},
    {
        "name": "lg_execute",
        "description": "Run a command",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}, "targets": {"type": "array"}},
            "required": ["command", "targets"],
        },
    },
]


class FakePeriscope:
    def __init__(self):
        self.calls = []
        self.closed = False

    def tools(self):
        return FAKE_TOOLS

    def call(self, tool, arguments=None):
        self.calls.append((tool, arguments))
        if tool == "boom":
            raise PeriscopeError("device unknown")
        return {"tool": tool, "arguments": arguments}

    def close(self):
        self.closed = True


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAI:
    """Records chat.completions.create kwargs and plays back scripted responses."""

    def __init__(self, responses):
        self.requests = []
        self._responses = list(responses)
        create = self._create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    def _create(self, **kwargs):
        # Snapshot messages: the agent mutates one shared list across steps.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


def _agent(responses, periscope=None):
    return Internet2Agent(
        settings=Settings(model="test-model"),
        client=FakeOpenAI(responses),
        periscope=periscope or FakePeriscope(),
    )


def test_plain_answer_no_tools():
    agent = _agent([_completion(content="42 devices")])
    assert agent.ask("how many devices?") == "42 devices"
    (request,) = agent.client.requests
    assert request["model"] == "test-model"
    assert request["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    names = [t["function"]["name"] for t in request["tools"]]
    assert names == ["lg_devices", "lg_execute"]
    assert request["tools"][1]["function"]["parameters"]["required"] == ["command", "targets"]


def test_tool_loop_executes_and_feeds_back_results():
    agent = _agent(
        [
            _completion(tool_calls=[_tool_call("tc1", "lg_devices", "{}")]),
            _completion(content="found rtr1"),
        ]
    )
    seen = []
    reply = agent.ask("list devices", on_tool=lambda n, a: seen.append((n, a)))

    assert reply == "found rtr1"
    assert seen == [("lg_devices", "{}")]
    assert agent.periscope.calls == [("lg_devices", {})]

    second_messages = agent.client.requests[1]["messages"]
    assert second_messages[-2]["role"] == "assistant"
    assert second_messages[-2]["tool_calls"][0]["function"]["name"] == "lg_devices"
    tool_msg = second_messages[-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "tc1"
    assert json.loads(tool_msg["content"]) == {"tool": "lg_devices", "arguments": {}}


def test_parallel_tool_calls_all_answered():
    agent = _agent(
        [
            _completion(
                tool_calls=[
                    _tool_call("tc1", "lg_devices", "{}"),
                    _tool_call("tc2", "lg_commands", "{}"),
                ]
            ),
            _completion(content="done"),
        ]
    )
    assert agent.ask("both") == "done"
    tool_msgs = [m for m in agent.client.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["tc1", "tc2"]


def test_tool_error_reported_to_model_not_raised():
    agent = _agent(
        [
            _completion(tool_calls=[_tool_call("tc1", "boom", "{}")]),
            _completion(content="that device does not exist"),
        ]
    )
    assert agent.ask("break it") == "that device does not exist"
    tool_msg = agent.client.requests[1]["messages"][-1]
    assert tool_msg["content"].startswith("ERROR:")


def test_dict_arguments_from_lenient_server_handled():
    # Some OpenAI-compatible servers hand back arguments as a JSON object.
    agent = _agent(
        [
            _completion(tool_calls=[_tool_call("tc1", "lg_devices", {"a": 1})]),
            _completion(content="ok"),
        ]
    )
    assert agent.ask("go") == "ok"
    assert agent.periscope.calls == [("lg_devices", {"a": 1})]
    sent = agent.client.requests[1]["messages"][-2]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(sent) == {"a": 1}


def test_missing_tool_name_reported_not_executed():
    agent = _agent(
        [
            _completion(tool_calls=[_tool_call("tc1", None, "{}")]),
            _completion(content="sorry"),
        ]
    )
    assert agent.ask("go") == "sorry"
    tool_msg = agent.client.requests[1]["messages"][-1]
    assert "missing a valid tool name" in tool_msg["content"]
    assert agent.periscope.calls == []


def test_unexpected_tool_exception_fed_back_to_model():
    class ExplodingPeriscope(FakePeriscope):
        def call(self, tool, arguments=None):
            raise ValueError("transport wedged")

    agent = _agent(
        [
            _completion(tool_calls=[_tool_call("tc1", "lg_devices", "{}")]),
            _completion(content="giving up"),
        ],
        periscope=ExplodingPeriscope(),
    )
    assert agent.ask("go") == "giving up"
    tool_msg = agent.client.requests[1]["messages"][-1]
    assert tool_msg["content"] == "ERROR: ValueError: transport wedged"


def test_giant_tool_output_truncated():
    class ChattyPeriscope(FakePeriscope):
        def call(self, tool, arguments=None):
            return "x" * 500

    agent = _agent(
        [
            _completion(tool_calls=[_tool_call("tc1", "lg_devices", "{}")]),
            _completion(content="done"),
        ],
        periscope=ChattyPeriscope(),
    )
    agent.max_result_chars = 100
    agent.ask("go")
    tool_msg = agent.client.requests[1]["messages"][-1]
    assert len(tool_msg["content"]) < 200
    assert "[truncated 400 characters]" in tool_msg["content"]


def test_invalid_json_arguments_reported_to_model():
    agent = _agent(
        [
            _completion(tool_calls=[_tool_call("tc1", "lg_devices", "{not json")]),
            _completion(content="retried"),
        ]
    )
    assert agent.ask("go") == "retried"
    tool_msg = agent.client.requests[1]["messages"][-1]
    assert "not valid JSON" in tool_msg["content"]
    assert agent.periscope.calls == []  # nothing was executed


def test_step_limit_stops_runaway_loop():
    tool_reply = _completion(tool_calls=[_tool_call("tc", "lg_devices", "{}")])
    agent = _agent([tool_reply] * 3)
    agent.max_steps = 3
    assert "step limit" in agent.ask("loop forever")


def test_history_accumulates_and_reset_keeps_system_prompt():
    agent = _agent([_completion(content="one"), _completion(content="two")])
    agent.ask("first")
    agent.ask("second")
    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    agent.reset()
    assert agent.messages == [{"role": "system", "content": SYSTEM_PROMPT}]


def test_context_manager_closes_periscope():
    periscope = FakePeriscope()
    with _agent([_completion(content="hi")], periscope=periscope) as agent:
        agent.ask("hello")
    assert periscope.closed
