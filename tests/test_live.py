"""Smoke tests against the live Periscope MCP server.

Excluded from the default run (see pyproject). Run explicitly with:

    pytest -m network
"""

import pytest

from internet2agent.periscope import PeriscopeClient

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def lg():
    client = PeriscopeClient()
    yield client
    client.close()


def test_config_reports_limits(lg):
    config = lg.config()
    assert config["max_targets"] >= 1
    assert config["rate_limit"] >= 1


def test_devices_nonempty(lg):
    devices = lg.devices()
    assert isinstance(devices, list) and devices
    assert "name" in devices[0]


def test_commands_nonempty(lg):
    commands = lg.commands()
    assert isinstance(commands, list) and commands
    assert "name" in commands[0]
