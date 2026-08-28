from internet2agent.config import DEFAULT_MCP_URL, DEFAULT_MODEL, Settings, save_env_values


def test_defaults_when_env_empty(monkeypatch):
    # Blank strings mask any values in the project's real .env (dotenv never
    # overrides existing environment variables), keeping this test hermetic.
    for var in ("PERISCOPE_MCP_URL", "PERISCOPE_AUTH_TOKEN", "I2A_MODEL"):
        monkeypatch.setenv(var, "")
    settings = Settings.from_env()
    assert settings.mcp_url == DEFAULT_MCP_URL
    assert settings.model == DEFAULT_MODEL


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("PERISCOPE_MCP_URL", "https://example.org/mcp")
    monkeypatch.setenv("PERISCOPE_AUTH_TOKEN", "sekrit")
    monkeypatch.setenv("I2A_MODEL", "claude-sonnet-5")
    settings = Settings.from_env()
    assert settings.mcp_url == "https://example.org/mcp"
    assert settings.periscope_token == "sekrit"
    assert settings.model == "claude-sonnet-5"


def test_blank_values_treated_as_unset(monkeypatch):
    monkeypatch.setenv("PERISCOPE_MCP_URL", "  ")
    monkeypatch.setenv("PERISCOPE_AUTH_TOKEN", "")
    settings = Settings.from_env()
    assert settings.mcp_url == DEFAULT_MCP_URL
    assert settings.periscope_token is None


def test_save_env_values_updates_existing_and_appends(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nOPENAI_API_KEY=\nI2A_MODEL=old-model\n", encoding="utf-8")
    save_env_values({"OPENAI_API_KEY": "sk-new", "OPENAI_BASE_URL": "http://host/v1"}, env)
    lines = env.read_text(encoding="utf-8").splitlines()
    assert "# comment" in lines
    assert "OPENAI_API_KEY=sk-new" in lines
    assert "I2A_MODEL=old-model" in lines           # untouched
    assert "OPENAI_BASE_URL=http://host/v1" in lines  # appended
    assert lines.count("OPENAI_API_KEY=sk-new") == 1


def test_save_env_values_creates_missing_file(tmp_path):
    env = tmp_path / ".env"
    save_env_values({"I2A_MODEL": "m"}, env)
    assert env.read_text(encoding="utf-8") == "I2A_MODEL=m\n"
