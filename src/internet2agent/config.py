"""Environment-driven configuration for internet2agent.

Credentials and settings come from environment variables, optionally loaded
from a ``.env`` file (current directory first, then the project root):

- ``OPENAI_BASE_URL``      - OpenAI-compatible endpoint for the agent's LLM
                             (unset = api.openai.com; e.g. Ollama:
                             ``http://localhost:11434/v1``).
- ``OPENAI_API_KEY``       - API key for that endpoint (optional for local
                             endpoints that don't check keys).
- ``I2A_MODEL``            - model name the agent requests (default: gpt-4o;
                             set it to whatever your endpoint serves).
- ``PERISCOPE_MCP_URL``    - Periscope MCP endpoint (default below).
- ``PERISCOPE_AUTH_TOKEN`` - future Internet2/Periscope bearer token. The
                             server currently requires no credentials; leave
                             unset until Internet2 issues one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MCP_URL = "https://periscope.ns.internet2.edu/mcp"
DEFAULT_MODEL = "gpt-4o"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    """Load ``.env`` from the current directory, then the project root.

    Real environment variables always win; among ``.env`` files, the first one
    that defines a key wins.
    """
    load_dotenv()
    load_dotenv(_PROJECT_ROOT / ".env")


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def env_file_path() -> Path:
    """The project-root .env file (may not exist yet)."""
    return _PROJECT_ROOT / ".env"


def save_env_values(values: dict[str, str], path: Path | None = None) -> Path:
    """Write ``KEY=value`` pairs into ``.env``, updating lines that already set them."""
    path = path or env_file_path()
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(values)
    for index, line in enumerate(lines):
        stripped = line.strip()
        for key in list(remaining):
            if stripped.startswith(f"{key}="):
                lines[index] = f"{key}={remaining.pop(key)}"
                break
    lines.extend(f"{key}={value}" for key, value in remaining.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class Settings:
    mcp_url: str = DEFAULT_MCP_URL
    periscope_token: str | None = None
    model: str = DEFAULT_MODEL
    openai_base_url: str | None = None
    openai_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        load_env()
        return cls(
            mcp_url=_clean(os.environ.get("PERISCOPE_MCP_URL")) or DEFAULT_MCP_URL,
            periscope_token=_clean(os.environ.get("PERISCOPE_AUTH_TOKEN")),
            model=_clean(os.environ.get("I2A_MODEL")) or DEFAULT_MODEL,
            openai_base_url=_clean(os.environ.get("OPENAI_BASE_URL")),
            openai_api_key=_clean(os.environ.get("OPENAI_API_KEY")),
        )
