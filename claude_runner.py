import os
import shutil
from typing import Dict, Optional, List


PLUGIN_MODE_SYSTEM_INSTRUCTION = (
    "CRITICAL: You are operating in Runtime Plugin Mode for BOTH server.py (MCP) and ui_server.py (FastAPI). "
    "Do NOT edit existing files in the project root or 'components/' directly. "
    "Implement changes by creating/updating Python plugin files only in the 'plugins/' directory. "
    "Any new plugin must include tests and be validated before integration. "
    "Prefer using superpowers workflow: brainstorm -> plan -> tdd -> review -> debug."
)


def resolve_claude_cli_command() -> List[str]:
    """Resolve an executable command for Claude Code CLI."""
    npx_cmd = "npx.cmd" if os.name == "nt" else "npx"
    npx_path = shutil.which(npx_cmd) or shutil.which("npx")
    if not npx_path:
        raise RuntimeError("Could not find 'npx' in PATH. Install Node.js/npm to run Claude CLI.")
    return [npx_path, "-y", "@anthropic-ai/claude-code"]


def build_claude_environment(
    api_key: Optional[str],
    api_base: Optional[str],
    base_fallback_url: str = "http://127.0.0.1:8004",
) -> Dict[str, str]:
    """Build environment variables for Claude CLI execution."""
    env = os.environ.copy()

    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key
    elif "HIGH_MCP_KEY" in env and "ANTHROPIC_API_KEY" not in env:
        env["ANTHROPIC_API_KEY"] = env["HIGH_MCP_KEY"]

    base_url = api_base.strip() if api_base and api_base.strip() else base_fallback_url
    env["ANTHROPIC_BASE_URL"] = base_url
    env["CLAUDE_BASE_URL"] = base_url

    # Claude SDK validates key format before request dispatch.
    if "ANTHROPIC_API_KEY" not in env:
        env["ANTHROPIC_API_KEY"] = "sk-ant-api03-dummy-key-for-local-proxy-1234567890"

    return env
