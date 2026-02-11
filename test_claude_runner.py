import os
from unittest.mock import patch

from claude_runner import (
    PLUGIN_MODE_SYSTEM_INSTRUCTION,
    build_claude_environment,
    resolve_claude_cli_command,
)


def test_build_claude_environment_prefers_explicit_inputs():
    with patch.dict(os.environ, {}, clear=True):
        env = build_claude_environment("sk-ant-api03-real", "https://proxy.example")
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-api03-real"
    assert env["ANTHROPIC_BASE_URL"] == "https://proxy.example"
    assert env["CLAUDE_BASE_URL"] == "https://proxy.example"


def test_build_claude_environment_uses_high_mcp_key_when_missing_api_key():
    with patch.dict(os.environ, {"HIGH_MCP_KEY": "sk-ant-api03-from-high"}, clear=True):
        env = build_claude_environment(None, None)
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-api03-from-high"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8004"


def test_build_claude_environment_injects_dummy_key_for_proxy_mode():
    with patch.dict(os.environ, {}, clear=True):
        env = build_claude_environment(None, None)
    assert env["ANTHROPIC_API_KEY"].startswith("sk-ant-api03-dummy-key")


def test_resolve_claude_cli_command_raises_without_npx():
    with patch("claude_runner.shutil.which", return_value=None):
        try:
            resolve_claude_cli_command()
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "Could not find 'npx'" in str(exc)


def test_resolve_claude_cli_command_uses_npx_path():
    with patch("claude_runner.shutil.which", side_effect=["/usr/bin/npx", None]):
        cmd = resolve_claude_cli_command()
    assert cmd[:3] == ["/usr/bin/npx", "-y", "@anthropic-ai/claude-code"]


def test_plugin_mode_instruction_mentions_both_runtimes():
    assert "server.py" in PLUGIN_MODE_SYSTEM_INSTRUCTION
    assert "ui_server.py" in PLUGIN_MODE_SYSTEM_INSTRUCTION
    assert "plugins/" in PLUGIN_MODE_SYSTEM_INSTRUCTION
