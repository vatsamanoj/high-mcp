import sys
import types
from pathlib import Path

# Provide lightweight stubs for optional runtime deps not required by these unit tests.
sys.modules.setdefault("aiosqlite", types.ModuleType("aiosqlite"))

from ai_engine import AIEngine


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_extract_prompt_terms_removes_noise():
    engine = AIEngine(quota_manager=None, base_dir='.')
    terms = engine._extract_prompt_terms("Fix plugin for server.py and ui_server.py with superpowers")
    assert "plugin" in terms
    assert ("server.py" in terms) or ("server" in terms)
    assert "and" not in terms


def test_project_context_prioritizes_plugin_first_paths(tmp_path):
    _write(tmp_path / "server.py", "print('server')\n")
    _write(tmp_path / "ui_server.py", "print('ui')\n")
    _write(tmp_path / "component_manager.py", "print('cm')\n")
    _write(tmp_path / "plugins" / "my_feature.py", "def setup(mcp=None, app=None):\n    pass\n")
    _write(tmp_path / "components" / "legacy.py", "print('legacy')\n")
    _write(tmp_path / "README.md", "project readme\n")

    engine = AIEngine(quota_manager=None, base_dir=str(tmp_path))
    context = engine._get_project_context("improve plugin my_feature for server and ui")

    assert "Project Structure:" in context
    assert "File: plugins/my_feature.py" in context
    assert "File: server.py" in context
    assert "File: ui_server.py" in context


def test_project_context_respects_char_budget(tmp_path):
    _write(tmp_path / "server.py", "x\n" * 100)
    _write(tmp_path / "ui_server.py", "x\n" * 100)
    for i in range(20):
        _write(tmp_path / "plugins" / f"p_{i}.py", ("line\n" * 2000))

    engine = AIEngine(quota_manager=None, base_dir=str(tmp_path))
    engine.MAX_CONTEXT_CHARS = 3000
    context = engine._get_project_context("plugins")

    assert len(context) <= 3500
    assert "File Contents:" in context
