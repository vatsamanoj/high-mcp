from pathlib import Path

from fastapi import FastAPI

from component_manager import ComponentManager


class DummyMCP:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator


def _write_demo_plugin(base_dir: Path):
    plugins_dir = base_dir / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    src = Path("plugins/claude_superpowers_demo.py").read_text()
    (plugins_dir / "claude_superpowers_demo.py").write_text(src)


def test_demo_plugin_loads_in_ui_runtime(tmp_path):
    _write_demo_plugin(tmp_path)
    app = FastAPI()
    manager = ComponentManager(str(tmp_path), trust_system=object(), fastapi_app=app)

    ok = manager.load_component("claude_superpowers_demo")

    assert ok is True
    paths = {route.path for route in app.routes}
    assert "/api/plugins/claude-superpowers-demo/health" in paths
    assert "/api/plugins/claude-superpowers-demo/prepare" in paths


def test_demo_plugin_loads_in_mcp_runtime(tmp_path):
    _write_demo_plugin(tmp_path)
    dummy_mcp = DummyMCP()
    manager = ComponentManager(str(tmp_path), trust_system=object(), mcp_server=dummy_mcp)

    ok = manager.load_component("claude_superpowers_demo")

    assert ok is True
    assert "prepare_claude_superpower_task" in dummy_mcp.tools
