from pathlib import Path

from component_manager import ComponentManager


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_list_components_reports_needs_plug_in_before_load(tmp_path):
    _write(tmp_path / "plugins" / "demo_plugin.py", "def setup(mcp=None, app=None):\n    pass\n")

    manager = ComponentManager(str(tmp_path), trust_system=object())
    items = manager.list_components()

    demo = next(x for x in items if x["name"] == "demo_plugin")
    assert demo["source"] == "plugin"
    assert demo["attached_to_project"] is False
    assert demo["needs_plug_in"] is True


def test_list_components_reports_in_use_after_load(tmp_path):
    _write(tmp_path / "plugins" / "demo_plugin.py", "def setup(mcp=None, app=None):\n    pass\n")

    manager = ComponentManager(str(tmp_path), trust_system=object())
    assert manager.load_component("demo_plugin") is True

    items = manager.list_components()
    demo = next(x for x in items if x["name"] == "demo_plugin")
    assert demo["attached_to_project"] is True
    assert demo["in_use"] is True
    assert demo["needs_plug_in"] is False
