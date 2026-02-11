# Architectural Refactor: Unified Component System & AI Integration

## Context
The project `high-mcp` consists of two main servers:
1. **`server.py`**: The main MCP (Model Context Protocol) server using `FastMCP`.
2. **`ui_server.py`**: A FastAPI-based UI and API server (serving `dashboard.html` and AI Coder endpoints).

Currently, `ComponentManager` exists but only supports `mcp_server`. `server.py` is broken due to a missing `start_ui_server` import.

## Objective
Refactor the system to support a **Unified Component/Plugin Architecture** where:
1.  **Both** servers (MCP and UI) can load and serve components.
2.  Components can inject **MCP Tools/Resources** AND **FastAPI Routes/Endpoints**.
3.  The **AI Coder** is aware of this architecture and can generate valid, hot-reloadable components.

## Technical Requirements

### 1. `ComponentManager` Upgrade
Modify `component_manager.py` to:
- Accept both `mcp_server` (optional) and `fastapi_app` (optional) in `__init__`.
- In `load_component(name)`, dynamically import the module.
- Look for a `setup(mcp, app)` function in the module.
- Call `setup` passing the available instances. Handle cases where one is `None` (gracefully skip or warn).
- Implement `reload_component(name)` to support hot-swapping (re-import module, re-run setup). *Note: For FastAPI, replacing routes is complex; for now, appending new routes or updating existing router references is acceptable.*

### 2. `ui_server.py` Integration
- Remove the dependency on `server.py` to start it (it runs independently or via subprocess).
- Instantiate `ComponentManager` in `lifespan` or startup event, passing the `app` instance.
- Load all components on startup.
- Add an endpoint `POST /api/components/reload` to trigger `component_manager.reload_component()`.

### 3. `server.py` Fix & Integration
- Fix the `ImportError: cannot import name 'start_ui_server'`.
- Instead of importing `start_ui_server`, use `subprocess.Popen` to launch `ui_server.py` as a separate process (or keep it in-process if preferred, but ensure `app` is accessible).
- Instantiate `ComponentManager` passing the `mcp` instance.
- Load components on startup.

### 4. Component Standard Interface
Define the standard structure for a component (`components/example_plugin.py`):
```python
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI

def setup(mcp: FastMCP = None, app: FastAPI = None):
    """
    Setup function called by ComponentManager.
    """
    if mcp:
        @mcp.tool()
        def example_tool(x: int) -> int:
            return x * 2

    if app:
        @app.get("/api/example")
        def example_route():
            return {"message": "Hello from Plugin!"}
```

### 5. AI Coder System Prompt Update
Update the `system_prompt` in `ai_engine.py` to include instructions on how to generate these components. The AI should know:
- Components go in `components/`.
- They must have a `setup(mcp, app)` function.
- They can use `mcp.tool()` and `app.get/post/etc`.

## Task Checklist
- [ ] Fix `server.py` imports and startup logic.
- [ ] Update `component_manager.py` logic.
- [ ] Integrate `ComponentManager` into `ui_server.py`.
- [ ] Create `components/demo_component.py` to verify the architecture.
- [ ] Update `ai_engine.py` system prompt.
