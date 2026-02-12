import os
import sys
import time
import subprocess
from mcp.server.fastmcp import FastMCP
from redis_quota_manager import RedisQuotaManager
from async_adapters import LocalQuotaManagerAsync
from dependencies import set_dependencies
from free_ai_sensor import FreeAISensor
from ai_engine import AIEngine
from error_manager import ErrorManager
from trust_system import TrustSystem
from component_manager import ComponentManager

# --- Trust Server Bootstrapper ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def run_plugin_audit() -> None:
    """
    Safe startup audit:
    - warn-only by default
    - strict fail only with PLUGIN_AUDIT_STRICT=1
    """
    enabled = os.environ.get("PLUGIN_AUDIT_ON_STARTUP", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        print("Plugin Audit: skipped (PLUGIN_AUDIT_ON_STARTUP disabled).")
        return

    strict = os.environ.get("PLUGIN_AUDIT_STRICT", "0").strip().lower() in {"1", "true", "yes", "on"}
    cmd = [sys.executable, os.path.join("verification_scripts", "audit_plugin_architecture.py")]
    if strict:
        cmd.append("--strict")
    print(f"Plugin Audit: running ({'strict' if strict else 'warn-only'})...")
    try:
        completed = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=30)
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        if stdout:
            print(stdout)
        if stderr:
            print(f"Plugin Audit STDERR: {stderr}")
        if completed.returncode != 0:
            msg = f"Plugin Audit found issues (exit={completed.returncode})."
            if strict:
                raise RuntimeError(msg)
            print(f"WARNING: {msg} Continuing startup.")
        else:
            print("Plugin Audit: clean.")
    except Exception as e:
        if strict:
            raise
        print(f"WARNING: Plugin Audit failed safely: {e}. Continuing startup.")

run_plugin_audit()

# 1. Initialize Trust System
print("\nðŸ›¡ï¸  Initializing Trust System...")
trust_system = TrustSystem(BASE_DIR)
# Create a startup snapshot
trust_system.create_snapshot("startup")

# 2. Initialize FastMCP Server
mcp = FastMCP("Quota Server")

# 3. Initialize Core Services
# These are essential for the server to function, so they are not dynamic "components" in the same sense,
# but they provide the foundation for components.
print("âš™ï¸  Initializing Core Services...")
# Use Async Adapter for Quota Manager so AIEngine can await it
quota_manager = LocalQuotaManagerAsync(BASE_DIR)
ai_engine = AIEngine(quota_manager)
error_manager = ErrorManager(BASE_DIR, ai_engine)

# Inject Dependencies for Components
set_dependencies(
    error_manager=error_manager,
    quota_manager=quota_manager,
    ai_engine=ai_engine,
    trust_system=trust_system
)

# 4. Initialize Component Manager
print("ðŸ§© Initializing Component Manager...")
# Note: In this MCP Node, we don't have the FastAPI app (it's in UI Node).
# So we pass mcp_server=mcp, fastapi_app=None
component_manager = ComponentManager(BASE_DIR, trust_system, mcp_server=mcp)

# 5. Load Components (Core AI + Plugins)
print("ðŸ“¥ Loading Components...")
component_manager.load_all_components()
component_manager.start_watcher()

# 6. Start Background Services
print("ðŸ“¡ Starting Background Sensors & UI...")
sensor = FreeAISensor(quota_manager)
sensor.start()

# Start UI Server as a separate subprocess
print("ðŸ–¥ï¸  Launching UI Node (subprocess)...")
ui_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "ui_server:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "1"])

print("âœ… Server Bootstrapped Successfully.")

if __name__ == "__main__":
    try:
        # Run MCP (This blocks in stdio mode usually, but fastmcp might handle it differently)
        # We'll run it directly.
        print("ðŸ”Œ MCP Server Running (STDIO)...")
        # If mcp.run() returns immediately (e.g. no client), we should keep the process alive for the UI.
        # We can't easily run mcp.run() and ui_process.wait() in parallel without threading mcp.run().
        # But for now, let's assume if mcp.run() returns, we just wait for UI.
        try:
            mcp.run()
        except Exception as e:
            print(f"MCP Run Error/Exit: {e}")
        
        if ui_process:
            print("â³ MCP Service exited/skipped. keeping UI alive...")
            ui_process.wait()
            
    except KeyboardInterrupt:
        print("ðŸ›‘ Server Stopped.")
        if ui_process:
            ui_process.terminate()
    except Exception as e:
        print(f"âŒ CRITICAL SERVER FAILURE: {e}")
        error_manager.log_error(e, "Main Server Process")
        if ui_process:
            ui_process.terminate()

