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

# 1. Initialize Trust System
print("\n🛡️  Initializing Trust System...")
trust_system = TrustSystem(BASE_DIR)
# Create a startup snapshot
trust_system.create_snapshot("startup")

# 2. Initialize FastMCP Server
mcp = FastMCP("Quota Server")

# 3. Initialize Core Services
# These are essential for the server to function, so they are not dynamic "components" in the same sense,
# but they provide the foundation for components.
print("⚙️  Initializing Core Services...")
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
print("🧩 Initializing Component Manager...")
# Note: In this MCP Node, we don't have the FastAPI app (it's in UI Node).
# So we pass mcp_server=mcp, fastapi_app=None
component_manager = ComponentManager(BASE_DIR, trust_system, mcp_server=mcp)

# 5. Load Components (Core AI + Plugins)
print("📥 Loading Components...")
component_manager.load_all_components()
component_manager.start_watcher()

# 6. Start Background Services
print("📡 Starting Background Sensors & UI...")
sensor = FreeAISensor(quota_manager)
sensor.start()

# Start UI Server as a separate subprocess
print("🖥️  Launching UI Node (subprocess)...")
ui_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "ui_server:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "1"])

print("✅ Server Bootstrapped Successfully.")

if __name__ == "__main__":
    try:
        # Run MCP (This blocks in stdio mode usually, but fastmcp might handle it differently)
        # We'll run it directly.
        print("🔌 MCP Server Running (STDIO)...")
        # If mcp.run() returns immediately (e.g. no client), we should keep the process alive for the UI.
        # We can't easily run mcp.run() and ui_process.wait() in parallel without threading mcp.run().
        # But for now, let's assume if mcp.run() returns, we just wait for UI.
        try:
            mcp.run()
        except Exception as e:
            print(f"MCP Run Error/Exit: {e}")
        
        if ui_process:
            print("⏳ MCP Service exited/skipped. keeping UI alive...")
            ui_process.wait()
            
    except KeyboardInterrupt:
        print("🛑 Server Stopped.")
        if ui_process:
            ui_process.terminate()
    except Exception as e:
        print(f"❌ CRITICAL SERVER FAILURE: {e}")
        error_manager.log_error(e, "Main Server Process")
        if ui_process:
            ui_process.terminate()
