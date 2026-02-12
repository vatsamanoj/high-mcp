import subprocess
import sys
import time
import os
import signal
import psutil
import httpx


def run_plugin_audit():
    """
    Safe startup audit:
    - Runs by default
    - Never blocks startup unless PLUGIN_AUDIT_STRICT=1
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
        completed = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True, timeout=30)
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
        print(f"WARNING: Plugin Audit failed to run safely: {e}. Continuing startup.")

def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            pid = int(proc.info.get('pid') or 0)
            if pid <= 0:
                continue
            for conn in proc.net_connections(kind='inet'):
                if conn.laddr.port == port:
                    print(f"Killing process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, ValueError):
            pass

def wait_for_server(url, timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False

def main():
    print("Starting High-MCP Multi-Node Cluster...")
    run_plugin_audit()

    # 1. Kill existing processes on ports 8002, 8003, 8004
    kill_process_on_port(8002)
    kill_process_on_port(8003)
    kill_process_on_port(8004)
    
    # Wait for ports to be released
    time.sleep(2)

    # 1.5 Initialize Database (Single Process to avoid locks)
    print("Initializing Database...")
    subprocess.run([sys.executable, "init_db.py"], check=True)

    # 2. Start Quota/State Server (Node 1)
    print("Starting Quota Server (Node 1)...")
    quota_server_cmd = [sys.executable, "quota_server.py"]
    quota_process = subprocess.Popen(quota_server_cmd, cwd=os.getcwd())
    
    if not wait_for_server("http://localhost:8003/status"):
        print("Failed to start Quota Server!")
        quota_process.kill()
        sys.exit(1)
    print("Quota Server is active.")

    # 3. Start UI Server (Node 2) with Workers
    # Windows doesn't support uvicorn workers via command line easily if code is not importable, 
    # but ui_server is a module.
    # 3. Start UI Server (Node 2) with Workers
    # On Windows, too many workers might cause socket/lock issues. 
    # Reducing to 1 for stability if needed, but 4 is target.
    # If "hanged", try 1.
    print("Starting UI Server Cluster (Node 2)...")
    ui_cmd = [
        sys.executable, "-m", "uvicorn", 
        "ui_server:app", 
        "--host", "0.0.0.0", 
        "--port", "8004", 
        "--workers", "1" 
    ]
    # Note: On Windows, --workers requires the app to be importable.
    
    ui_process = subprocess.Popen(ui_cmd, cwd=os.getcwd())

    print("\nCluster is RUNNING!")
    print("   - Quota Node: http://localhost:8003")
    print("   - UI Node:    http://localhost:8004/dashboard")
    print("\nPress Ctrl+C to stop the cluster.")

    try:
        ui_process.wait()
        quota_process.wait()
    except KeyboardInterrupt:
        print("\nStopping cluster...")
        ui_process.terminate()
        quota_process.terminate()
        print("Done.")

if __name__ == "__main__":
    main()
