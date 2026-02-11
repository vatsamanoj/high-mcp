import subprocess
import sys
import time
import os
import signal
import psutil
import httpx

def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port:
                    print(f"Killing process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
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
    print("🚀 Starting High-MCP Multi-Node Cluster...")

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
        print("❌ Failed to start Quota Server!")
        quota_process.kill()
        sys.exit(1)
    print("✅ Quota Server is active.")

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

    print("\n✅ Cluster is RUNNING!")
    print("   - Quota Node: http://localhost:8003")
    print("   - UI Node:    http://localhost:8004/dashboard")
    print("\nPress Ctrl+C to stop the cluster.")

    try:
        ui_process.wait()
        quota_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping cluster...")
        ui_process.terminate()
        quota_process.terminate()
        print("Done.")

if __name__ == "__main__":
    main()
