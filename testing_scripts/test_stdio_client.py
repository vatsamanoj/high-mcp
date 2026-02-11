import subprocess
import json
import sys
import os
import time

def test_stdio_server():
    print("Starting server process...")
    # Point to server.py in the project root
    server_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server.py")
    
    process = subprocess.Popen(
        [sys.executable, server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    def send_request(req):
        print(f"Sending: {json.dumps(req)}")
        process.stdin.write(json.dumps(req) + "\n")
        process.stdin.flush()

    def read_response(expect_id=None):
        start_time = time.time()
        while time.time() - start_time < 5:
            line = process.stdout.readline()
            if not line:
                break
            
            # print(f"Raw: {line.strip()}")
            try:
                data = json.loads(line)
                if expect_id is not None and data.get("id") == expect_id:
                    return data
                if expect_id is None and "method" in data: # Notification or request from server
                    pass
            except json.JSONDecodeError:
                pass
        return None

    try:
        # 1. Initialize
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"}
            }
        }
        send_request(init_req)
        resp = read_response(expect_id=1)
        if resp:
            print("Initialize response received.")
        else:
            print("FAIL: No initialize response.")
            return

        # 2. Initialized notification
        send_request({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        })

        # 3. List Tools
        list_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        send_request(list_req)
        resp = read_response(expect_id=2)
        
        if resp and "result" in resp:
            print("SUCCESS: Received tools list.")
            tools = resp["result"].get("tools", [])
            tool_names = [t["name"] for t in tools]
            print(f"Available tools: {tool_names}")
            
            if "generate_content" in tool_names:
                print("PASS: generate_content tool found.")
            else:
                print("FAIL: generate_content tool NOT found.")
        else:
            print(f"FAIL: Invalid response: {resp}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        process.terminate()
        try:
            outs, errs = process.communicate(timeout=1)
            # if errs: print(f"Stderr: {errs}")
        except:
            pass

if __name__ == "__main__":
    test_stdio_server()
