import requests
import json
import sys

def test_claude_run():
    url = "http://localhost:8004/api/claude/run"
    # Provide a prompt that should trigger a tool use or at least a response
    payload = {
        "prompt": "List your available tools please.",
        "api_base": "http://localhost:8004", # Force local proxy
        # "api_key": "sk-ant-test-key" # Optional, if we want to override server key
    }
    
    print(f"Testing {url}...")
    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(response.text)
                return

            print("Connected. Reading stream...")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    print(f"RAW: {decoded_line}") # Debug raw output
                    
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        try:
                            data = json.loads(data_str)
                            if data['type'] == 'log':
                                # print(f"[LOG] {data['message']}")
                                pass
                            elif data['type'] == 'result':
                                print(f"[RESULT] {data['data']}")
                        except json.JSONDecodeError:
                            print(f"[JSON ERROR] {data_str}")
                    else:
                        print(f"RAW: {decoded_line}")
            
            print("Stream ended.")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_claude_run()