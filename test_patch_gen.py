import httpx
import asyncio
import json
import sys

async def test_stream():
    url = "http://localhost:8004/api/coder/generate_stream"
    payload = {"prompt": "Create a hello world python file named hello.py", "model": "gemini-2.0-flash-lite"}
    
    print(f"Connecting to {url}...")
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                print(f"Status: {response.status_code}")
                if response.status_code != 200:
                    print(await response.read())
                    return

                async for line in response.aiter_lines():
                    if not line: continue
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        try:
                            obj = json.loads(data)
                            if obj['type'] == 'log':
                                print(f"Log: {obj['message']}")
                                with open("test_output.txt", "a", encoding="utf-8") as f:
                                    f.write(f"Log: {obj['message']}\n")
                            elif obj['type'] == 'result':
                                print("Result received!")
                                with open("test_output.txt", "a", encoding="utf-8") as f:
                                    f.write(f"Result received: {json.dumps(obj['data'])}\n")
                                if 'error' in obj['data']:
                                    print(f"❌ Error: {obj['data']['error']}")
                                    if 'raw_response' in obj['data']:
                                        print(f"Raw Response Preview: {obj['data']['raw_response'][:200]}...")
                                else:
                                    print("✅ Success! Patches found:")
                                    print(json.dumps(obj['data'], indent=2))
                            elif obj['type'] == 'error':
                                print(f"❌ Stream Error: {obj['message']}")
                        except Exception as e:
                            print(f"Parse error: {e} for line: {line}")
            print("Stream finished.")
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_stream())
