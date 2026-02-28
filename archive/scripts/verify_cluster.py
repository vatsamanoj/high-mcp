import httpx
import sys

print("Starting verification...")
try:
    print("Checking Quota Node...")
    r1 = httpx.get("http://localhost:8003/status", timeout=5)
    print(f"Quota: {r1.status_code}")
except Exception as e:
    print(f"Quota Error: {e}")

try:
    print("Checking UI Node...")
    r2 = httpx.get("http://localhost:8004/dashboard", timeout=5)
    print(f"UI: {r2.status_code}")
except Exception as e:
    print(f"UI Error: {e}")

try:
    print("Fetching Models...")
    r_models = httpx.get("http://localhost:8004/api/chat/models", timeout=5)
    if r_models.status_code == 200:
        models = r_models.json().get("models", [])
        print(f"Models found: {len(models)}")
        if models:
            model_name = models[0].get("model")
            print(f"Testing Chat with model: {model_name}")
            payload = {
                "model": model_name,
                "message": "Hello, are you working?"
            }
            
            print("--- Request 1 ---")
            r_chat1 = httpx.post("http://localhost:8004/api/chat", json=payload, timeout=30)
            print(f"Response 1: {r_chat1.status_code}")
            
            print("--- Request 2 (Should be Cache Hit) ---")
            r_chat2 = httpx.post("http://localhost:8004/api/chat", json=payload, timeout=30)
            print(f"Response 2: {r_chat2.status_code}")
            
        else:
            print("No models available to test chat.")
    else:
        print(f"Failed to fetch models: {r_models.status_code}")
except Exception as e:
    print(f"Chat Test Error: {e}")
