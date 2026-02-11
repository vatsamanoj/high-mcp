import json
import httpx
import os
import sys
import time
import asyncio

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redis_quota_manager import RedisQuotaManager

# Configuration
# Point to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
qm = RedisQuotaManager(BASE_DIR)
TEST_PROMPT = "Hello, respond with 'OK' if you can hear me."

async def debug_model(model_name):
    with open("debug_log.txt", "a", encoding='utf-8') as f:
        f.write(f"\n--- DEBUGGING Model: {model_name} ---\n")
    
    config_file = qm.redis.get(f"model:{model_name}:config_file")
    if not config_file:
        print(f"❌ SKIPPED: No config file found for {model_name}")
        return

    config_file = config_file.decode('utf-8')
    endpoint = qm.redis.get(f"config:{config_file}:endpoint").decode('utf-8')
    key = qm.redis.get(f"config:{config_file}:key").decode('utf-8')
    
    base_url = endpoint.rstrip("/")
    if not base_url.endswith("models"):
        base_url += "/models"

    with open("debug_log.txt", "a", encoding='utf-8') as f:
        f.write(f"Endpoint: {base_url}\n")
        f.write(f"API Key: {key[:5]}...{key[-5:]}\n")

    # 1. Try to find the model in the list
    resolved_id = model_name
    print("Fetching model list...")
    try:
        async with httpx.AsyncClient() as client:
            list_resp = await client.get(f"{base_url}?key={key}")
            if list_resp.status_code == 200:
                models_data = list_resp.json()
                found = False
                for m in models_data.get("models", []):
                    name = m.get("name")
                    display = m.get("displayName", "")
                    if model_name.lower() in name.lower() or model_name.lower() in display.lower():
                        with open("debug_log.txt", "a", encoding='utf-8') as f:
                            f.write(f"Found candidate: Name='{name}', Display='{display}'\n")
                        if name.endswith(model_name):
                             resolved_id = name.replace("models/", "")
                             found = True
                             # break # Don't break, see all candidates
                if not found:
                    with open("debug_log.txt", "a", encoding='utf-8') as f:
                        f.write("No exact match found in model list.\n")
            else:
                 with open("debug_log.txt", "a", encoding='utf-8') as f:
                    f.write(f"Failed to list models: {list_resp.status_code} {list_resp.text}\n")
    except Exception as e:
         with open("debug_log.txt", "a", encoding='utf-8') as f:
            f.write(f"Error listing models: {e}\n")

    # 2. Try generation
    with open("debug_log.txt", "a", encoding='utf-8') as f:
        f.write(f"Testing generation with ID: {resolved_id}\n")
    
    url = f"{base_url}/{resolved_id}:generateContent?key={key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": TEST_PROMPT}]}]}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            
        with open("debug_log.txt", "a", encoding='utf-8') as f:
            f.write(f"Status: {response.status_code}\n")
            f.write(f"Headers: {dict(response.headers)}\n")
            f.write(f"Body: {response.text}\n")
        
    except Exception as e:
         with open("debug_log.txt", "a", encoding='utf-8') as f:
            f.write(f"Generation Error: {e}\n")

async def main():
    target_models = ["gemini-2.0-flash-lite", "gemini-2.5-pro"]
    for m in target_models:
        await debug_model(m)
    qm.close()

if __name__ == "__main__":
    asyncio.run(main())
