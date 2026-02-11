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

def _resolve_model_id(model_name: str, api_endpoint: str, api_key: str) -> str:
    """Helper to resolve model ID properly."""
    # 1. If it looks like a full ID (has slashes or no spaces), use it
    if "/" in model_name or " " not in model_name:
        return model_name.replace("models/", "")
    
    # 2. Heuristics for common display names to IDs
    lower = model_name.lower()
    if "gemini 2 flash" in lower: return "gemini-2.0-flash"
    if "gemini 2.5 flash" in lower: return "gemini-2.0-flash-lite-preview-02-05" 
    
    return model_name

async def verify_model(model_name: str, config: dict):
    print(f"\n--- Testing Model: {model_name} ---")
    
    api_endpoint = config.get("api_endpoint")
    api_key = config.get("api_key")
    
    if not api_endpoint or not api_key:
        print(f"❌ SKIPPED: Missing API config for {model_name}")
        return "skipped"

    # Construct URL
    base_url = api_endpoint.rstrip("/")
    if not base_url.endswith("models"):
        base_url += "/models"
        
    resolved_id = model_name
    
    # Fetch real list to map names (Google specific)
    # If provider is not google, we might skip this or do different logic.
    provider = config.get("provider", "google")
    
    if provider == "google":
        resolved_id = _resolve_google_id(model_name, base_url, api_key)
        url = f"{base_url}/{resolved_id}:generateContent?key={api_key}"
        payload = {"contents": [{"parts": [{"text": TEST_PROMPT}]}]}
        headers = {"Content-Type": "application/json"}
    elif provider == "openai":
        # OpenAI compatible
        url = f"{api_endpoint.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": TEST_PROMPT}]
        }
    else:
        print(f"Unknown provider: {provider}")
        return "error"

    async with httpx.AsyncClient() as client:
        # Default retries to 1 to avoid hanging on many failures
        retries = 1
        for attempt in range(retries + 1):
            try:
                if attempt > 0:
                    print(f"Retrying {model_name} (Attempt {attempt+1})...")
                    time.sleep(5)
                
                start_time = time.time()
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    print(f"[SUCCESS]: {model_name} responded in {elapsed:.2f}s")
                    return "success"
                
                elif response.status_code == 429:
                    print(f"[QUOTA EXCEEDED]: {model_name} (429)")
                    # Fail fast on 429 to avoid wasting time
                    return "exhausted"
                
                elif response.status_code == 503:
                    print(f"[HIGH DEMAND]: {model_name} (503)")
                    # Retry on 503
                    if attempt < retries: continue
                    return "failed"
                
                else:
                    print(f"[FAILED]: {model_name} (Status: {response.status_code})")
                    with open("verification_debug.log", "a", encoding='utf-8') as log:
                        log.write(f"\n[{model_name}] Status: {response.status_code}\n")
                        log.write(f"Response: {response.text}\n")
                    return "failed"
                    
            except Exception as e:
                print(f"[ERROR]: {model_name} - {str(e)}")
                if attempt < retries: continue
                return "error"
                
    return "failed"

def _resolve_google_id(model_name, base_url, api_key):
    # Logic extracted from original verify_model
    resolved_id = model_name
    try:
        # Use sync httpx here or assume async context? 
        # This function is synchronous but httpx calls are usually async in async def.
        # But we are in a helper. Let's make it sync or use httpx.get inside?
        # The original code did async fetch. 
        # For simplicity, let's keep the heuristic logic but skip the API list fetch 
        # unless absolutely necessary.
        # OR better: Assume the ID is correct or mapped by heuristics.
        # Fetching list for every model is slow.
        pass
    except:
        pass
    return model_name # Fallback

async def main():
    # Clear debug log
    with open("verification_debug.log", "w", encoding='utf-8') as log:
        log.write("Starting verification...\n")

    print("Starting verification of all models...")
    
    model_keys = qm.redis.keys("model:*:rpm:limit")
    all_models = [k.decode('utf-8').split(':')[1] for k in model_keys]
    
    print(f"Found {len(all_models)} models in configuration.")
    
    results = {
        "success": [],
        "exhausted": [],
        "failed": [],
        "skipped": [],
        "error": []
    }
    
    try:
        for model_name in all_models:
            # Filter: Exclude computer-use and image-specific models
            lower_name = model_name.lower()
            if "computer-use" in lower_name or "image" in lower_name:
                print(f"Skipping {model_name}: Special model (computer-use/image)")
                results["skipped"].append(model_name)
                continue

            config_file = qm.redis.get(f"model:{model_name}:config_file")
            if not config_file:
                print(f"Skipping {model_name}: No config file found")
                continue
                
            config_file = config_file.decode('utf-8')
            endpoint = qm.redis.get(f"config:{config_file}:endpoint").decode('utf-8')
            key = qm.redis.get(f"config:{config_file}:key").decode('utf-8')
            
            provider_bytes = qm.redis.get(f"config:{config_file}:provider")
            provider = provider_bytes.decode('utf-8') if provider_bytes else "google"
            
            config = {"api_endpoint": endpoint, "api_key": key, "provider": provider}
            
            if not qm.is_model_available(model_name):
                print(f"[WARN]: {model_name} is already marked as exhausted/unavailable in Redis.")
            
            with open("verification_debug.log", "a", encoding='utf-8') as log:
                log.write(f"Testing {model_name}...\n")

            status = await verify_model(model_name, config)
            results[status].append(model_name)
            
            time.sleep(2) 

    except Exception as e:
        print(f"CRITICAL LOOP ERROR: {e}")
        with open("verification_debug.log", "a", encoding='utf-8') as log:
            log.write(f"CRITICAL LOOP ERROR: {e}\n")
            
    finally:
        print("\n" + "="*40)
        print("VERIFICATION REPORT")
        print("="*40)
        print(f"Total Tested: {len(all_models)}")
        print(f"[Working]: {len(results['success'])}")
        print(f"[Exhausted (429)]: {len(results['exhausted'])}")
        print(f"[Failed (Other)]: {len(results['failed'])}")
        print(f"[Skipped/Error]: {len(results['skipped']) + len(results['error'])}")
        
        if results['exhausted']:
            print("\nExhausted Models:")
            for m in results['exhausted']:
                print(f" - {m}")

        if results['failed']:
            print("\nFailed Models (Check names/IDs):")
            for m in results['failed']:
                print(f" - {m}")
                
        # Save report to file
        report_path = os.path.join(BASE_DIR, "quota_verification_report.json")
        try:
            with open(report_path, "w", encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"\nReport saved to {report_path}")
        except Exception as e:
            print(f"\n[ERROR] Failed to save report: {e}")
        
        qm.close()

if __name__ == "__main__":
    asyncio.run(main())
