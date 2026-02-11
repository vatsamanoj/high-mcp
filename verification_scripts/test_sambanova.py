import sys
import os
import json
import httpx
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redis_quota_manager import RedisQuotaManager

def test_sambanova_integration():
    print("🚀 Testing SambaNova Integration (All Models)...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    quota_manager = RedisQuotaManager(base_dir)
    
    # Force sync to ensure we have the latest from files
    quota_manager._sync_configuration_from_json()
    
    config_filename = "quota_sambanova.json"
    
    # Retrieve config from Redis
    api_key_bytes = quota_manager.redis.get(f"config:{config_filename}:key")
    api_endpoint_bytes = quota_manager.redis.get(f"config:{config_filename}:endpoint")
    
    if not api_key_bytes or not api_endpoint_bytes:
        print(f"❌ SambaNova configuration ({config_filename}) not found in Redis!")
        return
        
    api_key = api_key_bytes.decode('utf-8')
    api_endpoint = api_endpoint_bytes.decode('utf-8')
    
    # Mask key for display
    masked_key = f"{api_key[:4]}...{api_key[-4:]}"
    print(f"✅ Configuration loaded for {config_filename}")
    print(f"   Endpoint: {api_endpoint}")
    print(f"   API Key: {masked_key}")
    
    # Load models from the file directly to iterate (Redis stores them individually)
    config_path = os.path.join(base_dir, "quotas", config_filename)
    with open(config_path, 'r') as f:
        config_data = json.load(f)
        
    models = [m["model"] for m in config_data.get("models", [])]
    print(f"📋 Found {len(models)} models to test: {', '.join(models)}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    results = []
    
    for model in models:
        print(f"\n🧪 Testing model: {model}...")
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Say 'Hello from {model}' if you can hear me."}
            ],
            "max_tokens": 50
        }
        
        try:
            url = f"{api_endpoint}/chat/completions"
            # print(f"   Sending request to {url}...")
            
            start_time = time.time()
            response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"✅ Success! ({duration:.2f}s)")
                print(f"   Response: {content.strip()}")
                results.append({"model": model, "status": "✅ Success", "time": f"{duration:.2f}s"})
            else:
                print(f"❌ Request failed with status code: {response.status_code}")
                print(f"   Response: {response.text}")
                results.append({"model": model, "status": f"❌ Failed ({response.status_code})", "time": f"{duration:.2f}s"})
                
        except Exception as e:
            print(f"❌ Error during request: {e}")
            results.append({"model": model, "status": f"❌ Error: {str(e)[:50]}...", "time": "0s"})
            
        # Small delay to avoid rate limits if testing many
        time.sleep(1)
            
    print("\n" + "="*50)
    print("📊 Test Summary:")
    print("="*50)
    for res in results:
        print(f"{res['status']} - {res['model']} ({res['time']})")
    print("="*50)

if __name__ == "__main__":
    test_sambanova_integration()