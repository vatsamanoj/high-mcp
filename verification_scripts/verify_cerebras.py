import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redis_quota_manager import RedisQuotaManager
from ai_engine import AIEngine

def test_cerebras():
    print("🚀 Initializing RedisQuotaManager...")
    # Initialize with persistence disabled for testing to avoid messing up main state
    # Actually, we want to test if it loads the file, so we need the base_dir to be correct.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    quota_manager = RedisQuotaManager(base_dir, persistence_enabled=False)
    
    # Wait a bit for async loading if any (though it seems synchronous in __init__)
    
    print("🔍 Checking available models...")
    models = quota_manager.get_all_models()
    cerebras_models = [m for m in models if "llama" in m["model"].lower() and "cerebras" in m.get("model", "") or m["model"] in ["llama3.1-8b", "llama-3.3-70b"]]
    
    # Simpler filter since we know what we just added and they are unique enough or we can check provider if exposed
    # But get_all_models doesn't return provider.
    # So let's just look for the names we expect
    cerebras_models = [m for m in models if m["model"] in ["llama3.1-8b", "llama-3.3-70b"]]
    
    if not cerebras_models:
        print("❌ No Cerebras models found in quota configuration!")
        return
        
    print(f"✅ Found {len(cerebras_models)} Cerebras models: {[m['model'] for m in cerebras_models]}")
    
    ai_engine = AIEngine(quota_manager)
    
    for model_info in cerebras_models:
        model_name = model_info["model"]
        print(f"\n🧪 Testing model: {model_name}...")
        
        try:
            start_time = time.time()
            response = ai_engine.generate_content(model_name, "Hello, what is 2+2? Answer briefly.")
            duration = time.time() - start_time
            
            if "Error" in response:
                print(f"❌ FAILED: {response}")
            else:
                print(f"✅ SUCCESS ({duration:.2f}s): {response.strip()}")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")

    # Helper to list models
    print("\n📋 Listing available models from Cerebras API...")
    import requests
    try:
        url = "https://api.cerebras.ai/v1/models"
        headers = {"Authorization": "Bearer csk-fxn2m5npywhp2vfnmerck4jyptncnwmpw6pc3wj4ptkhd96e"}
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            print("Available models:")
            for m in data.get('data', []):
                print(f" - {m['id']}")
        else:
            print(f"Failed to list models: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    test_cerebras()
