import sys
import os
import time
import requests

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redis_quota_manager import RedisQuotaManager
from ai_engine import AIEngine

def test_groq():
    print("🚀 Initializing RedisQuotaManager...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    quota_manager = RedisQuotaManager(base_dir, persistence_enabled=False)
    
    print("🔍 Checking available Groq models from config...")
    models = quota_manager.get_all_models()
    # Filter for models that are likely Groq (based on the names in quota_groq.json)
    # Since we can't filter by provider easily in get_all_models without extra calls, 
    # we'll look for known Groq model names or models defined in quota_groq.json
    
    # We'll use the API to list actual models first, which is better.
    print("\n📋 Listing available models from Groq API...")
    groq_api_key = os.environ.get("GROQ_API_KEY", "gsk_PLACEHOLDER")
    try:
        url = "https://api.groq.com/openai/v1/models"
        headers = {"Authorization": f"Bearer {groq_api_key}"}
        resp = requests.get(url, headers=headers)
        
        available_groq_models = []
        if resp.status_code == 200:
            data = resp.json()
            print("Available models on Groq:")
            for m in data.get('data', []):
                print(f" - {m['id']}")
                available_groq_models.append(m['id'])
        else:
            print(f"Failed to list models: {resp.status_code} {resp.text}")
            
    except Exception as e:
        print(f"Error listing models: {e}")
        available_groq_models = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]

    # Now let's test a few of them
    ai_engine = AIEngine(quota_manager)
    
    # Test models that are both in our config AND in the API
    # Or just test what's in our config to see if it works
    
    test_candidates = [m for m in available_groq_models if any(k in m for k in ["llama-3", "qwen", "gemma"])]
    if not test_candidates:
        test_candidates = ["llama-3.1-8b-instant"] 

    print(f"\n🧪 Will test up to 3 models: {test_candidates[:3]}")

    for model_name in test_candidates[:3]:
        print(f"\n🧪 Testing model: {model_name}...")
        try:
            # Check if configured
            if not quota_manager.is_model_available(model_name):
                 print(f"⚠️ Model {model_name} not in current local config (or quota full), skipping.")
                 # But wait, we want to test connectivity even if not in config?
                 # No, AIEngine needs config.
                 # Let's try to force it for the test if it's not there, or just rely on the ones we added.
                 pass

            start_time = time.time()
            response = ai_engine.generate_content(model_name, "Hello, what is 2+2? Answer briefly.")
            duration = time.time() - start_time
            
            if "Error" in response:
                print(f"❌ FAILED: {response}")
            else:
                print(f"✅ SUCCESS ({duration:.2f}s): {response.strip()}")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")

if __name__ == "__main__":
    test_groq()
