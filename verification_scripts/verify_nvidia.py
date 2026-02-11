import sys
import os
import time
import requests

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redis_quota_manager import RedisQuotaManager
from ai_engine import AIEngine

def test_nvidia():
    print("🚀 Initializing RedisQuotaManager...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    quota_manager = RedisQuotaManager(base_dir, persistence_enabled=False)
    
    print("🔍 Checking available Nvidia models from config...")
    # Wait for file watcher/sync if needed, but since we just wrote it, we might need to trigger sync
    # But we initialized RedisQuotaManager with persistence_enabled=False which loads everything on init.
    
    models = quota_manager.get_all_models()
    nvidia_models = [m for m in models if "moonshotai" in m["model"].lower()]
    
    if not nvidia_models:
        print("❌ No Nvidia/Kimi models found in quota configuration!")
        return

    # Prioritize k2-instruct for debugging
    nvidia_models.sort(key=lambda x: "instruct" in x["model"], reverse=True)
        
    print(f"✅ Found {len(nvidia_models)} Nvidia models: {[m['model'] for m in nvidia_models]}")
    
    ai_engine = AIEngine(quota_manager)
    
    # Sort so k2-instruct comes first (it might be more reliable?)
    # nvidia_models.sort(key=lambda x: len(x["model"])) 
    
    for model_info in nvidia_models:
        model_name = model_info["model"]
        print(f"\n🧪 Testing model: {model_name}...")
        
        try:
            start_time = time.time()
            # Kimi might be slow with "thinking" enabled
            print("   (This might take a moment if 'thinking' is enabled...)")
            response = ai_engine.generate_content(model_name, "Why is the sky blue? Answer briefly.")
            duration = time.time() - start_time
            
            if "Error" in response:
                print(f"❌ FAILED: {response}")
            else:
                print(f"✅ SUCCESS ({duration:.2f}s): {response.strip()[:200]}...") # Truncate output
                
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")

if __name__ == "__main__":
    test_nvidia()
