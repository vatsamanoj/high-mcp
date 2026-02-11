import sys
import os
import json
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redis_quota_manager import RedisQuotaManager
from ai_engine import AIEngine

def test_blocking_system():
    print("🚀 Starting Blocked Account Verification...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    quota_manager = RedisQuotaManager(base_dir, persistence_enabled=False)
    
    # 1. Create a dummy quota with an INVALID key
    fake_config_path = os.path.join(base_dir, "quotas", "quota_fake_block.json")
    fake_config = {
        "provider": "google",
        "api_endpoint": "https://generativelanguage.googleapis.com/v1beta",
        "api_key": "AIzaSyFakeKeyForTestingBlockingSystem12345",
        "models": [
            {
                "model": "gemini-fake-block",
                "category": "fast",
                "rpm": { "limit": 10 }
            }
        ]
    }
    
    with open(fake_config_path, "w") as f:
        json.dump(fake_config, f)
        
    print(f"📝 Created fake config at {fake_config_path}")
    
    # Wait for hot-reload
    time.sleep(3)
    
    # Force sync just in case
    quota_manager._sync_configuration_from_json()
    
    # Verify model is loaded
    if not quota_manager.is_model_available("gemini-fake-block"):
        print("❌ Model failed to load or is already blocked!")
        return
        
    print("✅ Model loaded successfully. Attempting to use it...")
    
    ai_engine = AIEngine(quota_manager)
    
    # 2. Trigger the error
    response = ai_engine.generate_content("gemini-fake-block", "Hello")
    print(f"📡 Response: {response}")
    
    # 3. Check if blocked
    # Note: Google might return 400 for invalid key, so we might need to adjust our logic
    # if it doesn't trigger 401/403.
    
    is_blocked = not quota_manager.is_model_available("gemini-fake-block")
    print(f"🔒 Is model blocked now? {is_blocked}")
    
    # Check specific reason in Redis (accessing private redis for test)
    config_file = "quota_fake_block.json"
    blocked_reason = quota_manager.redis.get(f"config:{config_file}:blocked")
    if blocked_reason:
        print(f"📝 Blocked Reason: {blocked_reason.decode('utf-8')}")
    else:
        print("⚠️ No blocked reason found in Redis.")
        
    # 4. Check notification log
    log_file = os.path.join(base_dir, "notifications.log")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = f.read()
            if "BLOCKED" in logs and "gemini-fake-block" in logs:
                print("✅ Notification found in log file.")
            else:
                print("❌ No notification found in log file!")
    else:
        print("❌ Log file not found!")

    # Cleanup
    try:
        os.remove(fake_config_path)
        print("🧹 Cleaned up fake config.")
    except:
        pass

if __name__ == "__main__":
    test_blocking_system()
