import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redis_quota_manager import RedisQuotaManager

def test_exhaustion():
    print("🚀 Testing API Exhaustion Notification...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    quota_manager = RedisQuotaManager(base_dir)
    
    model_name = "test-exhaust-model"
    
    # Reset any previous usage for this test
    print("♻️ Resetting usage counters...")
    quota_manager.redis.delete(f"quota:{model_name}:rpm:used")
    quota_manager.redis.delete(f"notify:{model_name}:rpm:exhausted")
    
    print(f"🔄 Simulating requests for {model_name} (Limit: 3 RPM)...")
    
    for i in range(1, 6):
        print(f"   Request {i}...")
        quota_manager.update_quota(model_name)
        time.sleep(0.5)
        
    print("\n✅ Exhaustion simulation complete.")
    print("   Check your mobile for a 'BLOCKED' notification about RPM exhaustion.")
    print("   Also check the console output above for any async notification logs.")

if __name__ == "__main__":
    test_exhaustion()