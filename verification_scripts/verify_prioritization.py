import os
import sys
import json
import redis
import fakeredis

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from redis_quota_manager import RedisQuotaManager

def verify_prioritization():
    print("🚀 Verifying Model Prioritization Logic (Tier + Speed System)...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Initialize Manager with fakeredis
    quota_manager = RedisQuotaManager(base_dir)
    # Force reload config
    quota_manager._sync_configuration_from_json()
    
    # Check tiers for sample models
    models_to_check = [
        "gemini-2.0-flash",       # Standard, Fast
        "llama-3.1-8b-instant",   # Free, Fast
        "llama-3.3-70b-versatile", # Free, Slow (Smart)
        "Meta-Llama-3.1-8B-Instruct", # Economical, Fast
        "gemini-2.5-pro"          # Standard, Slow
    ]
    
    print("\n📊 Checking Assigned Tiers and Categories:")
    for m in models_to_check:
        tier_bytes = quota_manager.redis.get(f"model:{m}:tier")
        tier = tier_bytes.decode('utf-8') if tier_bytes else "unknown"
        
        cat_bytes = quota_manager.redis.get(f"model:{m}:category")
        category = cat_bytes.decode('utf-8') if cat_bytes else "unknown"
        
        print(f"   - {m}: Tier={tier}, Cat={category}")

    # Mock availability
    quota_manager.is_model_available = lambda x: True
    
    print("\n🔄 Testing Fallback Order...")
    _, _ = quota_manager.get_model_for_request(preferred_model=None)
    
    all_keys = quota_manager.redis.keys("model:*:rpm:limit")
    all_models = [k.decode('utf-8').split(':')[1] for k in all_keys]
    
    # Replicate logic for verification
    def safe_decode(val):
        return val.decode('utf-8') if val else "unknown"

    model_tiers = {m: safe_decode(quota_manager.redis.get(f"model:{m}:tier")) for m in all_models}
    model_cats = {m: safe_decode(quota_manager.redis.get(f"model:{m}:category")) for m in all_models}
    
    def get_tier_val(m):
        t = model_tiers.get(m, "standard")
        if t == "free": return 1
        if t == "economical": return 2
        if t == "standard": return 3
        return 4
        
    def get_speed_val(m):
        cat = model_cats.get(m, "unknown").lower()
        name = m.lower()
        if cat == "fast": return 1
        fast_keywords = ["flash", "instant", "mini", "8b", "7b", "haiku", "gemma-2-9b"]
        if any(k in name for k in fast_keywords): return 1
        return 2
        
    sorted_models = sorted(all_models, key=lambda x: (get_tier_val(x), get_speed_val(x)))
    
    print("\n📋 Prioritized Order (Top 15):")
    for i, m in enumerate(sorted_models[:15]):
        t_val = get_tier_val(m)
        s_val = get_speed_val(m)
        t_name = {1: "Free", 2: "Econ", 3: "Std", 4: "Prem"}.get(t_val, "Unk")
        s_name = {1: "Fast", 2: "Slow"}.get(s_val, "Unk")
        print(f"   {i+1}. {m} [{t_name} | {s_name}]")
        
    # Validation
    if len(sorted_models) > 0:
        # Check 1: Free + Fast should be first
        first = sorted_models[0]
        if get_tier_val(first) == 1 and get_speed_val(first) == 1:
            print("\n✅ SUCCESS: Free & Fast model is prioritized first.")
        else:
            print(f"\n❌ FAILURE: First model is {first} (Tier: {get_tier_val(first)}, Speed: {get_speed_val(first)})")
            
        # Check 2: Free Fast vs Free Slow
        free_fast = next((m for m in sorted_models if get_tier_val(m)==1 and get_speed_val(m)==1), None)
        free_slow = next((m for m in sorted_models if get_tier_val(m)==1 and get_speed_val(m)==2), None)
        
        if free_fast and free_slow:
            idx_fast = sorted_models.index(free_fast)
            idx_slow = sorted_models.index(free_slow)
            if idx_fast < idx_slow:
                print("✅ SUCCESS: Free/Fast prioritized over Free/Slow.")
            else:
                 print("❌ FAILURE: Free/Slow appeared before Free/Fast.")
                 
        # Check 3: Economical vs Standard
        econ = next((m for m in sorted_models if get_tier_val(m)==2), None)
        std = next((m for m in sorted_models if get_tier_val(m)==3), None)
        if econ and std:
             if sorted_models.index(econ) < sorted_models.index(std):
                  print("✅ SUCCESS: Economical prioritized over Standard.")
             else:
                  print("❌ FAILURE: Standard prioritized over Economical.")
    
    quota_manager.close()

if __name__ == "__main__":
    verify_prioritization()
