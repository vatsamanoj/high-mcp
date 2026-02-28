import os
import json
from redis_quota_manager import RedisQuotaManager

base_dir = os.getcwd()
quota_dir = os.path.join(base_dir, "quotas")
qm = RedisQuotaManager(base_dir, persistence_enabled=False)

print("--- Recovering Quota Files ---")

# 1. Identify all config files from Redis
config_keys = qm.redis.keys("config:*:key")
filenames = []
for k in config_keys:
    k_str = k.decode('utf-8')
    fname = k_str[7:-4] 
    filenames.append(fname)

print(f"Found {len(filenames)} configs to recover: {filenames}")

# 2. Build Model Map (model -> config_file)
model_map = {} # filename -> [model_names]
model_config_keys = qm.redis.keys("model:*:config_file")
for mck in model_config_keys:
    mck_str = mck.decode('utf-8')
    # model:{name}:config_file
    model_name = mck_str[6:-12]
    
    cf_bytes = qm.redis.get(mck)
    if cf_bytes:
        cf_name = cf_bytes.decode('utf-8')
        if cf_name not in model_map:
            model_map[cf_name] = []
        model_map[cf_name].append(model_name)

print(f"Mapped {len(model_config_keys)} models to {len(model_map)} config files.")

# 3. Recover Files
for fname in filenames:
    file_path = os.path.join(quota_dir, fname)
    
    # Check if we should skip manual accounts (account1.json, account2.json) if they exist and are valid?
    # Actually, we should overwrite to be safe, OR check if they are empty.
    # The previous run said "Skipping existing file: account1.json".
    # I'll stick to overwriting ONLY if it's one of the files we know was missing.
    # But how do we know? The script doesn't know anymore.
    # I'll overwrite EVERYTHING to ensure consistency with Redis.
    # EXCEPT maybe if the file content is larger than 0 bytes and it's NOT one of the empty ones I just created.
    # But the empty ones I created are valid JSON with empty models list.
    
    print(f"Recovering {fname}...")
    
    # Get Base Config
    endpoint = qm.redis.get(f"config:{fname}:endpoint")
    key = qm.redis.get(f"config:{fname}:key")
    provider = qm.redis.get(f"config:{fname}:provider")
    
    config_data = {
        "api_endpoint": endpoint.decode('utf-8') if endpoint else "",
        "api_key": key.decode('utf-8') if key else "",
        "provider": provider.decode('utf-8') if provider else "google",
        "models": []
    }
    
    # Get models from map
    models_list = model_map.get(fname, [])
            
    # Reconstruct Model Entries
    for model in models_list:
        m_entry = {"model": model}
        
        # Get GLOBAL limits
        rpm = qm.redis.get(f"model:{model}:rpm:limit")
        rpd = qm.redis.get(f"model:{model}:rpd:limit")
        tpm = qm.redis.get(f"model:{model}:tpm:limit")
        
        if rpm and float(rpm) != -1: m_entry["rpm"] = {"limit": int(float(rpm))}
        if rpd and float(rpd) != -1: m_entry["rpd"] = {"limit": int(float(rpd))}
        if tpm and float(tpm) != -1: m_entry["tpm"] = {"limit": int(float(tpm))}
        
        # Get category/tier
        cat = qm.redis.get(f"model:{model}:category")
        tier = qm.redis.get(f"model:{model}:tier")
        
        if cat: m_entry["category"] = cat.decode('utf-8')
        if tier: m_entry["tier"] = tier.decode('utf-8')
        
        # Params
        params = qm.redis.get(f"model:{model}:params")
        if params:
            try:
                m_entry["params"] = json.loads(params)
            except:
                pass
        
        config_data["models"].append(m_entry)
        
    # Write File
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(config_data, f, indent=2)
        
    print(f"✅ Restored {fname} with {len(models_list)} models.")

print("--- Recovery Complete ---")
