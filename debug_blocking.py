import os
import sys
from redis_quota_manager import RedisQuotaManager

base_dir = os.getcwd()
qm = RedisQuotaManager(base_dir, persistence_enabled=False)

print("--- Debugging Orphan Configs ---")

# Scan for keys that look like config attributes
all_keys = qm.redis.keys("config:*:key")
print(f"Found {len(all_keys)} config keys in Redis.")

for k in all_keys:
    # k is bytes: config:filename:key
    k_str = k.decode('utf-8')
    parts = k_str.split(':')
    # parts: ['config', 'filename', 'key']
    if len(parts) >= 3:
        filename = parts[1]
        print(f"Found config in Redis: {filename}")
        
        # Check if file exists
        file_path = os.path.join(base_dir, "quotas", filename)
        if not os.path.exists(file_path):
            print(f"  MISSING FILE: {file_path}")
            # Retrieve details to prove we can recover
            endpoint = qm.redis.get(f"config:{filename}:endpoint")
            provider = qm.redis.get(f"config:{filename}:provider")
            print(f"  Recoverable Data: Endpoint={endpoint}, Provider={provider}")

print("--- End Debug ---")
