import os
from redis_quota_manager import RedisQuotaManager

base_dir = os.getcwd()
qm = RedisQuotaManager(base_dir, persistence_enabled=False)

# Check model config link
model = "llama3.1-8b"
print(f"Checking {model}...")

cf = qm.redis.get(f"model:{model}:config_file")
print(f"Config File: {cf}")

# Check legacy keys
keys = qm.redis.keys("model:*:config_file")
print(f"Found {len(keys)} model-config links.")
for k in keys[:5]:
    m = k.decode('utf-8').split(":")[1]
    v = qm.redis.get(k).decode('utf-8')
    print(f"  {m} -> {v}")
