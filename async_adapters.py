import asyncio
from redis_quota_manager import RedisQuotaManager

class LocalQuotaManagerAsync:
    def __init__(self, base_dir: str):
        self.inner = RedisQuotaManager(base_dir)

    async def get_model_for_request(self, preferred_model: str = None):
        return await asyncio.to_thread(self.inner.get_model_for_request, preferred_model)

    async def update_quota(self, model_name: str, tokens_used: int, request_count: int = 1, config_file: str = None):
        await asyncio.to_thread(self.inner.update_quota, model_name, tokens_used, request_count, config_file)

    async def mark_provider_blocked(self, model_name: str, reason: str, config_file: str = None):
        await asyncio.to_thread(self.inner.mark_provider_blocked, model_name, reason, config_file)
    
    # Proxy other methods to inner if needed (sync)
    def __getattr__(self, name):
        return getattr(self.inner, name)
