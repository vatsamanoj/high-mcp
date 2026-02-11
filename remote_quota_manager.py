import httpx
import logging
from typing import Optional, Dict, Any, List, Tuple

class RemoteQuotaManager:
    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
        # Explicitly set shorter timeouts for connect and read to avoid hanging
        self.client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(5.0, connect=2.0))
        self.logger = logging.getLogger("RemoteQuotaManager")

    async def get_model_for_request(self, preferred_model: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        try:
            params = {}
            if preferred_model:
                params["preferred_model"] = preferred_model
            
            response = await self.client.get("/get_model_for_request", params=params)
            if response.status_code == 200:
                data = response.json()
                return data.get("model"), data.get("config", {})
            return None, {}
        except Exception as e:
            self.logger.error(f"Error calling Quota Server: {e}")
            return None, {}

    async def update_quota(self, model_name: str, tokens_used: int, request_count: int = 1, config_file: Optional[str] = None):
        try:
            payload = {
                "model_name": model_name,
                "tokens_used": tokens_used,
                "request_count": request_count,
                "config_file": config_file
            }
            # Fire and forget mostly, but we await to ensure it's sent
            await self.client.post("/update_quota", json=payload)
        except Exception as e:
            self.logger.error(f"Error updating quota: {e}")

    async def mark_provider_blocked(self, model_name: str, reason: str, config_file: Optional[str] = None):
        try:
            payload = {
                "model_name": model_name,
                "reason": reason,
                "config_file": config_file
            }
            await self.client.post("/mark_provider_blocked", json=payload)
        except Exception as e:
            self.logger.error(f"Error blocking provider: {e}")

    async def get_all_models(self) -> List[Dict[str, Any]]:
        try:
            response = await self.client.get("/get_all_models")
            if response.status_code == 200:
                return response.json().get("models", [])
            return []
        except Exception as e:
            self.logger.error(f"Error getting models: {e}")
            return []

    async def get_speed_override(self) -> bool:
        try:
            response = await self.client.get("/get_speed_override")
            return response.json().get("enabled", False)
        except Exception:
            return False

    async def set_speed_override(self, enabled: bool):
        try:
            await self.client.post("/set_speed_override", json={"enabled": enabled})
        except Exception as e:
            self.logger.error(f"Error setting speed override: {e}")

    async def _sync_configuration_from_json(self):
        try:
            await self.client.post("/sync_configuration")
        except Exception as e:
            self.logger.error(f"Error syncing config: {e}")

    async def list_quotas(self) -> List[str]:
        try:
            response = await self.client.get("/quotas")
            if response.status_code == 200:
                return response.json().get("files", [])
            return []
        except Exception as e:
            self.logger.error(f"Error listing quotas: {e}")
            return []

    async def upload_quota(self, filename: str, content: str):
        try:
            await self.client.post("/quotas/upload", json={"filename": filename, "content": content})
        except Exception as e:
            self.logger.error(f"Error uploading quota: {e}")
            raise e

    async def delete_quota(self, filename: str):
        try:
            await self.client.delete(f"/quotas/{filename}")
        except Exception as e:
            self.logger.error(f"Error deleting quota: {e}")
            raise e

    async def close(self):
        await self.client.aclose()
