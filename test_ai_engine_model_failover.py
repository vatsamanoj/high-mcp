import asyncio
from pathlib import Path

from ai_engine import AIEngine


class DummyQuotaManager:
    def __init__(self):
        self.calls = []

    async def get_model_for_request(self, preferred_model=None):
        self.calls.append(preferred_model)
        if preferred_model == "model-b":
            return "model-b", {
                "config_file": "b.json",
                "api_endpoint": "https://example.invalid",
                "api_key": "dummy",
                "provider": "google",
                "params": {},
            }
        # Simulate biased selector returning the same first model by default.
        return "model-a", {
            "config_file": "a.json",
            "api_endpoint": "https://example.invalid",
            "api_key": "dummy",
            "provider": "google",
            "params": {},
        }

    def get_all_models(self):
        return [{"model": "model-a"}, {"model": "model-b"}]

    async def update_quota(self, model_name, tokens_used=0, request_count=1, config_file=None):
        return None

    async def mark_provider_blocked(self, model_name, reason, config_file=None):
        return None


def test_generate_content_switches_to_alternative_model_on_failure(tmp_path):
    qm = DummyQuotaManager()
    engine = AIEngine(quota_manager=qm, base_dir=str(Path(tmp_path)))

    async def fake_resolve(model_name, api_endpoint, api_key):
        return model_name

    async def fake_google_call(api_endpoint, api_key, model_id, text, images=None, response_format="text", tools=None):
        if model_id == "model-a":
            raise Exception("temporary provider failure")
        return {"text": "fallback-ok", "usage": 1, "tool_calls": []}

    engine._resolve_model_id = fake_resolve
    engine._call_google_api = fake_google_call
    
    async def fake_ensure_repo():
        return None

    engine._ensure_repo = fake_ensure_repo

    result = asyncio.run(engine.generate_content(None, "hello world", response_format="json"))

    assert result == "fallback-ok"
    assert any(call == "model-b" for call in qm.calls)
