import threading
import time
import httpx
import json
import os
from redis_quota_manager import RedisQuotaManager

class FreeAISensor:
    def __init__(self, quota_manager: RedisQuotaManager, interval_seconds: int = 14400): # Check every 4 hours
        self.qm = quota_manager
        self.interval = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sensor_loop, daemon=True)
        
    def start(self):
        print("Starting FreeAISensor...")
        self.thread.start()
        
    def stop(self):
        self.stop_event.set()
        
    def _sensor_loop(self):
        # Initial delay to let server start
        time.sleep(10)
        
        while not self.stop_event.is_set():
            print("FreeAISensor: Checking model health...")
            self._check_all_models()
            
            # Wait for interval
            for _ in range(self.interval):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
            
    def _check_all_models(self):
        models = self.qm.get_all_models()
        for m in models:
            if self.stop_event.is_set():
                break
                
            model_name = m['model']
            
            print(f"FreeAISensor: Probing {model_name}...")
            
            try:
                config = self.qm.get_model_config(model_name)
                if not config:
                    continue
                    
                probe = self._probe_model(model_name, config)
                if probe is None:
                    self.qm.redis.delete(f"model:{model_name}:health")
                    print(f"FreeAISensor: {model_name} skipped (unconfigured)")
                    continue

                status = "ok" if probe else "down"
                self.qm.redis.set(f"model:{model_name}:health", status)
                print(f"FreeAISensor: {model_name} is {status}")
                
            except Exception as e:
                print(f"FreeAISensor: Error checking {model_name}: {e}")

    def _probe_model(self, model_name: str, config: dict):
        provider = config.get("provider", "google")
        api_key = config.get("api_key")
        endpoint = config.get("api_endpoint")
        
        if not api_key or "YOUR_" in api_key:
            return None
            
        try:
            if provider == "google":
                return self._probe_google(endpoint, api_key, model_name)
            elif provider == "openai":
                return self._probe_openai(endpoint, api_key, model_name)
            else:
                return False
        except Exception as e:
            print(f"Probe failed for {model_name}: {e}")
            return False

    def _probe_google(self, endpoint, key, model):
        # Fix endpoint construction
        base_url = endpoint
        if not base_url.endswith("models"):
            base_url = f"{base_url.rstrip('/')}/models"
            
        # Handle model ID resolution (simple case)
        # If it's a gemini model, it's usually just passed as is or mapped.
        # But here we don't have the mapping logic. 
        # We'll assume the model name in config is valid or close enough.
        # If it fails 404, it might be ID issue.
        url = f"{base_url}/{model}:generateContent?key={key}"

        payload = {"contents": [{"parts": [{"text": "Hi"}]}]}
        resp = httpx.post(url, json=payload, timeout=10)
        return resp.status_code == 200

    def _probe_openai(self, endpoint, key, model):
        url = f"{endpoint.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {key}"}
        payload = {"model": model, "messages": [{"role": "user", "content": "Hi"}]}
        resp = httpx.post(url, headers=headers, json=payload, timeout=10)
        return resp.status_code == 200
