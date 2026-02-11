import httpx
import json
import os
import glob

# Search for quota files in the quotas directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUOTA_DIR = os.path.join(BASE_DIR, "quotas")
quota_files = glob.glob(os.path.join(QUOTA_DIR, "quota_*.json"))

if not quota_files:
    print(f"No quota files found in {QUOTA_DIR}")
    exit(1)

# Use the first one found or specific one if needed
try:
    with open(quota_files[0], 'r', encoding='utf-8') as f:
        config = json.load(f)
        api_key = config.get('api_key')
        api_endpoint = config.get('api_endpoint', "https://generativelanguage.googleapis.com/v1/models")
except Exception as e:
    print(f"Error loading config: {e}")
    exit(1)

def list_models():
    print(f"Fetching models from {api_endpoint}...")
    try:
        # Check v1 first
        url = api_endpoint
        if not url.endswith("models"):
            url = f"{url.rstrip('/')}/models"
            
        resp = httpx.get(f"{url}?key={api_key}")
        if resp.status_code != 200:
            print(f"Error (v1): {resp.status_code} - {resp.text}")
        else:
            data = resp.json()
            models = data.get('models', [])
            print(f"\nFound {len(models)} models available to your key (v1):\n")
            print(f"{'NAME (ID)':<50} | {'DISPLAY NAME':<30} | {'VERSION'}")
            print("-" * 100)
            for m in models:
                name = m.get('name', '').replace('models/', '')
                display = m.get('displayName', '')
                version = m.get('version', '')
                print(f"{name:<50} | {display:<30} | {version}")

        # Check v1beta
        print("\nChecking v1beta endpoint...")
        url_beta = "https://generativelanguage.googleapis.com/v1beta/models"
        resp_beta = httpx.get(f"{url_beta}?key={api_key}")
        if resp_beta.status_code != 200:
            print(f"Error (v1beta): {resp_beta.status_code} - {resp_beta.text}")
        else:
            data = resp_beta.json()
            models = data.get('models', [])
            print(f"\nFound {len(models)} models available to your key (v1beta):\n")
            print(f"{'NAME (ID)':<50} | {'DISPLAY NAME':<30} | {'VERSION'}")
            print("-" * 100)
            for m in models:
                name = m.get('name', '').replace('models/', '')
                display = m.get('displayName', '')
                version = m.get('version', '')
                print(f"{name:<50} | {display:<30} | {version}")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    list_models()
