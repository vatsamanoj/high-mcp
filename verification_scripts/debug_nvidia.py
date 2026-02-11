import os
import requests


def test_nvidia_direct():
    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    api_key = os.environ.get("NVIDIA_API_KEY")

    if not api_key:
        print("Set NVIDIA_API_KEY in your environment before running this script.")
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "model": "moonshotai/kimi-k2.5",
        "messages": [{"role": "user", "content": "Why is the sky blue? Answer briefly."}],
        "max_tokens": 1024,
        "temperature": 1.00,
        "top_p": 1.00,
        "stream": False,
        "chat_template_kwargs": {"thinking": True},
    }

    print(f"Sending payload to {invoke_url}...")
    try:
        response = requests.post(invoke_url, headers=headers, json=payload, timeout=60)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:500]}")
    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    test_nvidia_direct()
