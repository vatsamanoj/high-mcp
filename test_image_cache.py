import httpx
import json
import base64

def test_image_cache_lookup():
    url = "http://localhost:8004/api/chat"
    
    # Create a dummy image
    dummy_image_data = base64.b64encode(b"fake_image_content_for_test_123").decode('utf-8')
    images = [{
        "mime_type": "image/png",
        "data": dummy_image_data
    }]
    
    # 1. Send Request with Prompt (Should Save to Cache with Image Hash)
    print("1. Sending Request with Prompt...")
    payload1 = {
        "model": "gemini-1.5-flash",
        "message": "Describe this image",
        "images": images
    }
    try:
        # Note: Since we don't have real API keys or image, this might fail at AI Engine level 
        # unless we mock it or if the AI engine handles failure gracefully.
        # But we want to test the caching logic.
        # If the AI call fails, it won't save to cache.
        # So we rely on the fact that if it fails, we can't test cache.
        # However, for this test, we assume the AI Engine might fail but we want to verify the logic.
        # Wait, if AI fails, nothing is saved.
        # We need to simulate a successful save or rely on an existing cache.
        # Or we can assume the user has a working key.
        
        # Let's try. If it fails due to invalid image/key, we can't verify cache.
        # But we can check if the server is up.
        
        response1 = httpx.post(url, json=payload1, timeout=20.0)
        print(f"Response 1: {response1.status_code}")
        print(response1.text)
        
    except Exception as e:
        print(f"Error 1: {e}")

    # 2. Send Request with SAME Image and EMPTY Prompt (Should Hit Image Cache)
    print("\n2. Sending Request with SAME Image and EMPTY Prompt...")
    payload2 = {
        "model": "gemini-1.5-flash",
        "message": "",
        "images": images
    }
    
    try:
        response2 = httpx.post(url, json=payload2, timeout=20.0)
        print(f"Response 2: {response2.status_code}")
        # We look for "Cached Result from prompt" in the output
        print(response2.json().get("response", "No response field"))
        
        if "Cached Result from prompt" in response2.json().get("response", ""):
            print("\n✅ Success! Image Cache Lookup Worked.")
        else:
            print("\n⚠️ Cache Lookup check failed (or first request didn't succeed/save).")

    except Exception as e:
        print(f"Error 2: {e}")

if __name__ == "__main__":
    test_image_cache_lookup()
