import httpx
import json

def test_empty_prompt():
    url = "http://localhost:8004/api/chat"
    payload = {
        "model": "gemini-1.5-flash", # Any model
        "message": "",
        "images": []
    }
    
    print("Sending empty prompt request...")
    try:
        response = httpx.post(url, json=payload, timeout=10.0)
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200 and "response" in response.json():
            print("\n✅ Success! Received response for empty prompt.")
        else:
            print("\n❌ Failed.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_empty_prompt()
