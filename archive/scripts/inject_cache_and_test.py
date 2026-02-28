import httpx
import json
import base64
import hashlib
import sqlite3
import time

def test_image_cache_injection():
    db_path = "requests.db"
    url = "http://localhost:8004/api/chat"
    
    # 1. Prepare Dummy Image
    dummy_image_data = base64.b64encode(b"test_image_lookup_content").decode('utf-8')
    images = [{
        "mime_type": "image/png",
        "data": dummy_image_data
    }]
    
    # Calculate Image Hash manually to match backend logic
    # Backend: json.dumps(images, sort_keys=True) -> sha256
    content = json.dumps(images, sort_keys=True)
    image_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    
    print(f"Computed Image Hash: {image_hash}")
    
    # 2. Inject Fake Record into DB
    print("Injecting fake record into DB...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Ensure table has image_hash (it should, based on my changes)
    try:
        cursor.execute("""
            INSERT INTO requests (input_hash, input_text, output_text, model_name, created_at, image_hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "fake_input_hash_123", 
            "Original Prompt: Describe this test image", 
            "Manually Injected Result: This is a test image.", 
            "test-model", 
            time.time(), 
            image_hash
        ))
        conn.commit()
        print("✅ Injection successful.")
    except Exception as e:
        print(f"❌ Injection failed: {e}")
        conn.close()
        return

    conn.close()
    
    # 3. Send Request with SAME Image and EMPTY Prompt
    print("\nSending Request with SAME Image and EMPTY Prompt...")
    payload = {
        "model": "gemini-1.5-flash",
        "message": "",
        "images": images
    }
    
    try:
        response = httpx.post(url, json=payload, timeout=10.0)
        print(f"Response Code: {response.status_code}")
        resp_json = response.json()
        print("Response Body Snippet:", str(resp_json)[:200])
        
        resp_text = resp_json.get("response", "")
        if "Manually Injected Result" in resp_text:
            print("\n✅ SUCCESS! The system returned the cached result based on image hash.")
        else:
            print("\n❌ FAILURE. Did not find injected result.")
            
    except Exception as e:
        print(f"Error calling API: {e}")

if __name__ == "__main__":
    test_image_cache_injection()
