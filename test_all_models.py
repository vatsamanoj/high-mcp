import asyncio
import httpx
import sys

# Constants
API_BASE = "http://localhost:8004/api"

async def test_all_models():
    print("🚀 Starting Model Verification Test...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Fetch Models
        print("\n📥 Fetching available models...")
        try:
            resp = await client.get(f"{API_BASE}/chat/models")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            print(f"✅ Found {len(models)} models.")
        except Exception as e:
            print(f"❌ Failed to fetch models: {e}")
            return

        if not models:
            print("⚠️ No models found to test.")
            return

        # 2. Test Each Model
        print("\n🧪 Testing each model with a simple prompt...")
        results = []
        
        for m in models:
            model_name = m['model']
            available = m['available']
            
            if not available:
                print(f"⏭️  Skipping {model_name} (Status: Blocked/Unavailable)")
                results.append({"model": model_name, "status": "skipped", "details": "Unavailable"})
                continue
                
            print(f"👉 Testing {model_name}...", end=" ", flush=True)
            
            try:
                # Send a minimal request
                payload = {
                    "model": model_name,
                    "message": "Say 'Test OK' if you can hear me.",
                    "images": []
                }
                
                chat_resp = await client.post(f"{API_BASE}/chat", json=payload)
                
                if chat_resp.status_code == 200:
                    response_text = chat_resp.json().get("response", "")
                    print(f"✅ OK")
                    results.append({"model": model_name, "status": "success", "response": response_text[:50] + "..."})
                else:
                    error_detail = chat_resp.text
                    print(f"❌ FAILED ({chat_resp.status_code})")
                    results.append({"model": model_name, "status": "failed", "error": error_detail})
                    
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
                results.append({"model": model_name, "status": "error", "error": str(e)})

        # 3. Summary
        print("\n" + "="*40)
        print("📊 TEST SUMMARY")
        print("="*40)
        success_count = sum(1 for r in results if r['status'] == 'success')
        failed_count = sum(1 for r in results if r['status'] in ['failed', 'error'])
        skipped_count = sum(1 for r in results if r['status'] == 'skipped')
        
        print(f"Total: {len(results)}")
        print(f"✅ Success: {success_count}")
        print(f"❌ Failed:  {failed_count}")
        print(f"⏭️  Skipped: {skipped_count}")
        print("-" * 40)
        
        for r in results:
            icon = "✅" if r['status'] == 'success' else ("⏭️ " if r['status'] == 'skipped' else "❌")
            print(f"{icon} {r['model']}")
            if r['status'] == 'failed' or r['status'] == 'error':
                print(f"   Error: {r.get('error')}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_all_models())
