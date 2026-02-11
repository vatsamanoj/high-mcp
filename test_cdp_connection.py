from playwright.sync_api import sync_playwright
import sys

def test_connection():
    print("🧪 TEST: Attempting to connect to Chrome via CDP...")
    try:
        with sync_playwright() as p:
            # Attempt connection
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
            print("✅ CONNECTION SUCCESSFUL!")
            
            # Check contexts
            contexts = browser.contexts
            print(f"   Contexts found: {len(contexts)}")
            
            if not contexts:
                print("   ⚠️ No contexts found (Browser is open but has no windows?)")
            else:
                # Try to get the first page of the first context
                pages = contexts[0].pages
                print(f"   Pages in first context: {len(pages)}")
                if pages:
                    print(f"   Active tab title: '{pages[0].title()}'")
            
            print("   Disconnecting...")
            browser.close()
            print("✅ TEST COMPLETED SUCCESSFULLY.")
            return True
            
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    if not success:
        sys.exit(1)
