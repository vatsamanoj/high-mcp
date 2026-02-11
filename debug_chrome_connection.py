import json
import sys
import urllib.request
import urllib.error

def check_chrome_debugging_port():
    print("🔍 Checking Chrome Remote Debugging Port (9222)...")
    url = "http://localhost:9222/json/version"
    
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status == 200:
                print("✅ SUCCESS: Chrome is listening on port 9222.")
                body = response.read().decode('utf-8')
                data = json.loads(body)
                print(f"   Browser: {data.get('Browser')}")
                print(f"   User Agent: {data.get('User-Agent')}")
                print(f"   WebSocket URL: {data.get('webSocketDebuggerUrl')}")
                return True
            else:
                print(f"⚠️ Port 9222 is open but returned status code: {response.status}")
                return False
                
    except urllib.error.URLError as e:
        print("❌ FAILURE: Could not connect to http://localhost:9222")
        print(f"   Error details: {e}")
        print("\n   This means Chrome is NOT running with remote debugging enabled.")
        print("   Common reasons:")
        print("   1. Chrome was already running before you ran 'start_chrome_debug.bat'.")
        print("      -> You must CLOSE ALL Chrome windows (check Task Manager) and run the .bat again.")
        print("   2. You are running as Administrator but Chrome is user-level (or vice versa).")
        return False
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    check_chrome_debugging_port()
