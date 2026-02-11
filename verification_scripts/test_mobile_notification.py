import sys
import os
import time
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from notification_system import NotificationSystem

def test_mobile_notification():
    print("🚀 Testing Mobile Notification (Ntfy.sh)...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notification_system = NotificationSystem(base_dir)
    
    # 1. Check if config exists
    config_path = os.path.join(base_dir, "notification_config.json")
    if os.path.exists(config_path):
        print(f"✅ Config file found at: {config_path}")
        with open(config_path, 'r') as f:
            print(f"   Config content: {f.read()}")
    else:
        print("❌ Config file not found!")
        return

    # 2. Send a test INFO notification
    print("\n📨 Sending INFO notification...")
    notification_system.notify("This is a test INFO notification from High-MCP.", "INFO")
    
    # 3. Send a test BLOCKED notification
    print("\n📨 Sending BLOCKED notification (High Priority)...")
    notification_system.notify("Test Alert: Your mobile configuration is successful!", "BLOCKED")
    
    print("\n✅ Notifications sent (async).")
    
    # Read topic from config to display correct link
    try:
        with open(config_path, 'r') as f:
            conf = json.load(f)
            topic = conf.get("ntfy", {}).get("topic", "unknown")
            print(f"   Topic: {topic}")
            print(f"   Link: https://ntfy.sh/{topic}")
    except:
        pass

    # Give async thread time to finish
    time.sleep(2)

if __name__ == "__main__":
    test_mobile_notification()
