import logging
import os
import sys
import json
import httpx
import threading
from datetime import datetime

class NotificationSystem:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.log_file = os.path.join(base_dir, f"notifications_{os.getpid()}.log")
        self.config_file = os.path.join(base_dir, "notification_config.json")
        
        # Configure logging
        self.logger = logging.getLogger(f"SystemNotification_{os.getpid()}")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        try:
            fh = logging.FileHandler(self.log_file)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        except Exception as e:
            print(f"Failed to setup file logging: {e}")

        # Load or create default config
        self.config = self._load_config()

    def _load_config(self):
        default_config = {
            "ntfy": {
                "enabled": True,
                "topic": "high_mcp_system_alerts",
                "priority": "high",
                "server": "https://ntfy.sh"
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading notification config: {e}")
                return default_config
        else:
            # Save default config
            try:
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
            except:
                pass
            return default_config

    def notify(self, message: str, level: str = "INFO"):
        """
        Sends a notification (logs to file, prints to console, and sends mobile push).
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Console output with visibility
        print(f"\n{'='*50}")
        print(f"📢 SYSTEM NOTIFICATION: {level}")
        print(f"{message}")
        print(f"{'='*50}\n")
        
        # Log to file
        if level.upper() in ["ERROR", "CRITICAL", "BLOCKED"]:
            self.logger.error(message)
            # Send high priority mobile notification for errors
            self._send_mobile_push(message, level, priority="high")
        else:
            self.logger.info(message)
            # Send default priority for info
            self._send_mobile_push(message, level, priority="default")

    def _send_mobile_push(self, message: str, level: str, priority: str = "default"):
        """
        Sends a push notification via Ntfy.sh (runs in a separate thread to be non-blocking).
        """
        if not self.config.get("ntfy", {}).get("enabled", False):
            return

        def _send():
            ntfy_config = self.config["ntfy"]
            topic = ntfy_config.get("topic", "high_mcp_system_alerts")
            server = ntfy_config.get("server", "https://ntfy.sh")
            
            # Map level to ntfy tags/priority
            tags = "information_source"
            push_priority = priority # Default from argument

            if level.upper() == "BLOCKED":
                tags = "no_entry_sign,rotating_light"
                push_priority = "urgent"
            elif level.upper() == "ERROR":
                tags = "warning"
                push_priority = "high"
            
            url = f"{server}/{topic}"
            
            headers = {
                "Title": f"MCP System Alert: {level}",
                "Priority": push_priority,
                "Tags": tags
            }
            
            try:
                httpx.post(url, content=message, headers=headers, timeout=5.0)
            except Exception as e:
                # Fail silently for push notifications to not disrupt main flow
                print(f"Warning: Failed to send mobile notification: {e}")

        # Run in thread
        threading.Thread(target=_send, daemon=True).start()


# Global instance for easy access if needed, though dependency injection is preferred
_instance = None

def get_notification_system(base_dir: str = None):
    global _instance
    if _instance is None and base_dir:
        _instance = NotificationSystem(base_dir)
    return _instance
