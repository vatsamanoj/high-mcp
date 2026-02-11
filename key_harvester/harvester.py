import sys
import os
import time
import json
import re
from typing import Optional, Dict, Any

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redis_quota_manager import RedisQuotaManager
from ai_engine import AIEngine
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page

class KeyHarvesterAgent:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Disable persistence to avoid file locking conflicts with the main server
        self.quota_manager = RedisQuotaManager(base_dir, persistence_enabled=False)
        self.ai_engine = AIEngine(self.quota_manager)
        self.screenshot_dir = os.path.join(os.path.dirname(__file__), "screenshots")
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def _clean_html(self, html: str) -> str:
        """Clean HTML to reduce token usage."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "svg", "path", "noscript"]):
            script.decompose()
            
        # Get text and minimal structure
        # Just keeping body content for now
        if soup.body:
            text = soup.body.prettify()
        else:
            text = soup.prettify()
            
        # Truncate if too long (rough limit)
        if len(text) > 50000:
            return text[:50000] + "... (truncated)"
        return text

    def _ask_ai_for_action(self, url: str, html_content: str, history: list) -> dict:
        """Ask Gemini what to do next based on the page content."""
        prompt = f"""
        I am an autonomous agent trying to harvest an API Key from this website: {url}.
        
        Here is the simplified HTML content of the current page:
        ```html
        {html_content}
        ```
        
        Here is my history of actions so far:
        {json.dumps(history, indent=2)}
        
        Analyze the HTML. My goal is to find an API Key string (starts with 'gsk_', 'sk-', etc.) OR find the button/link to create/view one.
        
        Return a JSON object with ONE of the following structures:
        
        1. If you see the API Key directly:
        {{
            "status": "found",
            "key": "THE_API_KEY_STRING",
            "provider": "likely_provider_name"
        }}
        
        2. If you see an element I should click to get closer (e.g. "Create Key", "API Keys", "View"):
        {{
            "status": "action",
            "action": "click",
            "selector": "CSS_SELECTOR_FOR_ELEMENT",
            "reason": "explanation"
        }}
        
        3. If I need to fill a form (e.g. name for the key):
        {{
            "status": "action",
            "action": "fill",
            "selector": "CSS_SELECTOR_FOR_INPUT",
            "value": "High-MCP-Key",
            "reason": "naming the key"
        }}
        
        4. If you are stuck or need user help (e.g. login required):
        {{
            "status": "stuck",
            "message": "Please log in manually and press Enter in the terminal."
        }}
        
        ONLY RETURN THE JSON.
        """
        
        response = self.ai_engine.generate_content("gemini-2.0-flash", prompt)
        
        # Check for AI Engine errors
        if response.startswith("Error:"):
            print(f"AI Generation Error: {response}")
            return {"status": "error", "message": response}

        # Clean response to get JSON
        try:
            # simple cleanup for markdown code blocks
            clean_resp = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_resp)
        except Exception as e:
            print(f"AI JSON Parse Error: {e}\nResponse was: {response}")
            return {"status": "error", "message": "Failed to parse AI response"}

    def harvest(self, url: str, provider_name: str, fresh_start: bool = False, use_existing_chrome: bool = False):
        print(f"\n--- Starting Harvest for {provider_name} at {url} ---")
        
        with sync_playwright() as p:
            context = None
            browser = None
            
            if use_existing_chrome:
                print("\n📡 Connecting to EXISTING Chrome instance on port 9222...")
                try:
                    browser = p.chromium.connect_over_cdp("http://localhost:9222")
                    if not browser.contexts:
                        print("⚠️ No existing contexts found in Chrome. This is unusual.")
                        print("Trying to use the default context...")
                    
                    # For CDP, we usually just want to attach to the target.
                    # But connect_over_cdp gives us a Browser object.
                    # Let's see if we can just get the first context.
                    context = browser.contexts[0]
                    
                    print(f"✅ Connected to Chrome! (Open contexts: {len(browser.contexts)})")
                    print("Opening a NEW TAB for harvesting...")
                    page = context.new_page()
                    print("✅ New tab opened successfully.")
                    
                except IndexError:
                    print("\n❌ Error: Connected to Chrome, but no contexts found.")
                    print("Please try closing Chrome fully and restarting 'start_chrome_debug.bat'.")
                    return
                except Exception as e:
                    print(f"\n❌ FAILED to connect: {e}")
                    print("Did you run 'start_chrome_debug.bat' first?")
                    print("Is Chrome actually running with remote debugging enabled?")
                    return
            else:
                # User Data Directory for persistence (cookies, sessions)
                user_data_dir = os.path.join(os.path.dirname(__file__), "chrome_profile")
                
                if fresh_start and os.path.exists(user_data_dir):
                    print("⚠️ Clearing previous session data as requested...")
                    import shutil
                    try:
                        shutil.rmtree(user_data_dir)
                        print("✅ Session cleared.")
                    except Exception as e:
                        print(f"❌ Failed to clear session: {e}")

                os.makedirs(user_data_dir, exist_ok=True)
                
                print("Launching browser with ADVANCED anti-detection features...")
                
                # Anti-detection args
                args = [
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-infobars",
                    "--start-maximized",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process"
                ]
                
                # Launch persistent context
                context = p.chromium.launch_persistent_context(
                    user_data_dir,
                    channel="chrome",
                    headless=False,
                    args=args,
                    ignore_default_args=["--enable-automation"],
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    viewport=None,
                    locale="en-US",
                    timezone_id="America/New_York",
                    permissions=["geolocation"]
                )
                
                page = context.pages[0] if context.pages else context.new_page()
                
                # INJECT ADVANCED STEALTH SCRIPTS
                stealth_js = """
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
                    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                    Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
                    const originalQuery = window.navigator.permissions.query;
                    window.navigator.permissions.query = (parameters) => (
                        parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                    );
                """
                page.add_init_script(stealth_js)
            
            try:
                print(f"Navigating to {url}...")
                page.goto(url)
                
                print("\n" + "="*60)
                print("🛑 MANUAL INTERVENTION REQUIRED 🛑")
                print("Please interact with the opened browser window:")
                print("1. Log in completely (using email/password or Magic Link).")
                print("2. Navigate to the page where the API Key is visible.")
                print("3. Ensure you are fully authenticated and the page is loaded.")
                print("="*60 + "\n")
                
                input(">> Press ENTER here ONLY when you are ready to start harvesting...")
                print("Starting automation...")

                # Initial wait for load
                page.wait_for_load_state("networkidle")
                
                history = []
                max_steps = 20
                
                for step in range(max_steps):
                    print(f"\nStep {step + 1}: Analyzing page...")
                    
                    # Screenshot for debug
                    screenshot_path = os.path.join(self.screenshot_dir, f"step_{step}.png")
                    page.screenshot(path=screenshot_path)
                    
                    # Get content
                    content = self._clean_html(page.content())
                    
                    # Ask AI
                    decision = self._ask_ai_for_action(page.url, content, history)
                    print(f"AI Decision: {decision}")
                    
                    status = decision.get("status")
                    
                    if status == "found":
                        key = decision.get("key")
                        print(f"\n🎉 SUCCESS! Found Key: {key}")
                        self._save_key(provider_name, key)
                        break
                        
                    elif status == "action":
                        action = decision.get("action")
                        selector = decision.get("selector")
                        reason = decision.get("reason")
                        
                        print(f"Executing: {action} on {selector} ({reason})")
                        
                        try:
                            if action == "click":
                                page.click(selector, timeout=5000)
                            elif action == "fill":
                                value = decision.get("value", "MyKey")
                                page.fill(selector, value)
                                
                            # Wait for reaction
                            time.sleep(2)
                            page.wait_for_load_state("domcontentloaded")
                            
                            history.append(decision)
                            
                        except Exception as e:
                            print(f"Action failed: {e}")
                            history.append({"error": str(e), "failed_decision": decision})
                            
                    elif status == "stuck":
                        print(f"\n🤖 AI is stuck: {decision.get('message')}")
                        input(">> Please perform the action manually in the browser window, then press ENTER here to continue...")
                        
                    elif status == "error":
                        print("AI Error. Retrying...")
                        
                    else:
                        print("Unknown status. Stopping.")
                        break
                        
            except Exception as e:
                print(f"Harvesting Error: {e}")
            finally:
                context.close()

    def _save_key(self, provider: str, key: str):
        """Save the key to a new quota file or update existing."""
        filename = f"quota_{provider.lower()}_auto.json"
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "quotas", filename)
        
        data = {
            "provider": "openai" if provider.lower() != "google" else "google",
            "api_endpoint": "https://api.groq.com/openai/v1" if "groq" in provider.lower() else "UNKNOWN_ENDPOINT",
            "api_key": key,
            "models": []
        }
        
        # Try to guess endpoint from provider name (simple mapping)
        if "groq" in provider.lower():
            data["api_endpoint"] = "https://api.groq.com/openai/v1"
        elif "cerebras" in provider.lower():
            data["api_endpoint"] = "https://api.cerebras.ai/v1"
        elif "sambanova" in provider.lower():
            data["api_endpoint"] = "https://api.sambanova.ai/v1"
            
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved key to {filepath}")

if __name__ == "__main__":
    agent = KeyHarvesterAgent()
    
    print("Available targets: groq, cerebras, sambanova")
    target = input("Enter target provider: ").strip()
    url = input("Enter login URL (or press Enter for default): ").strip()
    
    defaults = {
        "groq": "https://console.groq.com/keys",
        "cerebras": "https://inference.cerebras.ai/",
        "sambanova": "https://cloud.sambanova.ai/"
    }
    
    if not url and target in defaults:
        url = defaults[target]
        
    print("\nHow would you like to connect?")
    print("1. Launch new Stealth Chrome (Standard)")
    print("2. Connect to EXISTING Chrome (Advanced - requires start_chrome_debug.bat)")
    mode = input("Select mode (1/2) [1]: ").strip()
    
    use_existing = mode == "2"
    fresh = False
    
    if not use_existing:
        fresh = input("Clear previous session cookies? (y/n) [n]: ").strip().lower() == "y"
        
    if url:
        agent.harvest(url, target, fresh_start=fresh, use_existing_chrome=use_existing)
    else:
        print("No URL provided.")
