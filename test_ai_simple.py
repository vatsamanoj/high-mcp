import asyncio
import os
import sys
from ai_engine import AIEngine

# Mock Quota Manager
class MockQuotaManager:
    async def check_quota(self, model: str, tokens: int = 0):
        return True
    async def record_usage(self, model: str, tokens: int, success: bool = True):
        pass

async def test_engine():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    engine = AIEngine(MockQuotaManager(), base_dir)
    
    print("Testing generate_content...")
    try:
        response = await engine.generate_content("gemini-1.5-flash", "Say hello", response_format="text")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTesting generate_content with JSON...")
    try:
        response = await engine.generate_content("gemini-1.5-flash", "Return a JSON object with key 'message' and value 'hello'", response_format="json")
        print(f"JSON Response: {response}")
    except Exception as e:
        print(f"JSON Error: {e}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_engine())
