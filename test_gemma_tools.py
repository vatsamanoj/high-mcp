
import asyncio
import os
import json
from ai_engine import AIEngine
from async_adapters import LocalQuotaManagerAsync

async def main():
    # api_key = os.environ.get("GOOGLE_API_KEY")
    # if not api_key:
    #     print("Skipping test: GOOGLE_API_KEY not set")
    #     return

    quota_manager = LocalQuotaManagerAsync(os.getcwd())
    ai = AIEngine(quota_manager=quota_manager)
    
    tools = [{
        "function_declarations": [{
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "location": {"type": "STRING", "description": "The city and state, e.g. San Francisco, CA"}
                },
                "required": ["location"]
            }
        }]
    }]

    import random
    print("Testing gemma-3-12b-it with tools...")
    try:
        result = await ai.generate_content(
            model_name="gemma-3-12b-it",
            text=f"What is the weather in San Francisco? {random.randint(1, 10000)}",
            tools=tools
        )
        print("Result:", json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
