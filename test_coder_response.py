import asyncio
import os
from ai_engine import AIEngine
from remote_quota_manager import RemoteQuotaManager

async def test():
    print("Initializing AI Engine...")
    qm = RemoteQuotaManager(base_url="http://localhost:8003")
    engine = AIEngine(qm, base_dir=os.getcwd())
    
    print(f"MAX_CONTEXT_CHARS: {engine.MAX_CONTEXT_CHARS}")
    
    print("\n--- Generating Project Context ---")
    context = await asyncio.to_thread(engine._get_project_context)
    
    print(f"Context Length: {len(context)} chars")
    print("-" * 30)
    print("Context Preview (First 1000 chars):")
    print(context[:1000])
    print("-" * 30)
    print("Context Preview (Last 500 chars):")
    print(context[-500:])
    print("-" * 30)
    
    await qm.close()

if __name__ == "__main__":
    asyncio.run(test())