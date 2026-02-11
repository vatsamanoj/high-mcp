import aiosqlite
import asyncio

async def seed():
    async with aiosqlite.connect("requests.db") as db:
        # Create table if not exists (in case repo didn't init yet, though ui_server will)
        # But we should rely on RequestRepository to create it. 
        # So we just assume it might not exist, but let's just create it here manually to be safe
        # or import RequestRepository.
        pass

    from request_repository import RequestRepository
    repo = RequestRepository()
    await repo.initialize()
    
    # Add sample template
    # Pattern: Translate "something" to Language
    pattern = r'Translate "(.+)" to (.+)'
    format_str = "Translate: {0} -> {1}"
    
    await repo.add_template(pattern, format_str, "Simple Translation Template")
    print("✅ Template added: Translate \"...\" to ...")

if __name__ == "__main__":
    asyncio.run(seed())
