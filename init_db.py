import asyncio
from request_repository import RequestRepository
import os

async def init_db():
    print("Initializing Database...")
    repo = RequestRepository("requests.db")
    await repo.initialize()
    print("✅ Database Initialized with WAL Mode.")

if __name__ == "__main__":
    asyncio.run(init_db())
