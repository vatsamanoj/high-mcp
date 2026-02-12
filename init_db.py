import asyncio
from request_repository import RequestRepository


async def init_db():
    print("Initializing Database...")
    repo = RequestRepository("requests.db")
    await repo.initialize()
    print("Database initialized with WAL mode.")


if __name__ == "__main__":
    asyncio.run(init_db())
