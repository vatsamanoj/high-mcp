import aiosqlite
import asyncio

async def list_templates():
    try:
        async with aiosqlite.connect("requests.db") as db:
            async with db.execute("SELECT id, pattern, minimized_prompt_format, description FROM templates") as cursor:
                rows = await cursor.fetchall()
                if not rows:
                    print("No templates found.")
                else:
                    for row in rows:
                        print(f"ID: {row[0]}")
                        print(f"Pattern: {row[1]}")
                        print(f"Format: {row[2]}")
                        print(f"Desc: {row[3]}")
                        print("-" * 20)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(list_templates())
