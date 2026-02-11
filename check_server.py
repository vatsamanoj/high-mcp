import httpx
import asyncio

async def check_server():
    try:
        async with httpx.AsyncClient() as client:
            # Check UI Node
            resp = await client.get('http://localhost:8004/dashboard', timeout=2.0)
            print(f"UI Node Status: {resp.status_code}")
            
            # Check Quota Node
            resp_q = await client.get('http://localhost:8003/status', timeout=2.0)
            print(f"Quota Node Status: {resp_q.status_code}")

            if resp.status_code == 200 and resp_q.status_code == 200:
                print("Cluster is fully responsive.")
            else:
                print("Cluster returned error.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {repr(e)}")

if __name__ == "__main__":
    asyncio.run(check_server())