from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI

def setup(mcp: FastMCP = None, app: FastAPI = None):
    """
    Demo Component to verify Unified Architecture.
    """
    print(f"👋 Demo Component Loading... (MCP={mcp is not None}, App={app is not None})")

    if mcp:
        @mcp.tool()
        def demo_tool(x: int) -> int:
            """Multiplies input by 10."""
            return x * 10
        print("   -> Registered 'demo_tool' on MCP")

    if app:
        @app.get("/api/demo")
        def demo_route():
            return {"message": "Hello from Demo Component!", "status": "active"}
        print("   -> Registered '/api/demo' on FastAPI")
