from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI

def setup(mcp: FastMCP = None, app: FastAPI = None):
    if app:
        @app.get('/api/injected')
        def injected():
            return {'ok': True}