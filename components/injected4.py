from fastapi import FastAPI

def setup(mcp=None, app: FastAPI = None):
    if app:
        @app.get('/api/injected4')
        def injected4():
            return {'ok': True, 'name': 'injected4'}