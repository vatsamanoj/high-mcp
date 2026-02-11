from fastapi import FastAPI

def setup(mcp=None, app: FastAPI = None):
    if app:
        @app.get('/api/injected2')
        def injected2():
            return {'ok': True, 'name': 'injected2'}