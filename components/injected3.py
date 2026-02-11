from fastapi import FastAPI

def setup(mcp=None, app: FastAPI = None):
    if app:
        @app.get('/api/injected3')
        def injected3():
            return {'ok': True, 'name': 'injected3', 'v': 2}
