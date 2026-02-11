import uvicorn
import os
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Tuple
from redis_quota_manager import RedisQuotaManager
from components.ModelAndQuotaPagingAndFiltering import pagination_and_filtering_component

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
quota_manager = RedisQuotaManager(BASE_DIR)
app = FastAPI(title="Quota Server (State Node)")

class UpdateQuotaRequest(BaseModel):
    model_name: str
    tokens_used: int
    request_count: int
    config_file: Optional[str] = None

class BlockProviderRequest(BaseModel):
    model_name: str
    reason: str
    config_file: Optional[str] = None

class ToggleRequest(BaseModel):
    enabled: bool

app.include_router(pagination_and_filtering_component)

@app.get("/status")
async def status():
    return {"status": "active", "pid": os.getpid()}

if __name__ == "__main__":
    print("Starting Quota Server on port 8003...")
    uvicorn.run(app, host="0.0.0.0", port=8003)
