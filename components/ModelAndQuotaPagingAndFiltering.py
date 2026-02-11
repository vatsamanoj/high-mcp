from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import Optional, List, Any
import sys
import os

# Ensure root dir is in path for importing dependencies
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dependencies import get_quota_manager

# Create the router
pagination_and_filtering_component = APIRouter()

# Models
class ModelRequest(BaseModel):
    model_name: str

class PaginationRequest(BaseModel):
    page: int
    per_page: int

class FilterRequest(BaseModel):
    model_name: Optional[str]

# Endpoints
@pagination_and_filtering_component.get("/api/models/filter")
def get_all_models(
    page: int = Query(1, ge=1),
    per_page: int = Query(5, ge=1, le=100),
    model_name: Optional[str] = None,
    qm = Depends(get_quota_manager)
):
    all_models = qm.get_all_models()
    
    # Filter
    if model_name:
        all_models = [m for m in all_models if model_name.lower() in m["model"].lower()]
    
    # Pagination
    total = len(all_models)
    start = (page - 1) * per_page
    end = start + per_page
    items = all_models[start:end]

    return {
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": (total + per_page - 1) // per_page,
        "filters": {"model_name": model_name},
        "items": items
    }

@pagination_and_filtering_component.get("/api/quotas/filter")
def get_quotas(
    page: int = Query(1, ge=1),
    per_page: int = Query(5, ge=1, le=100),
    model_name: Optional[str] = None,
    qm = Depends(get_quota_manager)
):
    # Reusing logic for now as quotas are effectively models in this system
    return get_all_models(page, per_page, model_name, qm)

# Component Manager Hook
def setup(mcp=None, app=None, **kwargs):
    if app:
        app.include_router(pagination_and_filtering_component)
        print("✅ ModelAndQuotaPagingAndFiltering component setup complete.")
    if mcp:
        print("ℹ️ ModelAndQuotaPagingAndFiltering: MCP support not implemented yet.")
