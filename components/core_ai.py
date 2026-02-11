
import os
import asyncio
from mcp.server.fastmcp import FastMCP
from dependencies import get_quota_manager, get_ai_engine, get_error_manager

# Global instances (so the component can access them)
quota_manager = None
ai_engine = None
error_manager = None

def setup(mcp=None, app=None):
    """
    Registers the core AI tools with the MCP server and/or FastAPI app.
    This function is called by ComponentManager.
    """
    global quota_manager, ai_engine, error_manager
    
    # Initialize services from dependencies
    # We assume dependencies have been set by the caller (server.py or ui_server.py)
    try:
        quota_manager = get_quota_manager()
        ai_engine = get_ai_engine()
        error_manager = get_error_manager()
    except Exception as e:
        print(f"⚠️ core_ai: Could not get dependencies: {e}")
        # If we can't get dependencies, we can't register tools safely
        return

    if mcp:
        @mcp.tool()
        async def generate_content(model: str, text: str) -> str:
            """Generate content using Google AI models with fallback support.
            
            Args:
                model: The preferred model name (e.g. "Gemini 2.5 Flash")
                text: The prompt text to send to the model
            """
            try:
                return await ai_engine.generate_content(model, text)
            except Exception as e:
                error_id = error_manager.log_error(e, context=f"generate_content(model={model})")
                return f"Error: {str(e)} (Log ID: {error_id})"

        @mcp.tool()
        def list_models() -> list[str]:
            """List all available models in the quota database."""
            # Use inner (sync) manager for sync tools
            models = quota_manager.inner.get_all_models()
            return [item["model"] for item in models]

        @mcp.tool()
        def get_quota(model_name: str) -> dict:
            """Get quota details for a specific model."""
            models = quota_manager.inner.get_all_models()
            for item in models:
                if item["model"].lower() == model_name.lower():
                    return item
            return {"error": f"Model '{model_name}' not found"}

        @mcp.tool()
        def get_available_models() -> list[dict]:
            """Get models that have remaining quota (< 90% usage)."""
            models = quota_manager.inner.get_all_models()
            available = []
            for item in models:
                if quota_manager.inner.is_model_available(item["model"]):
                    available.append(item)
            return available

        @mcp.tool()
        def get_api_config(model_name: str = None) -> dict:
            """Get the API configuration for a model (or the first available one)."""
            _, config = quota_manager.inner.get_model_for_request(model_name)
            if config:
                return config
            return {"error": "No configuration found"}
