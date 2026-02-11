from fastapi import HTTPException, Depends
from typing import Callable, Any, Optional

# Global internal state
_error_manager: Any = None
_quota_manager: Any = None
_ai_engine: Any = None
_trust_system: Any = None

# Setters for initialization
def set_dependencies(error_manager=None, quota_manager=None, ai_engine=None, trust_system=None):
    global _error_manager, _quota_manager, _ai_engine, _trust_system
    if error_manager: _error_manager = error_manager
    if quota_manager: _quota_manager = quota_manager
    if ai_engine: _ai_engine = ai_engine
    if trust_system: _trust_system = trust_system

# Getters (Dependency Injection Providers)
def get_error_manager():
    if not _error_manager:
        raise HTTPException(503, "Error Manager not initialized")
    return _error_manager

def get_quota_manager():
    if not _quota_manager:
        raise HTTPException(503, "Quota Manager not initialized")
    return _quota_manager

def get_ai_engine():
    if not _ai_engine:
        raise HTTPException(503, "AI Engine not initialized")
    return _ai_engine

def get_trust_system():
    if not _trust_system:
        raise HTTPException(503, "Trust System not initialized")
    return _trust_system

# Route Handlers Factory
def patch_action(action_name: str):
    def handler(
        patch_id: str,
        manager = Depends(get_error_manager)
    ):
        method = getattr(manager, action_name)
        return method(patch_id)
    return handler
