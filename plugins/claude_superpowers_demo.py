"""Demo plugin that works in both MCP (server.py) and FastAPI (ui_server.py) runtimes."""

from typing import Any, Dict
from fastapi import APIRouter
from pydantic import BaseModel

SUPERPOWER_WORKFLOW = [
    "1) Use superpower_brainstorm to analyze requirements and alternatives.",
    "2) Use superpower_plan to produce a concrete implementation plan.",
    "3) Implement changes only as plugin files under plugins/.",
    "4) Generate tests with superpower_tdd and run them before integration.",
    "5) Run superpower_review and superpower_debug if failures occur.",
]


class PrepareRequest(BaseModel):
    prompt: str


def build_demo_payload(prompt: str) -> Dict[str, Any]:
    clean_prompt = (prompt or "").strip()
    return {
        "ready": bool(clean_prompt),
        "task": clean_prompt,
        "workflow": SUPERPOWER_WORKFLOW,
        "claude_prompt": (
            "You are operating in a 100% plugin-based runtime (server.py + ui_server.py). "
            "Create or update plugin files only under plugins/. "
            "Before plugging in, write and run tests for every plugin change.\n\n"
            f"User task: {clean_prompt}\n\n"
            "Follow workflow:\n- " + "\n- ".join(SUPERPOWER_WORKFLOW)
        ),
    }


router = APIRouter(prefix="/api/plugins/claude-superpowers-demo", tags=["plugins", "claude", "superpowers"])


@router.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "plugin": "claude_superpowers_demo"}


@router.post("/prepare")
async def prepare(req: PrepareRequest) -> Dict[str, Any]:
    return build_demo_payload(req.prompt)


def setup(mcp=None, app=None):
    if app is not None:
        app.include_router(router)

    if mcp is not None:

        @mcp.tool()
        async def prepare_claude_superpower_task(prompt: str) -> str:
            """Prepare a Claude + Superpowers plugin-oriented task prompt."""
            return build_demo_payload(prompt)["claude_prompt"]
