import json
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP
from dependencies import get_ai_engine

# Setup Logger
logger = logging.getLogger("superpowers")

# --- Models ---

class BrainstormRequest(BaseModel):
    problem: str
    context: Optional[str] = None
    model: Optional[str] = None

class PlanRequest(BaseModel):
    design_doc: str
    model: Optional[str] = None

class CodeReviewRequest(BaseModel):
    code: str
    requirements: Optional[str] = None
    model: Optional[str] = None

class TDDRequest(BaseModel):
    feature_spec: str
    existing_code: Optional[str] = None
    model: Optional[str] = None

class DebugRequest(BaseModel):
    error_log: str
    code_snippet: Optional[str] = None
    model: Optional[str] = None

# --- Superpowers Logic ---

class SuperpowersService:
    def __init__(self):
        pass

    @staticmethod
    def _normalize_text(value: Optional[str]) -> str:
        return (value or "").strip()

    async def _generate(self, model: Optional[str], prompt: str, response_format: Optional[str] = None) -> str:
        engine = get_ai_engine()
        return await engine.generate_content(model, prompt, response_format=response_format)

    async def brainstorm(self, problem: str, context: str = "", model: str = None) -> str:
        problem = self._normalize_text(problem)
        context = self._normalize_text(context)
        if not problem:
            raise HTTPException(status_code=400, detail="problem must not be empty")
        prompt = (
            "You are a Principal Software Architect using the 'Brainstorming' superpower.\n"
            "Your goal is to deeply analyze the user's problem before any code is written.\n"
            "1. Analyze the requirements and identify ambiguity.\n"
            "2. Propose 3 distinct architectural approaches (Low complexity, Medium complexity, High complexity).\n"
            "3. List potential edge cases and failure modes.\n"
            "4. Recommend a testing strategy.\n\n"
            f"Problem: {problem}\n"
            f"Context: {context}\n"
        )
        return await self._generate(model, prompt)

    async def create_plan(self, design_doc: str, model: str = None) -> List[Dict[str, Any]]:
        design_doc = self._normalize_text(design_doc)
        if not design_doc:
            raise HTTPException(status_code=400, detail="design_doc must not be empty")
        prompt = (
            "You are a Technical Project Manager using the 'Writing Plans' superpower.\n"
            "Convert the following design document into a detailed, step-by-step implementation plan.\n"
            "Return ONLY a valid JSON array of objects, where each object has:\n"
            "- 'step': Step number (int)\n"
            "- 'title': Short title (str)\n"
            "- 'description': Detailed instruction (str)\n"
            "- 'verification': How to verify this step is done (str)\n"
            "- 'files': List of files to be touched (List[str])\n\n"
            f"Design Document:\n{design_doc}\n"
        )
        response = await self._generate(model, prompt, response_format="json")
        try:
            # Extract JSON from response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != -1:
                return json.loads(response[start:end])
            return [{"error": "Could not parse JSON plan", "raw": response}]
        except json.JSONDecodeError as e:
            return [{"error": f"JSON Parse Error: {str(e)}", "raw": response}]

    async def review_code(self, code: str, requirements: str = "", model: str = None) -> str:
        code = self._normalize_text(code)
        requirements = self._normalize_text(requirements)
        if not code:
            raise HTTPException(status_code=400, detail="code must not be empty")
        prompt = (
            "You are a Senior Engineer using the 'Code Review' superpower.\n"
            "Review the following code strictly against the requirements and best practices.\n"
            "Focus on:\n"
            "- Bugs and Logic Errors\n"
            "- Security Vulnerabilities\n"
            "- Performance Bottlenecks\n"
            "- Readability and Style (DRY, YAGNI)\n\n"
            f"Requirements: {requirements}\n"
            f"Code:\n{code}\n"
        )
        return await self._generate(model, prompt)

    async def generate_tests(self, feature_spec: str, existing_code: str = "", model: str = None) -> str:
        feature_spec = self._normalize_text(feature_spec)
        existing_code = self._normalize_text(existing_code)
        if not feature_spec:
            raise HTTPException(status_code=400, detail="feature_spec must not be empty")
        prompt = (
            "You are a QA Engineer using the 'Test Driven Development' superpower.\n"
            "Write a comprehensive pytest suite for the following feature specification.\n"
            "Follow the RED-GREEN-REFACTOR methodology: write tests that fail if the feature isn't implemented.\n"
            "Include edge cases and error handling tests.\n\n"
            f"Feature Spec: {feature_spec}\n"
            f"Existing Code Context: {existing_code}\n"
        )
        return await self._generate(model, prompt)

    async def systematic_debug(self, error_log: str, code_snippet: str = "", model: str = None) -> str:
        error_log = self._normalize_text(error_log)
        code_snippet = self._normalize_text(code_snippet)
        if not error_log:
            raise HTTPException(status_code=400, detail="error_log must not be empty")
        prompt = (
            "You are a Debugging Specialist using the 'Systematic Debugging' superpower.\n"
            "Apply the 4-phase root cause analysis process:\n"
            "1. Observation: What exactly is failing?\n"
            "2. Hypothesis: What are the possible causes?\n"
            "3. Experiment: How can we validate the hypothesis?\n"
            "4. Solution: Proposed fix.\n\n"
            f"Error Log:\n{error_log}\n"
            f"Code Snippet:\n{code_snippet}\n"
        )
        return await self._generate(model, prompt)

# --- Component Setup ---

service = SuperpowersService()
router = APIRouter(prefix="/api/superpowers", tags=["superpowers"])

@router.post("/brainstorm")
async def api_brainstorm(req: BrainstormRequest):
    result = await service.brainstorm(req.problem, req.context, req.model)
    return {"result": result}

@router.post("/plan")
async def api_plan(req: PlanRequest):
    result = await service.create_plan(req.design_doc, req.model)
    return {"plan": result}

@router.post("/review")
async def api_review(req: CodeReviewRequest):
    result = await service.review_code(req.code, req.requirements, req.model)
    return {"review": result}

@router.post("/tdd")
async def api_tdd(req: TDDRequest):
    result = await service.generate_tests(req.feature_spec, req.existing_code, req.model)
    return {"test_code": result}

@router.post("/debug")
async def api_debug(req: DebugRequest):
    result = await service.systematic_debug(req.error_log, req.code_snippet, req.model)
    return {"analysis": result}

def setup(mcp: FastMCP = None, app: Any = None):
    """
    Registers Superpowers skills as MCP tools and FastAPI routes.
    """
    
    # 1. Register FastAPI Routes
    if app:
        logger.info("🦸 Registering Superpowers API routes...")
        app.include_router(router)
    
    # 2. Register MCP Tools
    if mcp:
        logger.info("🦸 Registering Superpowers MCP tools...")
        
        @mcp.tool()
        async def superpower_brainstorm(problem: str, context: str = "") -> str:
            """
            [Superpower: Brainstorming] Analyze a problem and propose architectural solutions.
            Use this BEFORE writing any code or plans.
            """
            return await service.brainstorm(problem, context)

        @mcp.tool()
        async def superpower_plan(design_doc: str) -> str:
            """
            [Superpower: Writing Plans] Convert a design document into a step-by-step JSON implementation plan.
            """
            plan = await service.create_plan(design_doc)
            return json.dumps(plan, indent=2)

        @mcp.tool()
        async def superpower_review(code: str, requirements: str = "") -> str:
            """
            [Superpower: Code Review] Critique code against requirements and best practices.
            """
            return await service.review_code(code, requirements)

        @mcp.tool()
        async def superpower_tdd(feature_spec: str, existing_code: str = "") -> str:
            """
            [Superpower: TDD] Generate pytest code for a feature specification.
            """
            return await service.generate_tests(feature_spec, existing_code)

        @mcp.tool()
        async def superpower_debug(error_log: str, code_snippet: str = "") -> str:
            """
            [Superpower: Systematic Debugging] Analyze an error log to find the root cause.
            """
            return await service.systematic_debug(error_log, code_snippet)
