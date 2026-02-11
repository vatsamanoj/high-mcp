import asyncio

import pytest
from fastapi import HTTPException

import components.superpowers as sp


class DummyEngine:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    async def generate_content(self, model, prompt, response_format=None):
        self.calls.append((model, prompt, response_format))
        return self.response


def test_create_plan_parses_json_payload(monkeypatch):
    engine = DummyEngine('[{"step":1,"title":"A","description":"d","verification":"v","files":["x.py"]}]')
    monkeypatch.setattr(sp, "get_ai_engine", lambda: engine)

    svc = sp.SuperpowersService()
    result = asyncio.run(svc.create_plan("Design"))

    assert isinstance(result, list)
    assert result[0]["step"] == 1
    assert engine.calls[0][2] == "json"


def test_create_plan_returns_error_when_json_invalid(monkeypatch):
    engine = DummyEngine("not json")
    monkeypatch.setattr(sp, "get_ai_engine", lambda: engine)

    svc = sp.SuperpowersService()
    result = asyncio.run(svc.create_plan("Design"))

    assert result[0]["error"].startswith("Could not parse JSON plan")


def test_brainstorm_validates_problem(monkeypatch):
    engine = DummyEngine("ok")
    monkeypatch.setattr(sp, "get_ai_engine", lambda: engine)

    svc = sp.SuperpowersService()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(svc.brainstorm("   "))

    assert exc.value.status_code == 400


def test_review_code_validates_code(monkeypatch):
    engine = DummyEngine("ok")
    monkeypatch.setattr(sp, "get_ai_engine", lambda: engine)

    svc = sp.SuperpowersService()
    with pytest.raises(HTTPException):
        asyncio.run(svc.review_code("\n\t"))
