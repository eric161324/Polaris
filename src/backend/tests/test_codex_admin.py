import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from app.core.db import get_sessionmaker
from app.core.llm.base import CompletionResult
from app.core.llm.codex_cli import CodexCLIProvider
from app.core.llm.router import get_llm_router
from app.models.llm_config import LLMProviderConfig, LLMUsage
from tests.conftest import register_and_login


async def setup_provider(client):
    token = await register_and_login(client, email="codex@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    response = await client.post(
        "/api/admin/llm/providers",
        headers=headers,
        json={
            "name": "Codex",
            "kind": "codex_cli",
            "models": ["fixture-model"],
            "api_key": "must-not-be-stored",
            "base_url": "https://unused.example",
        },
    )
    assert response.status_code == 201, response.text
    provider = response.json()
    assert provider["api_key_masked"] == "" and provider["base_url"] is None
    return headers, provider["id"]


async def test_codex_config_routes_usage_and_probe(client, monkeypatch):
    headers, provider_id = await setup_provider(client)
    response = await client.put(
        "/api/admin/llm/routes",
        headers=headers,
        json=[
            {
                "stage": "default",
                "provider_id": provider_id,
                "model": "fixture-model",
            }
        ],
    )
    assert response.status_code == 200
    router = get_llm_router()
    for stage in ("agent", "librarian", "writing", "review"):
        provider, route = await router.resolve(stage)
        assert isinstance(provider, CodexCLIProvider) and route.api_key == ""
    for stage in ("embedding", "rerank"):
        response = await client.put(
            "/api/admin/llm/routes",
            headers=headers,
            json=[
                {
                    "stage": stage,
                    "provider_id": provider_id,
                    "model": "fixture-model",
                }
            ],
        )
        assert response.status_code == 400
        with pytest.raises(NotImplementedError):
            await router.resolve(stage)
    # Rejected route replacement must not delete the valid default route.
    assert (await client.get("/api/admin/llm/routes", headers=headers)).json()[0][
        "stage"
    ] == "default"
    monkeypatch.setattr(
        CodexCLIProvider,
        "complete",
        AsyncMock(
            return_value=CompletionResult(
                content="真实响应替身",
                model="fixture-model",
                usage={"prompt_tokens": 42, "completion_tokens": 9},
            )
        ),
    )
    result = await router.complete("writing", [])
    assert result.content == "真实响应替身"
    response = await client.post(
        "/api/admin/llm/test-model",
        headers=headers,
        json={
            "provider_id": provider_id,
            "model": "fixture-model",
            "capability": "chat",
        },
    )
    assert response.json()["ok"] is True
    async with get_sessionmaker()() as session:
        config = await session.get(LLMProviderConfig, uuid.UUID(provider_id))
        assert config.api_key_encrypted is None
        usage = (await session.scalars(select(LLMUsage))).all()
        assert len(usage) == 1 and usage[0].prompt_tokens == 42


async def test_login_endpoint_requires_auth_and_returns_no_credentials(client, monkeypatch):
    assert (await client.get("/api/admin/llm/codex/status")).status_code == 401
    headers, _ = await setup_provider(client)
    monkeypatch.setattr(
        CodexCLIProvider, "login_status", AsyncMock(return_value={"ok": True, "error": None})
    )
    response = await client.get("/api/admin/llm/codex/status", headers=headers)
    assert response.json() == {"ok": True, "error": None}


async def test_provider_conversion_cannot_break_embedding_routes(client):
    headers, codex_id = await setup_provider(client)
    response = await client.post(
        "/api/admin/llm/providers",
        headers=headers,
        json={
            "name": "Vectors",
            "kind": "openai_compat",
            "api_key": "sk-fixture-key",
        },
    )
    vector_id = response.json()["id"]
    response = await client.put(
        "/api/admin/llm/routes",
        headers=headers,
        json=[
            {"stage": "default", "provider_id": codex_id, "model": "fixture-model"},
            {"stage": "embedding", "provider_id": vector_id, "model": "embedding-model"},
            {"stage": "rerank", "provider_id": vector_id, "model": "rerank-model"},
        ],
    )
    assert response.status_code == 200
    response = await client.patch(
        f"/api/admin/llm/providers/{vector_id}", headers=headers, json={"kind": "codex_cli"}
    )
    assert response.status_code == 400
    _, route = await get_llm_router().resolve("embedding")
    assert route.provider_kind == "openai_compat" and route.api_key == "sk-fixture-key"


async def test_codex_remote_eval_does_not_export_credentials():
    from app.agents.voyage.actions_experiment import _eval_model_config_file

    llm = SimpleNamespace(
        resolve=AsyncMock(return_value=(None, SimpleNamespace(provider_kind="codex_cli")))
    )
    # _params reads the execution checkpoint.
    ctx = SimpleNamespace(llm=llm, checkpoint={"params": {"eval_model": "x"}})
    with pytest.raises(ValueError, match="CODEX_EVAL_API_UNSUPPORTED"):
        await _eval_model_config_file(ctx)
