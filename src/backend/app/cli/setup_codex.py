"""Idempotently add a Codex provider and an otherwise absent default route."""

import argparse
import asyncio

from sqlalchemy import select

import app.models  # noqa: F401 -- register all ORM relationships
from app.core.db import get_sessionmaker
from app.models.llm_config import LLMProviderConfig, ModelRoute


async def setup(model: str) -> None:
    async with get_sessionmaker()() as session:
        provider = await session.scalar(
            select(LLMProviderConfig).where(
                LLMProviderConfig.name == "Codex subscription",
                LLMProviderConfig.owner_id.is_(None),
            )
        )
        if provider is None:
            provider = LLMProviderConfig(
                name="Codex subscription",
                kind="codex_cli",
                models=[model],
                enabled=True,
            )
            session.add(provider)
            await session.flush()
        elif provider.kind != "codex_cli":
            raise SystemExit("The provider name is already in use; configure Codex in Settings.")
        route = await session.scalar(
            select(ModelRoute).where(
                ModelRoute.stage == "default",
                ModelRoute.owner_id.is_(None),
            )
        )
        if route is None:
            session.add(ModelRoute(stage="default", provider_id=provider.id, model=model))
            print("Created default Codex route. Embedding/rerank routes are unchanged.")
        else:
            print("Existing default route preserved. Select Codex in Settings if needed.")
        await session.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-6-astra")
    asyncio.run(setup(parser.parse_args().model))
