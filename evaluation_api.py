"""FastAPI routes for provider-backed model evaluations."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncGenerator, Generator

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from ai_hub import AIServiceError, MINI_MODELS, ProviderConfig, complete_chat
from evaluation_store import (
    create_database_engine,
    create_session_factory,
    default_database_url,
    evaluation_to_dict,
    initialize_database,
    recent_evaluations,
    save_evaluation,
)


class EvaluationRequest(BaseModel):
    model: str
    prompt: str = Field(min_length=1, max_length=12_000)

    @field_validator("model")
    @classmethod
    def validate_model(cls, model: str) -> str:
        if model not in MINI_MODELS:
            supported = ", ".join(MINI_MODELS)
            raise ValueError(f"Choose one of the supported mini models: {supported}.")
        return model

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("The prompt cannot be empty.")
        return prompt


class EvaluationResponse(BaseModel):
    id: int
    model: str
    prompt: str
    response: str
    latency_ms: float
    created_at: str


def get_session(request: Request) -> Generator[Session, None, None]:
    with request.app.state.session_factory() as session:
        yield session


def create_app(database_url: str | None = None) -> FastAPI:
    configured_database_url = (
        database_url or os.getenv("EVALSPHERE_DATABASE_URL") or default_database_url()
    )
    engine = create_database_engine(configured_database_url)
    session_factory = create_session_factory(engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        initialize_database(engine)
        yield
        engine.dispose()

    api = FastAPI(
        title="EvalSphere evaluation API",
        version="1.0.0",
        lifespan=lifespan,
    )
    api.state.session_factory = session_factory

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.post("/api/evaluations", response_model=EvaluationResponse)
    def create_evaluation(
        request_body: EvaluationRequest,
        provider_api_key: str | None = Header(
            default=None,
            alias="X-Provider-API-Key",
        ),
        session: Session = Depends(get_session),
    ) -> dict[str, object]:
        api_key = provider_api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="Configure a provider API key before running an evaluation.",
            )

        config = ProviderConfig(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=request_body.model,
        )
        started = perf_counter()
        try:
            response = complete_chat(config=config, prompt=request_body.prompt)
        except AIServiceError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

        record = save_evaluation(
            session,
            model=request_body.model,
            prompt=request_body.prompt,
            response=response,
            latency_ms=(perf_counter() - started) * 1000,
        )
        return evaluation_to_dict(record)

    @api.get("/api/evaluations", response_model=list[EvaluationResponse])
    def list_evaluations(
        limit: int = Query(default=20, ge=1, le=50),
        session: Session = Depends(get_session),
    ) -> list[dict[str, object]]:
        return [
            evaluation_to_dict(record)
            for record in recent_evaluations(session, limit=limit)
        ]

    return api


app = create_app()
