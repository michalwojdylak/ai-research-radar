import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from research_radar.config import Settings

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    async def generate_structured[T: BaseModel](
        self, *, system_prompt: str, user_prompt: str, schema: type[T]
    ) -> T: ...


class LLMResponseError(RuntimeError):
    """The model refused or did not return a complete, valid structured response."""


def fingerprint(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def request_key[T: BaseModel](
    *, system_prompt: str, user_prompt: str, schema: type[T], model: str, context: object
) -> str:
    return fingerprint(
        {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema": schema.model_json_schema(),
            "model": model,
            "context": context,
        }
    )


def validate_response[T: BaseModel](schema: type[T], value: object) -> T:
    try:
        payload = value.model_dump() if isinstance(value, BaseModel) else value
        return schema.model_validate(payload)
    except ValidationError as error:
        raise LLMResponseError(f"Invalid structured {schema.__name__} response") from error


async def read_cached[T: BaseModel](path: Path, schema: type[T]) -> T | None:
    try:
        text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return schema.model_validate_json(text)
    except FileNotFoundError:
        return None
    except (OSError, ValidationError) as error:
        logger.warning(
            "Structured cache unusable (%s); requesting fresh data", type(error).__name__
        )
        return None


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(f".{uuid4().hex}.part")
    try:
        staging.write_text(text, encoding="utf-8")
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


async def write_cached(path: Path, result: BaseModel) -> None:
    await asyncio.to_thread(_write, path, result.model_dump_json())


async def cached_generate[T: BaseModel](
    settings: Settings,
    provider: LLMProvider,
    *,
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    model: str,
    paper_id: str,
    stage: str,
    version: str,
) -> T:
    key = request_key(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=schema,
        model=model,
        context={"paper_id": paper_id, "stage": stage, "version": version},
    )
    path = settings.cache_dir / "llm" / f"{key}.json"
    cached = await read_cached(path, schema)
    if cached is not None:
        return cached
    result = validate_response(
        schema,
        await provider.generate_structured(
            system_prompt=system_prompt, user_prompt=user_prompt, schema=schema
        ),
    )
    await write_cached(path, result)
    return result


class OpenAIProvider:
    def __init__(self, settings: Settings, model: str) -> None:
        if settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required for OpenAIProvider")
        self.settings = settings
        self.model = model
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(), timeout=settings.http_timeout
        )

    async def generate_structured[T: BaseModel](
        self, *, system_prompt: str, user_prompt: str, schema: type[T]
    ) -> T:
        key = request_key(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            model=self.model,
            context="openai-structured-v1",
        )
        path = self.settings.cache_dir / "llm" / f"{key}.json"
        cached = await read_cached(path, schema)
        if cached is not None:
            return cached
        completion = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=schema,
        )
        if not completion.choices:
            raise LLMResponseError("OpenAI returned no completion choices")
        choice = completion.choices[0]
        if choice.message.refusal:
            raise LLMResponseError(
                f"OpenAI refused structured generation: {choice.message.refusal}"
            )
        if choice.finish_reason != "stop" or choice.message.parsed is None:
            raise LLMResponseError("OpenAI returned an incomplete or unparsed structured response")
        result = validate_response(schema, choice.message.parsed)
        await write_cached(path, result)
        return result

    async def aclose(self) -> None:
        await self.client.close()
