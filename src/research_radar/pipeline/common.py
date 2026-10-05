import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from xml.etree.ElementTree import ParseError

import httpx
from openai import OpenAIError
from pydantic import ValidationError

from research_radar.analysis.llm import LLMResponseError
from research_radar.models.run import PipelineFailure, RunReport

logger = logging.getLogger(__name__)
PAPER_ERRORS = (
    httpx.HTTPError,
    OpenAIError,
    LLMResponseError,
    ValidationError,
    ParseError,
    ValueError,
    OSError,
)


def report_failure(
    report: RunReport, stage: str, error: Exception, paper_id: str | None = None
) -> None:
    # Provider exception strings can contain request data or credentials; log only their type.
    kind = type(error).__name__
    message = f"{stage} failed with {kind}; inspect configuration and source availability."
    logger.warning("%s [%s]", message, paper_id or "pipeline")
    report.failures.append(
        PipelineFailure(stage=stage, paper_id=paper_id, error_type=kind, message=message)
    )


async def bounded_map[T, R](
    items: Sequence[T],
    operation: Callable[[T], Awaitable[R]],
    *,
    concurrency: int,
) -> list[R]:
    """Use a fixed worker pool rather than creating one task per paper."""
    queue: asyncio.Queue[tuple[int, T]] = asyncio.Queue()
    for index, item in enumerate(items):
        queue.put_nowait((index, item))
    results: dict[int, R] = {}

    async def worker() -> None:
        while not queue.empty():
            index, item = queue.get_nowait()
            try:
                results[index] = await operation(item)
            finally:
                queue.task_done()

    async with asyncio.TaskGroup() as group:
        for _ in range(min(concurrency, len(items))):
            group.create_task(worker())
    return [results[index] for index in range(len(items))]
