import asyncio
import logging
from urllib.parse import unquote, urljoin, urlsplit
from uuid import uuid4

import httpx
import pymupdf
from pydantic import BaseModel

from research_radar.analysis.llm import fingerprint, read_cached, write_cached
from research_radar.config import Settings
from research_radar.models.paper import Paper

logger = logging.getLogger(__name__)
PDF_HOSTS = frozenset({"arxiv.org", "export.arxiv.org", "www.arxiv.org"})


class PaperText(BaseModel):
    text: str
    full_text_available: bool
    truncated: bool = False


def trusted_pdf_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in PDF_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or not parsed.path.startswith("/pdf/")
        or ".." in unquote(parsed.path).split("/")
    ):
        raise ValueError("PDF URL is not a trusted arXiv PDF endpoint")
    return parsed._replace(scheme="https", fragment="").geturl()


def extract_pdf(data: bytes, character_limit: int) -> PaperText:
    parts: list[str] = []
    length = 0
    with pymupdf.open(stream=data, filetype="pdf") as document:  # type: ignore[no-untyped-call]
        for page in document:
            text = page.get_text()
            parts.append(text[: max(0, character_limit + 1 - length)])
            length += len(parts[-1])
            if length > character_limit:
                break
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError("PDF contains no extractable text")
    return PaperText(
        text=text[:character_limit],
        full_text_available=True,
        truncated=len(text) > character_limit,
    )


class FullTextLoader:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client or httpx.AsyncClient(
            timeout=settings.http_timeout, follow_redirects=False
        )
        self._owns_client = client is None

    async def _download(self, url: str) -> bytes:
        for _ in range(6):
            url = trusted_pdf_url(url)
            async with self.client.stream(
                "GET", url, follow_redirects=False, headers={"Accept-Encoding": "identity"}
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if location is None:
                        raise ValueError("PDF redirect has no destination")
                    url = trusted_pdf_url(urljoin(url, location))
                    continue
                response.raise_for_status()
                declared = response.headers.get("content-length")
                if declared is not None and int(declared) > self.settings.max_pdf_bytes:
                    raise ValueError("PDF exceeds configured byte limit")
                data = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    if len(data) + len(chunk) > self.settings.max_pdf_bytes:
                        raise ValueError("PDF exceeds configured byte limit")
                    data.extend(chunk)
                if not data.lstrip().startswith(b"%PDF-"):
                    raise ValueError("PDF endpoint did not return a PDF")
                return bytes(data)
        raise ValueError("Too many PDF redirects")

    async def load(self, paper: Paper) -> PaperText:
        if not paper.pdf_url:
            logger.warning("Full text unavailable for %s: no PDF URL; using abstract", paper.id)
            return PaperText(text="", full_text_available=False)
        try:
            url = trusted_pdf_url(paper.pdf_url)
            identity = {
                "paper_id": paper.id,
                "updated_at": paper.updated_at,
                "pdf_url": paper.pdf_url,
                "max_bytes": self.settings.max_pdf_bytes,
            }
            pdf_path = self.settings.cache_dir / "pdf" / f"{fingerprint(identity)}.pdf"
            text_key = fingerprint(
                {
                    **identity,
                    "max_characters": self.settings.max_analysis_characters,
                    "extractor": "pymupdf-v1",
                }
            )
            text_path = self.settings.cache_dir / "pdf" / f"{text_key}.json"
            cached = await read_cached(text_path, PaperText)
            if cached is not None and cached.full_text_available and cached.text:
                return cached
            try:
                size = await asyncio.to_thread(lambda: pdf_path.stat().st_size)
                if size > self.settings.max_pdf_bytes:
                    raise ValueError("Cached PDF exceeds configured byte limit")
                data = await asyncio.to_thread(pdf_path.read_bytes)
            except FileNotFoundError:
                async with asyncio.timeout(self.settings.http_timeout):
                    data = await self._download(url)
                await asyncio.to_thread(pdf_path.parent.mkdir, parents=True, exist_ok=True)
                staging = pdf_path.with_suffix(f".{uuid4().hex}.part")
                try:
                    await asyncio.to_thread(staging.write_bytes, data)
                    await asyncio.to_thread(staging.replace, pdf_path)
                finally:
                    await asyncio.to_thread(staging.unlink, missing_ok=True)
            try:
                extracted = await asyncio.to_thread(
                    extract_pdf, data, self.settings.max_analysis_characters
                )
            except (RuntimeError, ValueError):
                await asyncio.to_thread(pdf_path.unlink, missing_ok=True)
                raise
            await write_cached(text_path, extracted)
            return extracted
        except (httpx.HTTPError, OSError, RuntimeError, ValueError) as error:
            logger.warning(
                "Full text unavailable for %s (%s); using abstract with reduced confidence",
                paper.id,
                type(error).__name__,
            )
            return PaperText(text="", full_text_available=False)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()
