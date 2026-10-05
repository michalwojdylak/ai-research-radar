"""Bounded, retrying HTTP access with a credential-free public response cache."""

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import time
import uuid
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from xml.etree import ElementTree

import httpx
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt

from research_radar.collectors.schemas import validate_provider_response
from research_radar.config import Settings

logger = logging.getLogger(__name__)
MAX_RESPONSE_BYTES = 5_000_000
_HOSTS = {
    "export.arxiv.org",
    "api.openalex.org",
    "api.semanticscholar.org",
    "api.github.com",
    "huggingface.co",
}
_AUTH_PARAMS = {"token", "access_token", "api_key", "apikey", "key", "authorization"}
NETWORK_ERRORS = (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError)


def _transient(error: BaseException) -> bool:
    return isinstance(error, NETWORK_ERRORS) or (
        isinstance(error, httpx.HTTPStatusError)
        and (error.response.status_code == 429 or 500 <= error.response.status_code < 600)
    )


def _retry_delay(state: RetryCallState) -> float:
    delay = min(0.5 * 2 ** (state.attempt_number - 1), 4.0)
    error = state.outcome.exception() if state.outcome else None
    if isinstance(error, httpx.HTTPStatusError):
        retry_after = error.response.headers.get("retry-after")
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                try:
                    delay = (parsedate_to_datetime(retry_after) - datetime.now(UTC)).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    pass
    return float(min(max(delay, 0.0), 30.0)) if math.isfinite(delay) else 0.5


class CachedHTTP:
    """Only public API hosts are fetched; authenticated responses are never persisted.

    The caller owns the AsyncClient. Redirects are disabled, including redirects
    to private hosts. Retries apply only to transport failures, 429, and 5xx.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client
        self._semaphore = asyncio.Semaphore(settings.concurrency)
        self._arxiv_lock = asyncio.Lock()
        self._last_arxiv_request: float | None = None

    async def get(
        self,
        url: str,
        *,
        params: dict[str, str | int] | None = None,
        headers: dict[str, str] | None = None,
        ttl: int | None = None,
    ) -> httpx.Response:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in _HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in {None, 443}
            or parsed.fragment
        ):
            raise ValueError("Only official HTTPS API hosts without credentials are allowed")
        request = self.client.build_request("GET", url, params=params, headers=headers)
        if any(name.lower() in _AUTH_PARAMS for name in request.url.params):
            raise ValueError("API credentials must be supplied through headers")
        # Client-level credentials count too. Cache entries must not cross auth scopes.
        cacheable = self.client.auth is None and not any(
            name.lower() in {"authorization", "x-api-key", "cookie"} for name in request.headers
        )
        lifetime = self.settings.http_cache_ttl if ttl is None else max(ttl, 0)
        representation = {
            "url": str(request.url),
            "accept": request.headers.get("accept", ""),
            "api_version": request.headers.get("x-github-api-version", ""),
        }
        key = hashlib.sha256(json.dumps(representation, sort_keys=True).encode()).hexdigest()
        path = self.settings.cache_dir / "http" / f"{key}.json"
        if cacheable and lifetime > 0:
            cached = await asyncio.to_thread(self._read_cache, path, lifetime, request)
            if cached is not None:
                return cached

        async with self._semaphore:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_transient),
                stop=stop_after_attempt(3),
                wait=_retry_delay,
                reraise=True,
            ):
                with attempt:
                    if parsed.hostname == "export.arxiv.org":
                        async with self._arxiv_lock:
                            await self._arxiv_wait()
                            response = await self._request(url, params, headers)
                    else:
                        response = await self._request(url, params, headers)

        self._validate_response(response)
        if cacheable and lifetime > 0:
            await asyncio.to_thread(self._write_cache, path, response)
        return response

    async def _arxiv_wait(self) -> None:
        interval = max(self.settings.arxiv_request_interval, 3.1)
        if self._last_arxiv_request is not None:
            delay = interval - (time.monotonic() - self._last_arxiv_request)
            if delay > 0:
                await asyncio.sleep(delay)
        self._last_arxiv_request = time.monotonic()

    async def _request(
        self,
        url: str,
        params: dict[str, str | int] | None,
        headers: dict[str, str] | None,
    ) -> httpx.Response:
        try:
            async with asyncio.timeout(self.settings.http_timeout):
                return await self._stream(url, params, headers)
        except TimeoutError as error:
            raise httpx.ReadTimeout(
                "HTTP request exceeded total timeout", request=httpx.Request("GET", url)
            ) from error

    async def _stream(
        self,
        url: str,
        params: dict[str, str | int] | None,
        headers: dict[str, str] | None,
    ) -> httpx.Response:
        async with self.client.stream(
            "GET",
            url,
            params=params,
            headers=headers,
            timeout=self.settings.http_timeout,
            follow_redirects=False,
        ) as streamed:
            streamed.raise_for_status()
            length = streamed.headers.get("content-length")
            if length is not None and int(length) > MAX_RESPONSE_BYTES:
                raise ValueError("HTTP response exceeds size limit")
            body = bytearray()
            async for chunk in streamed.aiter_bytes():
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise ValueError("HTTP response exceeds size limit")
            # Do not retain credential-bearing request or response headers.
            safe_headers = {
                key: value
                for key, value in streamed.headers.items()
                if key in {"content-type", "link"}
            }
            return httpx.Response(
                streamed.status_code,
                content=bytes(body),
                headers=safe_headers,
                request=httpx.Request("GET", streamed.url),
            )

    @staticmethod
    def _validate_response(response: httpx.Response) -> None:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("Expected HTTP 200 with provider data")
        if response.request.url.host == "export.arxiv.org":
            root = ElementTree.fromstring(response.content)
            if root.tag != "{http://www.w3.org/2005/Atom}feed":
                raise ValueError("Expected an Atom feed")
            for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                identifier = entry.findtext("{http://www.w3.org/2005/Atom}id", "")
                if "/api/errors" in identifier:
                    raise ValueError("arXiv API returned an error feed")
        else:
            try:
                data = response.json()
            except RecursionError as error:
                raise ValueError("Provider JSON nesting exceeds parser limit") from error
            if not isinstance(data, (dict, list)):
                raise ValueError("Expected a JSON object or array")
            if isinstance(data, dict) and (
                data.get("error") or data.get("errors") or data.get("private") is True
            ):
                raise ValueError("Provider returned an error or private resource")
            validate_provider_response(response)

    @classmethod
    def _read_cache(cls, path: Path, ttl: int, request: httpx.Request) -> httpx.Response | None:
        try:
            if not path.is_file() or path.stat().st_size > MAX_RESPONSE_BYTES * 2:
                return None
            envelope = json.loads(path.read_bytes())
            if not isinstance(envelope, dict):
                return None
            age = time.time() - float(envelope["saved_at"])
            if not 0 <= age < ttl:
                return None
            body = base64.b64decode(envelope["body"], validate=True)
            if len(body) > MAX_RESPONSE_BYTES:
                return None
            response = httpx.Response(
                200,
                content=body,
                headers=envelope["headers"],
                request=httpx.Request("GET", request.url),
            )
            cls._validate_response(response)
            return response
        except (
            OSError,
            RecursionError,
            ValueError,
            TypeError,
            KeyError,
            ElementTree.ParseError,
            httpx.HTTPError,
        ):
            logger.debug("http cache read unavailable")
            return None

    @staticmethod
    def _write_cache(path: Path, response: httpx.Response) -> None:
        staging: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            envelope: dict[str, Any] = {
                "saved_at": time.time(),
                "headers": dict(response.headers),
                "body": base64.b64encode(response.content).decode("ascii"),
            }
            staging = path.with_suffix(f".{uuid.uuid4().hex}.part")
            fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(envelope, handle, separators=(",", ":"))
            staging.replace(path)
        except OSError:
            logger.debug("http cache write unavailable")
        finally:
            if staging is not None:
                try:
                    staging.unlink(missing_ok=True)
                except OSError:
                    logger.debug("http cache cleanup unavailable")
