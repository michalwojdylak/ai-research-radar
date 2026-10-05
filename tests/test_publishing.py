import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from research_radar.config import Settings
from research_radar.publishing import create_review_pr


@pytest.mark.asyncio
async def test_dry_run_performs_no_github_write(tmp_path: Path) -> None:
    def unexpected(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected request: {request.method}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
        result = await create_review_pr(
            Settings(_env_file=None), client, [tmp_path / "draft.json"], now=datetime.now(UTC)
        )
    assert result is None


@pytest.mark.asyncio
async def test_submission_requires_explicit_credentials() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="GITHUB_REPOSITORY"):
            await create_review_pr(
                Settings(_env_file=None, dry_run=False), client, [], now=datetime.now(UTC)
            )


@pytest.mark.asyncio
async def test_submission_opens_draft_pr_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data" / "papers" / "2026" / "10" / "test.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"status":"review","michaels_take":null}', encoding="utf-8")
    writes: list[tuple[str, str, object]] = []

    def response(request: httpx.Request) -> httpx.Response:
        suffix = request.url.path.removeprefix("/repos/owner/repo")
        body = json.loads(request.content) if request.content else None
        if request.method != "GET":
            writes.append((request.method, suffix, body))
        if suffix == "/git/ref/heads/research-radar/2026-10-05":
            return httpx.Response(404)
        if suffix == "/git/ref/heads/master":
            return httpx.Response(200, json={"object": {"sha": "base"}})
        if suffix == "/git/commits/base":
            return httpx.Response(200, json={"sha": "base", "tree": {"sha": "base-tree"}})
        if suffix == "/git/trees/base-tree":
            return httpx.Response(200, json={"tree": [], "truncated": False})
        if suffix in {"/git/blobs", "/git/trees", "/git/commits"}:
            return httpx.Response(201, json={"sha": "new-sha"})
        if suffix == "/git/refs":
            return httpx.Response(201, json={})
        if suffix == "/pulls" and request.method == "GET":
            return httpx.Response(200, json=[])
        if suffix == "/pulls" and request.method == "POST":
            return httpx.Response(201, json={"html_url": "https://github.com/owner/repo/pull/1"})
        raise AssertionError(f"Unexpected request {request.method} {suffix}")

    settings = Settings(
        _env_file=None,
        dry_run=False,
        github_repository="owner/repo",
        github_token="test-not-a-secret",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        url = await create_review_pr(
            settings, client, [path], now=datetime(2026, 10, 5, tzinfo=UTC)
        )
    assert url == "https://github.com/owner/repo/pull/1"
    pr = next(body for _, suffix, body in writes if suffix == "/pulls")
    assert isinstance(pr, dict) and pr["draft"] is True
    assert all("merge" not in suffix and "reviews" not in suffix for _, suffix, _ in writes)
