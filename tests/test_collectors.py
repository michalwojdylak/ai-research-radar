import asyncio
import base64
import json
import shutil
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import httpx
import pytest
from pydantic import SecretStr, ValidationError

import research_radar.collectors.github as github_module
import research_radar.collectors.http as http_module
from research_radar.collectors.arxiv import ARXIV_API, ArxivCollector
from research_radar.collectors.common import github_repository_url, merge_enrichment
from research_radar.collectors.github import GitHubCollector
from research_radar.collectors.http import CachedHTTP
from research_radar.collectors.huggingface import HuggingFaceCollector
from research_radar.collectors.openalex import OpenAlexCollector
from research_radar.collectors.semantic_scholar import SemanticScholarCollector
from research_radar.config import Settings
from research_radar.enrichment import EnrichmentPipeline
from research_radar.models.paper import Enrichment, Paper


@pytest.fixture
def settings() -> Iterator[Settings]:
    # Scratch/cache data stays inside the project, not the OS temporary directory.
    directory = Path(".cache/collector-tests") / uuid.uuid4().hex
    yield Settings(_env_file=None, cache_dir=directory, http_cache_ttl=0)
    if directory.exists():
        shutil.rmtree(directory)


@pytest.fixture
def paper() -> Paper:
    return Paper(
        id="2501.12345",
        arxiv_id="2501.12345v2",
        title="A useful AI method",
        abstract="A new method.",
        authors=["A. Researcher"],
        published_at=datetime(2025, 1, 3, tzinfo=UTC),
        categories=["cs.AI"],
        abs_url="https://arxiv.org/abs/2501.12345",
    )


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_module, "_retry_delay", lambda state: 0)


def _entry(
    identifier: str,
    *,
    published: str = "2025-01-03T12:00:00Z",
    updated: str = "2025-01-03T12:00:00Z",
    category: str = "cs.AI",
    abstract: str = "Method description.",
) -> str:
    return f"""
      <entry>
        <id>http://arxiv.org/abs/{identifier}</id>
        <title> A useful
        AI method </title>
        <summary>{escape(abstract)}</summary>
        <published>{published}</published>
        <updated>{updated}</updated>
        <author><name>A. Researcher</name></author>
        <category term="{category}"/>
        <link href="http://arxiv.org/pdf/{identifier}" type="application/pdf"/>
        <arxiv:doi>10.1000/example</arxiv:doi>
      </entry>
    """


def _atom(entries: list[str], total: int | None = None) -> str:
    count = f"<opensearch:totalResults>{total}</opensearch:totalResults>" if total else ""
    return (
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" '
        'xmlns:arxiv="http://arxiv.org/schemas/atom">'
        f"<title>arXiv results</title>{count}{''.join(entries)}</feed>"
    )


@pytest.mark.asyncio
async def test_arxiv_paginates_normalizes_versions_and_filters_utc(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.arxiv_page_size = 2
    calls: list[httpx.Request] = []

    async def no_spacing(self: CachedHTTP) -> None:
        pass

    monkeypatch.setattr(CachedHTTP, "_arxiv_wait", no_spacing)

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        offset = int(request.url.params["start"])
        pages = {
            0: [_entry("2501.12345v1"), _entry("cs/9901001v3")],
            2: [
                _entry("2501.12345v2", updated="2025-01-04T13:00:00Z"),
                _entry("2501.99999", category="physics.optics"),
            ],
        }
        return httpx.Response(
            200,
            text=_atom(pages[offset], total=4),
            headers={"Content-Type": "application/atom+xml"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        papers = await ArxivCollector(settings, client).discover(
            days=7, now=datetime.fromisoformat("2025-01-05T14:00:00+02:00")
        )
    assert {item.id for item in papers} == {"2501.12345", "cs/9901001"}
    modern = next(item for item in papers if item.id == "2501.12345")
    assert modern.title == "A useful AI method"
    assert modern.arxiv_id == "2501.12345v2"
    assert modern.updated_at == datetime(2025, 1, 4, 13, tzinfo=UTC)
    assert modern.pdf_url == "https://arxiv.org/pdf/2501.12345v2"
    assert modern.doi == "10.1000/example"
    assert [request.url.params["start"] for request in calls] == ["0", "2"]
    query = calls[0].url.params["search_query"]
    assert "cat:cs.AI OR cat:cs.LG" in query
    assert "submittedDate:[202412291200 TO 202501051200]" in query


@pytest.mark.asyncio
async def test_arxiv_skips_bad_entries_and_enforces_exact_date_window(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(CachedHTTP, "_arxiv_wait", _no_spacing)
    body = _atom(
        [
            _entry("2501.12345", published="2025-01-03T12:00:00-02:00"),
            _entry("2501.00001", published="2024-12-28T11:59:59Z"),
            _entry("2501.00002", published="2025-01-05T12:00:01Z"),
            _entry("2501.00003", published="not-a-date"),
        ],
        total=4,
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body))
    ) as client:
        papers = await ArxivCollector(settings, client).discover(
            now=datetime(2025, 1, 5, 12, tzinfo=UTC)
        )
    assert len(papers) == 1
    assert papers[0].published_at == datetime(2025, 1, 3, 14, tzinfo=UTC)


async def _no_spacing(self: CachedHTTP) -> None:
    pass


@pytest.mark.asyncio
async def test_arxiv_preserves_highest_version_on_equal_dates_without_pdf_links(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(CachedHTTP, "_arxiv_wait", _no_spacing)
    newest = _entry("2501.12345v10").replace(
        '<link href="http://arxiv.org/pdf/2501.12345v10" type="application/pdf"/>', ""
    )
    body = _atom([_entry("2501.12345v2"), newest], total=2)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body))
    ) as client:
        papers = await ArxivCollector(settings, client).discover(
            now=datetime(2025, 1, 5, 12, tzinfo=UTC)
        )
    assert len(papers) == 1
    assert papers[0].id == "2501.12345"
    assert papers[0].arxiv_id == "2501.12345v10"
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2501.12345v10"


@pytest.mark.asyncio
async def test_arxiv_spaces_every_network_attempt_at_least_3_point_1_seconds(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [0.0]
    sleeps: list[float] = []
    monkeypatch.setattr(http_module.time, "monotonic", lambda: clock[0])

    async def sleep(delay: float) -> None:
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(http_module.asyncio, "sleep", sleep)
    attempts = 0

    def handle(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, text=_atom([]))

    settings.arxiv_request_interval = 3.0
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        http = CachedHTTP(settings, client)
        await http.get(ARXIV_API)
        await http.get(ARXIV_API)
    assert attempts == 3
    assert [delay for delay in sleeps if delay > 0] == [3.1, 3.1]


@pytest.mark.asyncio
@pytest.mark.parametrize("first", [429, 503, "network"])
async def test_http_retries_only_transient_failures(settings: Settings, first: int | str) -> None:
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            if first == "network":
                raise httpx.ConnectError("offline", request=request)
            return httpx.Response(int(first))
        return httpx.Response(200, json={"paperId": "abc"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        response = await CachedHTTP(settings, client).get(
            "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2501.12345"
        )
    assert attempts == 2
    assert response.json() == {"paperId": "abc"}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_http_does_not_retry_or_cache_permanent_errors(
    settings: Settings, status: int
) -> None:
    settings.http_cache_ttl = 60
    attempts = 0

    def handle(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(status, json={"error": "missing"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await CachedHTTP(settings, client).get("https://api.github.com/repos/org/repo")
    assert attempts == 1
    assert not settings.cache_dir.exists()


@pytest.mark.asyncio
async def test_http_public_cache_and_authenticated_cache_isolation(settings: Settings) -> None:
    settings.http_cache_ttl = 60
    attempts = 0

    def handle(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            json={"full_name": "org/repo", "stargazers_count": attempts},
            headers={"Set-Cookie": "private-cookie"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        http = CachedHTTP(settings, client)
        first = await http.get("https://api.github.com/repos/org/repo")
        client.cookies.clear()
        second = await http.get("https://api.github.com/repos/org/repo")
        authenticated = await http.get(
            "https://api.github.com/repos/org/repo", headers={"Authorization": "Bearer test-only"}
        )
    assert first.json() == second.json()
    assert authenticated.json()["stargazers_count"] == 2
    assert attempts == 2
    cache_files = list(settings.cache_dir.rglob("*.json"))
    assert len(cache_files) == 1
    cache_text = cache_files[0].read_text()
    assert "test-only" not in cache_text
    assert "private-cookie" not in cache_text
    assert "authorization" not in cache_text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [b"broken JSON", b"null", b'{"error":"bad"}'])
async def test_http_validates_body_before_caching(settings: Settings, content: bytes) -> None:
    settings.http_cache_ttl = 60
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content))
    ) as client:
        with pytest.raises(ValueError):
            await CachedHTTP(settings, client).get("https://api.openalex.org/works/W1")
    assert not settings.cache_dir.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/data",
        "https://api.github.com.evil.example/repos/org/repo",
        "http://api.github.com/repos/org/repo",
        "https://user:secret@api.github.com/repos/org/repo",
        "https://api.github.com/repos/org/repo?api_key=test-only",
    ],
)
async def test_http_rejects_private_hosts_credentials_and_query_auth(
    settings: Settings, url: str
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("Must not send a request"))
    ) as client:
        with pytest.raises(ValueError):
            await CachedHTTP(settings, client).get(url)


@pytest.mark.asyncio
async def test_http_rejects_large_and_redirected_responses(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(http_module, "MAX_RESPONSE_BYTES", 10)
    for response in (
        httpx.Response(200, content=b'{"value":"a too large body"}'),
        httpx.Response(302, headers={"Location": "http://127.0.0.1/private"}),
    ):
        async with httpx.AsyncClient(
            follow_redirects=True,
            transport=httpx.MockTransport(lambda _, response=response: response),
        ) as client:
            with pytest.raises((ValueError, httpx.HTTPStatusError)):
                await CachedHTTP(settings, client).get("https://api.github.com/repos/org/repo")


@pytest.mark.asyncio
async def test_openalex_exact_doi_and_topical_related_works(
    settings: Settings, paper: Paper
) -> None:
    paper.doi = "https://doi.org/10.1000/Example"
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/W2"):
            return httpx.Response(
                200,
                json={
                    "id": "https://openalex.org/W2",
                    "display_name": "Related",
                    "publication_year": 2025,
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.1000/example",
                "cited_by_count": None,
                "topics": [{"display_name": "Machine learning"}],
                "primary_location": None,
                "related_works": ["https://openalex.org/W2", "https://127.0.0.1/private"],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await OpenAlexCollector(settings, client).enrich(paper)
    assert result.openalex_citations is None
    assert result.topics == ["Machine learning"]
    assert result.related_works[0].title == "Related"
    assert len(requests) == 2
    assert "https://doi.org/10.1000/example" in requests[0].url.path
    assert all("search" not in request.url.params for request in requests)


@pytest.mark.asyncio
async def test_openalex_rejects_unrelated_work_and_accepts_exact_arxiv_location(
    settings: Settings, paper: Paper
) -> None:
    for location, accepted in (
        ("https://arxiv.org/abs/2501.00001", False),
        ("https://arxiv.org/abs/2501.12345v4", True),
    ):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _, location=location: httpx.Response(
                    200,
                    json={
                        "id": "https://openalex.org/W1",
                        "doi": None,
                        "locations": [{"landing_page_url": location}],
                        "cited_by_count": 7,
                    },
                )
            )
        ) as client:
            if accepted:
                result = await OpenAlexCollector(settings, client).enrich(paper)
                assert result.openalex_citations == 7
            else:
                with pytest.raises(ValueError, match="matching DOI"):
                    await OpenAlexCollector(settings, client).enrich(paper)


@pytest.mark.asyncio
async def test_semantic_scholar_missing_null_and_strict_counter_types(
    settings: Settings, paper: Paper
) -> None:
    bodies = (
        {"paperId": "abc", "citationCount": None, "influentialCitationCount": None},
        {"paperId": "abc", "citationCount": "22"},
        {"paperId": "abc", "citationCount": -1},
        {"paperId": "abc", "externalIds": {"ArXiv": "2501.00001"}},
    )
    for index, body in enumerate(bodies):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _, body=body: httpx.Response(200, json=body))
        ) as client:
            collector = SemanticScholarCollector(settings, client)
            if index == 0:
                result = await collector.enrich(paper)
                assert result.citations is None
                assert result.influential_citations is None
            else:
                with pytest.raises((ValidationError, ValueError)):
                    await collector.enrich(paper)


@pytest.mark.asyncio
async def test_huggingface_uses_only_exposed_attention_fields(
    settings: Settings, paper: Paper
) -> None:
    bodies: tuple[dict[str, Any], ...] = (
        {"paper": {"id": "2501.12345", "upvotes": 0}},
        {
            "paper": {"id": "2501.12345v2", "upvotes": 8},
            "models": ["org/model", {"id": "org/other"}],
            "datasets": [],
            "isTrending": False,
        },
    )
    for index, body in enumerate(bodies):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _, body=body: httpx.Response(200, json=body))
        ) as client:
            result = await HuggingFaceCollector(settings, client).enrich(paper)
        if index == 0:
            assert result.hf_upvotes == 0
            assert result.hf_models is None
            assert result.hf_datasets is None
            assert result.hf_trending is None
        else:
            assert result.hf_upvotes == 8
            assert result.hf_models == ["org/model", "org/other"]
            assert result.hf_datasets == []
            assert result.hf_trending is False


def test_merging_never_overwrites_available_fields_with_none() -> None:
    result = merge_enrichment(
        Enrichment(openalex_citations=5, hf_models=["org/one"], topics=["AI"], venue="Venue"),
        Enrichment(
            semantic_scholar_citations=7,
            hf_models=["org/one", "org/two"],
            topics=["AI", "ML"],
        ),
    )
    assert result.citations == 7
    assert result.venue == "Venue"
    assert result.topics == ["AI", "ML"]
    assert result.hf_models == ["org/one", "org/two"]
    assert merge_enrichment(Enrichment(), Enrichment(hf_models=["org/one"])).hf_models == [
        "org/one"
    ]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/lab/repo", "https://github.com/lab/repo"),
        ("http://github.com/lab/repo.git", "https://github.com/lab/repo"),
        ("https://github.com/lab/repo/tree/main", "https://github.com/lab/repo"),
        ("https://github.com.evil.example/lab/repo", None),
        ("https://github.com@127.0.0.1/lab/repo", None),
        ("https://github.com:bad/lab/repo", None),
        ("https://github.com/topics/machine-learning", None),
        ("https://github.com/lab/../private", None),
    ],
)
def test_github_url_validation(url: str, expected: str | None) -> None:
    assert github_repository_url(url) == expected


@pytest.mark.parametrize(
    ("text", "matches"),
    [
        ("See arXiv:2501.12345.", True),
        ("https://arxiv.org/abs/2501.12345v3", True),
        ("https://doi.org/10.48550/arXiv.2501.12345", True),
        ("Same title but no identifier", False),
        ("2501.123450", False),
        ("2501.12345-extra", False),
        ("2501.12345v3other", False),
    ],
)
def test_github_readme_requires_exact_identifier(text: str, matches: bool, paper: Paper) -> None:
    assert github_module._readme_matches(text, paper) is matches


def test_github_readme_accepts_exact_legacy_id_and_doi_only_papers(paper: Paper) -> None:
    legacy = paper.model_copy(update={"id": "cs/9901001", "arxiv_id": "cs/9901001v2"})
    assert github_module._readme_matches("https://arxiv.org/abs/cs%2F9901001v3.", legacy)
    doi_only = paper.model_copy(
        update={
            "id": "doi:10.1000/example",
            "arxiv_id": None,
            "abs_url": None,
            "doi": "https://doi.org/10.1000/example",
        }
    )
    assert github_module._readme_matches("Read https://doi.org/10.1000/EXAMPLE.", doi_only)
    assert not github_module._readme_matches("https://doi.org/10.1000/example-other", doi_only)


def _readme_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"content": base64.b64encode(text.encode()).decode(), "encoding": "base64"},
    )


@pytest.mark.asyncio
async def test_github_search_requires_exact_readme_and_counts_real_pages(
    settings: Settings, paper: Paper
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/search/repositories":
            return httpx.Response(
                200,
                json={"items": [{"full_name": "lab/decoy"}, {"full_name": "lab/real"}]},
            )
        if request.url.path == "/repos/lab/decoy/readme":
            return _readme_response("A useful AI method: arxiv:2501.123450. Not this paper.")
        if request.url.path == "/repos/lab/real/readme":
            return _readme_response("Paper: https://arxiv.org/abs/2501.12345v2")
        if request.url.path == "/repos/lab/real":
            return httpx.Response(
                200,
                json={
                    "full_name": "lab/real",
                    "stargazers_count": 12,
                    "forks_count": None,
                    "pushed_at": "2025-01-04T12:00:00Z",
                },
            )
        if request.url.path.endswith("/contributors"):
            page = int(request.url.params["page"])
            if page == 1:
                return httpx.Response(
                    200,
                    json=[{"id": index} for index in range(1, 101)],
                    headers={"Link": '<http://127.0.0.1/private>; rel="next"'},
                )
            return httpx.Response(200, json=[{"id": 101}, {"id": 102}])
        if request.url.path.endswith("/releases"):
            return httpx.Response(200, json=[{"id": 7}, {"id": 8}])
        pytest.fail(f"Unexpected endpoint {request.url.path}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await GitHubCollector(settings, client).enrich(paper)
    assert result.github_url == "https://github.com/lab/real"
    assert result.github_stars == 12
    assert result.github_forks is None
    assert result.github_contributors == 102
    assert result.github_releases == 2
    assert result.github_updated_at == datetime(2025, 1, 4, 12, tzinfo=UTC)
    assert result.repository_evidence is not None
    assert "exact paper" in result.repository_evidence.reason
    assert requests[0].url.params["q"] == '"2501.12345" in:readme'
    assert all(request.url.host == "api.github.com" for request in requests)


@pytest.mark.asyncio
async def test_github_abstract_link_is_strong_evidence_and_missing_metrics_stay_none(
    settings: Settings, paper: Paper
) -> None:
    paper.abstract = "Code (https://github.com/lab/real)."

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path != "/search/repositories"
        if request.url.path == "/repos/lab/real":
            return httpx.Response(200, json={"full_name": "lab/real"})
        if request.url.path.endswith("/readme"):
            return _readme_response("A repository without a paper identifier.")
        if request.url.path.endswith("/contributors"):
            return httpx.Response(404)
        return httpx.Response(200, json=[])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await GitHubCollector(settings, client).enrich(paper)
    assert result.github_url == "https://github.com/lab/real"
    assert result.github_stars is None
    assert result.github_contributors is None
    assert result.github_releases == 0
    assert result.repository_evidence and "abstract" in result.repository_evidence.reason


@pytest.mark.asyncio
async def test_github_pagination_cap_returns_unknown_not_partial_count(
    settings: Settings, paper: Paper, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(github_module, "MAX_COUNT_PAGES", 2)
    paper.abstract = "https://github.com/lab/real"

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/lab/real":
            return httpx.Response(200, json={"full_name": "lab/real"})
        if request.url.path.endswith("/readme"):
            return httpx.Response(404)
        if request.url.path.endswith("/releases"):
            return httpx.Response(200, json=[])
        page = int(request.url.params["page"])
        return httpx.Response(
            200,
            json=[{"id": index} for index in range(page * 100, (page + 1) * 100)],
            headers={"Link": '<https://api.github.com/next>; rel="next"'},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await GitHubCollector(settings, client).enrich(paper)
    assert result.github_contributors is None
    assert result.github_releases == 0


@pytest.mark.asyncio
async def test_pipeline_isolates_source_errors_without_logging_secrets(
    settings: Settings, paper: Paper, caplog: pytest.LogCaptureFixture
) -> None:
    settings.semantic_scholar_api_key = SecretStr("test-secret-only")
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "api.openalex.org":
            return httpx.Response(503, json={"error": "test-secret-only"})
        if request.url.host == "api.semanticscholar.org":
            assert request.headers["x-api-key"] == "test-secret-only"
            return httpx.Response(
                200, json={"paperId": "abc", "citationCount": 42, "venue": "Venue"}
            )
        if request.url.host == "huggingface.co":
            return httpx.Response(200, json={"paper": {"id": "2501.12345", "upvotes": -1}})
        return httpx.Response(200, json={"items": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await EnrichmentPipeline(settings, client).enrich(paper)
    assert result.citations == 42
    assert result.venue == "Venue"
    assert result.hf_upvotes is None
    assert len([request for request in requests if request.url.host == "api.openalex.org"]) == 3
    assert "source=openalex type=HTTPStatusError status=503" in caplog.text
    assert "source=huggingface type=ValidationError" in caplog.text
    assert "test-secret-only" not in caplog.text


@pytest.mark.asyncio
async def test_pipeline_metadata_repository_link_is_validated_then_enriched(
    settings: Settings, paper: Paper
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host in {"api.openalex.org", "api.semanticscholar.org"}:
            return httpx.Response(404)
        if request.url.host == "huggingface.co":
            return httpx.Response(
                200,
                json={
                    "paper": {
                        "id": "2501.12345",
                        "upvotes": 5,
                        "githubRepo": "https://github.com/lab/real",
                    }
                },
            )
        if request.url.path == "/repos/lab/real":
            return httpx.Response(200, json={"full_name": "lab/real", "stargazers_count": 10})
        if request.url.path.endswith("/readme"):
            return httpx.Response(404)
        if request.url.path.endswith(("/contributors", "/releases")):
            return httpx.Response(200, json=[])
        pytest.fail("No repository search is needed for an explicit metadata link")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await EnrichmentPipeline(settings, client).enrich(paper)
    assert result.hf_upvotes == 5
    assert result.github_stars == 10
    assert result.repository_evidence
    assert result.repository_evidence.evidence_url == "https://huggingface.co/papers/2501.12345"


@pytest.mark.asyncio
async def test_http_bounds_concurrent_requests(settings: Settings) -> None:
    settings.concurrency = 2
    active = 0
    maximum = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.001)
        active -= 1
        return httpx.Response(
            200, json={"id": f"https://openalex.org/{request.url.path.split('/')[-1]}"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        http = CachedHTTP(settings, client)
        await asyncio.gather(*(http.get(f"https://api.openalex.org/works/W{i}") for i in range(8)))
    assert maximum == 2


@pytest.mark.asyncio
async def test_http_corrupt_and_expired_cache_are_refetched(settings: Settings) -> None:
    settings.http_cache_ttl = 60
    attempts = 0

    def handle(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json={"id": "https://openalex.org/W1"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        http = CachedHTTP(settings, client)
        await http.get("https://api.openalex.org/works/W1")
        cache_file = next(settings.cache_dir.rglob("*.json"))
        cache_file.write_text("{bad json")
        await http.get("https://api.openalex.org/works/W1")
        envelope = json.loads(cache_file.read_text())
        envelope["saved_at"] = (datetime.now(UTC) - timedelta(seconds=100)).timestamp()
        cache_file.write_text(json.dumps(envelope))
        await http.get("https://api.openalex.org/works/W1")
    assert attempts == 3


@pytest.mark.asyncio
async def test_http_provider_validation_precedes_cache_write(settings: Settings) -> None:
    settings.http_cache_ttl = 60
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"paperId": "abc", "citationCount": -1})
        )
    ) as client:
        with pytest.raises(ValidationError):
            await CachedHTTP(settings, client).get(
                "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2501.12345"
            )
    assert not settings.cache_dir.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "body"),
    [
        (
            "https://api.openalex.org/works/W1",
            {"id": "https://openalex.org/W2"},
        ),
        (
            "https://huggingface.co/api/papers/2501.12345",
            {"paper": {"id": "2501.00001", "upvotes": 12}},
        ),
        (
            "https://api.github.com/repos/lab/real/readme",
            {"content": "not-base64!", "encoding": "base64"},
        ),
    ],
)
async def test_http_mismatched_identity_and_invalid_readme_are_not_cached(
    settings: Settings, url: str, body: dict[str, Any]
) -> None:
    settings.http_cache_ttl = 60
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    ) as client:
        with pytest.raises(ValueError):
            await CachedHTTP(settings, client).get(url)
    assert not settings.cache_dir.exists()


class _SlowStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        await asyncio.sleep(0.1)
        yield b'{"id":"https://openalex.org/W1"}'


@pytest.mark.asyncio
async def test_http_total_stream_timeout_is_bounded_and_retried(settings: Settings) -> None:
    settings.http_timeout = 0.005
    attempts = 0

    def handle(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, stream=_SlowStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(httpx.ReadTimeout):
            await CachedHTTP(settings, client).get("https://api.openalex.org/works/W1")
    assert attempts == 3


@pytest.mark.asyncio
async def test_pipeline_does_not_swallow_programming_errors(
    settings: Settings, paper: Paper, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(self: OpenAlexCollector, paper: Paper) -> Enrichment:
        raise RuntimeError("Programming error")

    monkeypatch.setattr(OpenAlexCollector, "enrich", broken)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(404))
    ) as client:
        with pytest.raises(RuntimeError, match="Programming error"):
            await EnrichmentPipeline(settings, client).enrich(paper)
