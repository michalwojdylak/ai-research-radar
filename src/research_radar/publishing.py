import base64
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict

from research_radar.config import Settings
from research_radar.models.analysis import ResearchPaper
from research_radar.models.radars import MonthlyRadar, WeeklyRadar, YearlyRadar
from research_radar.pipeline.generate import extract_human_block


class GitTree(BaseModel):
    sha: str


class GitCommit(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sha: str
    tree: GitTree


class GitRef(BaseModel):
    model_config = ConfigDict(extra="ignore")
    object: GitTree


class TreeEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str
    sha: str
    type: str


class TreeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tree: list[TreeEntry]
    truncated: bool


class PullRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    html_url: str


class GitBlob(BaseModel):
    model_config = ConfigDict(extra="ignore")
    encoding: str
    content: str


def _editorial_document(
    content: str | bytes, path: str
) -> ResearchPaper | WeeklyRadar | MonthlyRadar | YearlyRadar:
    if not path.startswith("data/radars/"):
        return ResearchPaper.model_validate_json(content)
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("Expected a structured Radar document")
    kind = data.get("type")
    document: WeeklyRadar | MonthlyRadar | YearlyRadar
    if kind == "weekly_radar":
        document = WeeklyRadar.model_validate(data)
        if document.is_preview:
            raise ValueError("Preview Radars must remain local, not be submitted as final drafts")
    elif kind == "monthly_radar":
        document = MonthlyRadar.model_validate(data)
    elif kind == "yearly_radar":
        document = YearlyRadar.model_validate(data)
    else:
        raise ValueError("Unknown Radar document type")
    expected_mode = "retrospective" if "/retrospective/" in path else "publication"
    if document.mode != expected_mode:
        raise ValueError("Radar mode does not match its destination")
    return document


async def create_review_pr(
    settings: Settings, client: httpx.AsyncClient, paths: list[Path], *, now: datetime
) -> str | None:
    """Submit only generated content, never approve or publish it."""
    if settings.dry_run:
        return None
    if not settings.github_repository or not settings.github_token:
        raise ValueError("PR submission requires GITHUB_REPOSITORY and GITHUB_TOKEN")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", settings.github_repository):
        raise ValueError("GITHUB_REPOSITORY must be owner/repository")
    if not paths:
        return None
    api = f"https://api.github.com/repos/{settings.github_repository}"
    headers = {
        "Authorization": f"Bearer {settings.github_token.get_secret_value()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async def request(
        method: str, suffix: str, body: dict[str, object] | None = None
    ) -> httpx.Response:
        response = await client.request(method, api + suffix, headers=headers, json=body)
        response.raise_for_status()
        return response

    branch = f"research-radar/{now:%Y-%m-%d}"
    response = await client.get(api + f"/git/ref/heads/{branch}", headers=headers)
    exists = response.status_code != 404
    if exists:
        response.raise_for_status()
    else:
        response = await request("GET", f"/git/ref/heads/{settings.github_base_branch}")
    ref = GitRef.model_validate(response.json())
    response = await request("GET", f"/git/commits/{ref.object.sha}")
    commit = GitCommit.model_validate(response.json())
    response = await request("GET", f"/git/trees/{commit.tree.sha}?recursive=1")
    tree = TreeResponse.model_validate(response.json())
    if tree.truncated:
        raise ValueError("GitHub tree is truncated; cannot safely compare generated content")
    old_files = {entry.path: entry.sha for entry in tree.tree if entry.type == "blob"}
    entries: list[dict[str, str]] = []
    for path in sorted(set(paths)):
        relative = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        if not relative.startswith(
            ("data/papers/", "data/articles/drafts/", "data/rankings/", "data/radars/")
        ) or (relative.startswith("data/radars/") and "/previews/" in relative):
            raise ValueError("Only generated research files may be submitted")
        content = path.read_bytes()
        if relative.startswith("data/radars/") and relative.endswith(".json"):
            _editorial_document(content, relative)
        blob_sha = hashlib.sha1(
            f"blob {len(content)}\0".encode() + content, usedforsecurity=False
        ).hexdigest()
        if old_files.get(relative) == blob_sha:
            continue
        if exists and relative in old_files and not relative.startswith("data/rankings/"):
            response = await request("GET", f"/git/blobs/{old_files[relative]}")
            previous_blob = GitBlob.model_validate(response.json())
            if previous_blob.encoding != "base64":
                raise ValueError("Unexpected GitHub blob encoding")
            previous_text = base64.b64decode(
                "".join(previous_blob.content.splitlines()), validate=True
            ).decode("utf-8")
            if relative.endswith(".json"):
                previous = _editorial_document(previous_text, relative)
                current = _editorial_document(content, relative)
                if previous.status in {"approved", "published"}:
                    raise ValueError("Remote content has human approval; refusing to overwrite")
                if previous.michaels_take and previous.michaels_take != current.michaels_take:
                    raise ValueError("Remote editorial content differs; refusing to overwrite")
            elif relative.endswith(".md"):
                previous_take = extract_human_block(previous_text)
                current_take = extract_human_block(content.decode("utf-8"))
                if previous_take and previous_take != current_take:
                    raise ValueError("Remote editorial content differs; refusing to overwrite")
        response = await request(
            "POST",
            "/git/blobs",
            {"content": base64.b64encode(content).decode(), "encoding": "base64"},
        )
        blob = GitTree.model_validate(response.json())
        entries.append({"path": relative, "mode": "100644", "type": "blob", "sha": blob.sha})
    if not entries:
        return None
    response = await request("POST", "/git/trees", {"base_tree": commit.tree.sha, "tree": entries})
    new_tree = GitTree.model_validate(response.json())
    response = await request(
        "POST",
        "/git/commits",
        {
            "message": f"Research Radar drafts: {now:%Y-%m-%d}\n\n"
            "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>",
            "tree": new_tree.sha,
            "parents": [commit.sha],
        },
    )
    new_commit = GitTree.model_validate(response.json())
    if exists:
        await request("PATCH", f"/git/refs/heads/{branch}", {"sha": new_commit.sha, "force": False})
    else:
        await request("POST", "/git/refs", {"ref": f"refs/heads/{branch}", "sha": new_commit.sha})
    owner = settings.github_repository.split("/")[0]
    response = await client.get(
        api + "/pulls",
        headers=headers,
        params={"head": f"{owner}:{branch}", "base": settings.github_base_branch, "state": "open"},
    )
    response.raise_for_status()
    prs = response.json()
    if not isinstance(prs, list):
        raise ValueError("GitHub returned malformed pull request list")
    if prs:
        return PullRequest.model_validate(prs[0]).html_url
    response = await request(
        "POST",
        "/pulls",
        {
            "title": f"AI Research Radar: {now:%Y-%m-%d}",
            "head": branch,
            "base": settings.github_base_branch,
            "draft": True,
            "body": "AI-assisted research drafts for human review. Check source support, "
            "critic issues, and scores before approval. Michał's take is human-authored "
            "and intentionally empty. This PR does not publish to the website.",
        },
    )
    return PullRequest.model_validate(response.json()).html_url
