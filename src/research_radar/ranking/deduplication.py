"""Conservative paper identity resolution, independent of input ordering."""

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from urllib.parse import unquote, urlsplit

from research_radar.models.paper import Paper

_ARXIV = re.compile(
    r"(?P<id>\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7})(?:v(?P<version>\d+))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PaperIdentityGroup:
    """A merged survivor and its exact original input members."""

    survivor: Paper
    members: tuple[Paper, ...]


def normalize_title(title: str) -> str:
    """Fold compatibility Unicode, case, punctuation, and whitespace.

    Apostrophes and invisible format characters are removed; other punctuation
    becomes a word boundary. Accents and meaningful letters/numbers are retained.
    """
    title = unicodedata.normalize("NFKC", title).casefold()
    characters: list[str] = []
    for character in title:
        category = unicodedata.category(character)
        if category == "Cf" or character in "'’‘ʼ`":
            continue
        characters.append(character if category[0] in {"L", "N", "M"} else " ")
    return " ".join("".join(characters).split())


def _arxiv_match(value: str | None) -> re.Match[str] | None:
    if not value:
        return None
    candidate = unquote(value.strip())
    if candidate.lower().startswith("arxiv:"):
        candidate = candidate[6:].strip()
    if "://" in candidate:
        parsed = urlsplit(candidate)
        host = (parsed.hostname or "").lower()
        if host != "arxiv.org" and not host.endswith(".arxiv.org"):
            return None
        candidate = parsed.path.strip("/")
        candidate = re.sub(r"^(?:abs|pdf|html)/", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\.pdf$", "", candidate, flags=re.IGNORECASE)
    return _ARXIV.fullmatch(candidate)


def canonical_arxiv_id(value: str | None) -> str | None:
    """Return a bare, lowercase, versionless modern or legacy arXiv ID."""
    match = _arxiv_match(value)
    return match.group("id").lower() if match else None


def normalize_doi(value: str | None) -> str | None:
    """Return a lowercase DOI without resolver URL or ``doi:`` prefix."""
    if not value:
        return None
    candidate = unquote(value.strip())
    if candidate.lower().startswith("doi:"):
        candidate = candidate[4:].strip()
    if "://" in candidate:
        parsed = urlsplit(candidate)
        if (parsed.hostname or "").lower() not in {"doi.org", "dx.doi.org"}:
            return None
        candidate = parsed.path.lstrip("/")
    candidate = candidate.strip().rstrip(".,;").lower()
    return candidate if re.fullmatch(r"10\.\d{4,9}/\S+", candidate) else None


def _arxiv(paper: Paper) -> str | None:
    return next(
        (
            identifier
            for value in (paper.arxiv_id, paper.id, paper.abs_url, paper.pdf_url)
            if (identifier := canonical_arxiv_id(value)) is not None
        ),
        None,
    )


def _version(paper: Paper) -> int:
    return max(
        (
            int(match.group("version") or 0)
            for value in (paper.arxiv_id, paper.id, paper.abs_url, paper.pdf_url)
            if (match := _arxiv_match(value)) is not None
        ),
        default=0,
    )


def _metadata_key(paper: Paper) -> tuple[int, float, str]:
    return (
        _version(paper),
        (paper.updated_at or paper.published_at).timestamp(),
        paper.model_dump_json(),
    )


def _compatible(paper: Paper, group: list[Paper]) -> bool:
    arxiv = _arxiv(paper)
    known_arxiv = {_arxiv(member) for member in group} - {None}
    if arxiv is not None and known_arxiv and arxiv not in known_arxiv:
        return False
    doi = normalize_doi(paper.doi)
    known_dois = {normalize_doi(member.doi) for member in group} - {None}
    if doi is not None and known_dois and doi not in known_dois:
        return arxiv is not None and arxiv in known_arxiv
    return True


def _title_match(left: Paper, right: Paper, *, exact: bool = False) -> bool:
    if left.published_at.year != right.published_at.year:
        return False
    left_authors = {normalize_title(author) for author in left.authors} - {""}
    right_authors = {normalize_title(author) for author in right.authors} - {""}
    if not left_authors.intersection(right_authors):
        return False
    left_title, right_title = normalize_title(left.title), normalize_title(right.title)
    if not left_title or not right_title:
        return False
    if left_title == right_title:
        return True
    if exact:
        return False
    if min(len(left_title), len(right_title)) < 40:
        return False
    left_words, right_words = set(left_title.split()), set(right_title.split())
    if min(len(left_words), len(right_words)) < 6:
        return False
    if re.findall(r"\d+", left_title) != re.findall(r"\d+", right_title):
        return False
    overlap = len(left_words & right_words) / len(left_words | right_words)
    return (
        len(left_title.split()) == len(right_title.split())
        and overlap >= 0.8
        and SequenceMatcher(None, left_title, right_title, autojunk=False).ratio() >= 0.97
    )


def _merge(group: list[Paper]) -> Paper:
    newest_first = sorted(group, key=_metadata_key, reverse=True)
    version_anchor = newest_first[0].updated_at or newest_first[0].published_at
    newer_unversioned = [
        member
        for member in newest_first
        if _version(member) == 0
        and member.updated_at is not None
        and member.updated_at > version_anchor
    ]
    if newer_unversioned:
        newest_first = sorted(
            newer_unversioned,
            key=lambda member: (member.updated_at or member.published_at, member.model_dump_json()),
            reverse=True,
        ) + [member for member in newest_first if member not in newer_unversioned]
    newest = newest_first[0]
    arxiv_ids = sorted(identifier for member in group if (identifier := _arxiv(member)))
    dois = [doi for member in newest_first if (doi := normalize_doi(member.doi))]
    identifier = (
        arxiv_ids[0]
        if arxiv_ids
        else f"doi:{dois[0]}"
        if dois
        else min(member.id for member in group)
    )
    version = max((_version(member) for member in group), default=0)
    arxiv_metadata = (
        (f"{arxiv_ids[0]}v{version}" if version else arxiv_ids[0]) if arxiv_ids else None
    )

    def text(field: str) -> str | None:
        for member in newest_first:
            value = getattr(member, field)
            if isinstance(value, str) and value.strip():
                return value
        return None

    authors = next((member.authors for member in newest_first if member.authors), [])
    updated = [member.updated_at for member in group if member.updated_at is not None]
    return Paper(
        id=identifier,
        arxiv_id=arxiv_metadata,
        title=text("title") or newest.title,
        abstract=text("abstract") or "",
        authors=list(authors),
        published_at=min(member.published_at for member in group),
        updated_at=max(updated) if updated else None,
        categories=sorted({category for member in group for category in member.categories}),
        doi=dois[0] if dois else None,
        pdf_url=text("pdf_url"),
        abs_url=text("abs_url"),
    )


def deduplicate_groups(papers: list[Paper]) -> list[PaperIdentityGroup]:
    """Merge arXiv, then DOI, then author/year-supported title identities.

    Conflicting strong identifiers never merge through a title or DOI bridge.
    Ambiguous weak matches remain separate. Latest metadata wins, with nonempty
    older fields retained, the original publication date, and all categories.
    IDs are canonical/versionless; arxiv_id retains a known latest version so
    repeated stored-record merges cannot downgrade it. Outputs are ID-sorted
    and do not mutate input papers.

    Every input appears in exactly one group's members, including fuzzy-title
    aliases. Use these memberships for repository migration, rather than
    re-matching merged survivors: ambiguity guards depend on the entire batch,
    and merging metadata can change the title or retain a different DOI.
    """
    ordered = sorted(
        papers,
        key=lambda paper: (
            0 if _arxiv(paper) else 1 if normalize_doi(paper.doi) else 2,
            _arxiv(paper) or normalize_doi(paper.doi) or paper.id,
            _metadata_key(paper),
        ),
    )
    groups: list[list[Paper]] = []
    arxiv_index: dict[str, set[int]] = defaultdict(set)
    doi_index: dict[str, set[int]] = defaultdict(set)
    id_index: dict[str, set[int]] = defaultdict(set)
    title_index: dict[str, set[int]] = defaultdict(set)
    author_year_index: dict[tuple[str, int], set[int]] = defaultdict(set)
    for paper in ordered:
        arxiv, doi = _arxiv(paper), normalize_doi(paper.doi)
        candidates = arxiv_index.get(arxiv, set()) if arxiv else set()
        if not candidates and doi:
            candidates = {
                index for index in doi_index.get(doi, set()) if _compatible(paper, groups[index])
            }
        if not candidates:
            candidates = {
                index
                for index in id_index.get(paper.id, set())
                if _compatible(paper, groups[index])
            }
        evidence: set[int] = set()
        for author in paper.authors:
            evidence.update(
                author_year_index.get((normalize_title(author), paper.published_at.year), set())
            )
        title = normalize_title(paper.title)
        if not candidates:
            candidates = {
                index
                for index in evidence & title_index.get(title, set())
                if _compatible(paper, groups[index])
                and any(_title_match(paper, member, exact=True) for member in groups[index])
            }
        if not candidates:
            candidates = {
                index
                for index in evidence
                if _compatible(paper, groups[index])
                and any(_title_match(paper, member) for member in groups[index])
            }
        if len(candidates) == 1:
            index = next(iter(candidates))
            groups[index].append(paper)
        else:
            index = len(groups)
            groups.append([paper])
        if arxiv:
            arxiv_index[arxiv].add(index)
        if doi:
            doi_index[doi].add(index)
        id_index[paper.id].add(index)
        title_index[title].add(index)
        for author in paper.authors:
            normalized = normalize_title(author)
            if normalized:
                author_year_index[normalized, paper.published_at.year].add(index)
    return sorted(
        (PaperIdentityGroup(survivor=_merge(group), members=tuple(group)) for group in groups),
        key=lambda group: group.survivor.id,
    )


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Return the survivors of :func:`deduplicate_groups` with identical matching."""
    return [group.survivor for group in deduplicate_groups(papers)]
