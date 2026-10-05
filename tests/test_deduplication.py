from datetime import UTC, datetime
from itertools import permutations

import pytest

from research_radar.models.paper import Paper
from research_radar.ranking.deduplication import (
    canonical_arxiv_id,
    deduplicate,
    deduplicate_groups,
    normalize_doi,
    normalize_title,
)


def paper(identifier: str, **changes: object) -> Paper:
    return Paper.model_validate(
        {
            "id": identifier,
            "title": "Efficient learning for large language models with limited data",
            "abstract": "Original abstract",
            "authors": ["Ada Lovelace"],
            "published_at": datetime(2026, 1, 1, tzinfo=UTC),
            "categories": ["cs.AI"],
            **changes,
        }
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2601.01234v7", "2601.01234"),
        (" https://arxiv.org/abs/2601.01234v2?download=1 ", "2601.01234"),
        ("http://export.arxiv.org/pdf/2601.0123v1.pdf", "2601.0123"),
        ("arXiv:math.GT/0309136v4", "math.gt/0309136"),
        ("https://arxiv.org/html/hep-th/9901001v2", "hep-th/9901001"),
        ("https://example.com/2601.01234", None),
        ("2601.01234garbage", None),
        ("", None),
        (None, None),
    ],
)
def test_canonical_arxiv(value: str | None, expected: str | None) -> None:
    assert canonical_arxiv_id(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" DOI:10.1234/AbC. ", "10.1234/abc"),
        ("https://doi.org/10.1234/Some%28Work%29?utm=x", "10.1234/some(work)"),
        ("http://dx.doi.org/10.12345/ABC", "10.12345/abc"),
        ("https://example.com/10.1234/abc", None),
        ("not a doi", None),
        (None, None),
    ],
)
def test_normalize_doi(value: str | None, expected: str | None) -> None:
    assert normalize_doi(value) == expected


def test_unicode_title_normalization() -> None:
    assert normalize_title("  ＡＩ—Based\u00a0Café: LLM’s ﬁne‐tuning\u200b! ") == (
        normalize_title("ai based Cafe\u0301 llm's fine-tuning")
    )
    assert normalize_title("naïve λ-learning") != normalize_title("naive l learning")


def test_versions_keep_latest_metadata_and_valid_older_fields() -> None:
    old = paper(
        "https://arxiv.org/abs/2601.01234v1",
        arxiv_id="2601.01234v1",
        doi="https://doi.org/10.1234/Valid",
        pdf_url="https://arxiv.org/pdf/2601.01234v1",
        categories=["cs.AI", "cs.LG"],
    )
    new = paper(
        "2601.01234v3",
        arxiv_id="2601.01234v3",
        title="Updated title",
        abstract=" ",
        authors=[],
        categories=["cs.CL"],
    )
    merged = deduplicate([old, new])
    assert len(merged) == 1
    assert merged[0].id == "2601.01234"
    assert merged[0].arxiv_id == "2601.01234v3"
    assert merged[0].title == "Updated title"
    assert merged[0].abstract == "Original abstract"
    assert merged[0].authors == ["Ada Lovelace"]
    assert merged[0].doi == "10.1234/valid"
    assert merged[0].pdf_url == old.pdf_url
    assert merged[0].categories == ["cs.AI", "cs.CL", "cs.LG"]
    assert old.arxiv_id == "2601.01234v1"
    rediscovered_old = paper("2601.01234v2", title="Obsolete title")
    assert deduplicate([merged[0], rediscovered_old])[0].title == "Updated title"


def test_doi_identity_and_original_publication_date() -> None:
    old = paper("old", doi="doi:10.1234/ABC")
    new = paper(
        "new",
        doi="https://doi.org/10.1234/abc",
        title="Revised",
        published_at=datetime(2026, 1, 5, tzinfo=UTC),
        updated_at=datetime(2026, 1, 6, tzinfo=UTC),
    )
    merged = deduplicate([new, old])[0]
    assert merged.id == "doi:10.1234/abc"
    assert merged.title == "Revised"
    assert merged.published_at == old.published_at
    assert merged.updated_at == new.updated_at


def test_newer_dated_metadata_without_version_does_not_regress() -> None:
    old = paper(
        "2601.00001v2",
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
        title="Old title",
    )
    newer = paper(
        "2601.00001",
        updated_at=datetime(2026, 1, 3, tzinfo=UTC),
        title="Updated title",
    )
    merged = deduplicate([old, newer])[0]
    assert merged.title == "Updated title"
    assert merged.id == "2601.00001"
    assert deduplicate([merged, old])[0].title == "Updated title"


def test_author_and_year_supported_title_fallback() -> None:
    left = paper("left")
    right = paper("right", title=left.title.upper().replace(" ", "\u00a0"))
    assert len(deduplicate([left, right])) == 1
    typo = paper("typo", title=left.title.replace("language", "langauge"))
    assert len(deduplicate([left, typo])) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"authors": ["Grace Hopper"]},
        {"authors": []},
        {"published_at": datetime(2025, 1, 1, tzinfo=UTC)},
        {"title": "Efficient learning for small language models with limited data"},
        {"title": "Efficient learning for large language models with limited data 2"},
    ],
)
def test_similar_or_identical_title_is_not_enough(changes: dict[str, object]) -> None:
    assert len(deduplicate([paper("left"), paper("right", **changes)])) == 2


def test_conflicting_strong_identifiers_never_title_merge() -> None:
    assert (
        len(
            deduplicate(
                [
                    paper("2601.00001", doi="10.1234/same"),
                    paper("2601.00002", doi="10.1234/same"),
                ]
            )
        )
        == 2
    )
    assert (
        len(
            deduplicate(
                [
                    paper("left", doi="10.1234/left"),
                    paper("right", doi="10.1234/right"),
                ]
            )
        )
        == 2
    )


def test_ambiguous_weak_bridge_stays_separate() -> None:
    result = deduplicate([paper("2601.00001"), paper("2601.00002"), paper("weak")])
    assert len(result) == 3


def test_deterministic_order_and_idempotence() -> None:
    inputs = [
        paper("arxiv:2601.00001v1", authors=[]),
        paper("2601.00001v2", authors=[]),
        paper("other", authors=[]),
        paper("other", authors=[], abstract="New"),
    ]
    expected = [p.model_dump() for p in deduplicate(inputs)]
    for permutation in permutations(inputs):
        assert [p.model_dump() for p in deduplicate(list(permutation))] == expected
    assert [p.model_dump() for p in deduplicate(deduplicate(inputs))] == expected


def test_groups_expose_exact_fuzzy_alias_memberships() -> None:
    original = paper("stored-alias")
    typo = paper("fresh-alias", title=original.title.replace("language", "langauge"))
    other = paper("unrelated", authors=["Grace Hopper"])
    inputs = [original, other, typo]
    groups = deduplicate_groups(inputs)
    assert [group.survivor for group in groups] == deduplicate(inputs)
    merged = next(group for group in groups if len(group.members) == 2)
    assert {member.id for member in merged.members} == {"stored-alias", "fresh-alias"}
    assert any(member is original for member in merged.members)
    assert any(member is typo for member in merged.members)
    assert sum(len(group.members) for group in groups) == len(inputs)
    assert deduplicate_groups(list(reversed(inputs))) == groups


def test_groups_never_assign_ambiguous_aliases_to_a_strong_survivor() -> None:
    inputs = [
        paper("2601.00001", doi="10.1234/shared"),
        paper("2601.00002", doi="10.1234/shared"),
        paper("weak", doi="10.1234/shared"),
    ]
    groups = deduplicate_groups(inputs)
    assert len(groups) == 3
    assert all(len(group.members) == 1 for group in groups)
    assert {group.survivor.id: group.members[0].id for group in groups} == {
        "2601.00001": "2601.00001",
        "2601.00002": "2601.00002",
        "doi:10.1234/shared": "weak",
    }


def test_version_group_preserves_earliest_publication_and_all_aliases() -> None:
    original = paper(
        "https://arxiv.org/abs/2601.00001v1",
        published_at=datetime(2025, 12, 1, tzinfo=UTC),
    )
    update = paper(
        "2601.00001v2",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 5, tzinfo=UTC),
    )
    group = deduplicate_groups([update, original])[0]
    assert group.survivor.id == "2601.00001"
    assert group.survivor.published_at == original.published_at
    assert {member.id for member in group.members} == {original.id, update.id}
