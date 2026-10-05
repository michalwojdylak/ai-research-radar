"""Source-matched, elapsed-time-normalized growth without inferred history."""

from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

from research_radar.models.metrics import MetricsSnapshot, PaperMetrics
from research_radar.models.paper import Enrichment


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _provenance(value: str | None) -> str | None:
    if not value or not value.strip():
        return None
    value = value.strip()
    if "://" not in value:
        return value.casefold()
    parsed = urlsplit(value)
    host = (parsed.hostname or "").casefold()
    path = parsed.path.rstrip("/")
    if host == "github.com":
        path = path.removesuffix(".git").casefold()
    return f"{host}{path}"


def _daily_velocity(
    current: MetricsSnapshot,
    history: list[MetricsSnapshot],
    *,
    count_field: str,
    source_field: str,
    horizon: timedelta,
    tolerance: timedelta,
) -> float | None:
    count = getattr(current, count_field)
    source = _provenance(getattr(current, source_field))
    if not isinstance(count, int) or source is None:
        return None
    target = current.timestamp - horizon
    eligible = [
        snapshot
        for snapshot in history
        if target - tolerance <= snapshot.timestamp <= target
        and isinstance(getattr(snapshot, count_field), int)
        and _provenance(getattr(snapshot, source_field)) == source
    ]
    if not eligible:
        return None
    baseline = max(eligible, key=lambda item: (item.timestamp, item.model_dump_json()))
    old_count = getattr(baseline, count_field)
    if not isinstance(old_count, int):
        return None
    elapsed_days = (current.timestamp - baseline.timestamp).total_seconds() / 86400
    return (count - old_count) / elapsed_days


def calculate_metrics(
    enrichment: Enrichment, snapshots: list[MetricsSnapshot], now: datetime
) -> PaperMetrics:
    """Use the latest snapshot at/before ``now`` as the sampled current state.

    All velocities are signed counts/day, including the 7d field. The 24h
    baselines must be 24–36h old; the 7d baseline must be 7–8d old. The closest
    eligible, same-provenance sample is used. Short history, unknown provenance,
    and absent counts produce None, not extrapolated or fabricated velocities.
    Without a current snapshot, enrichment counts are usable but velocity is not.
    ``observed_at`` records the actual endpoint, including stale/replayed data.
    """
    now = _utc(now)
    history = [snapshot for snapshot in snapshots if snapshot.timestamp <= now]
    current = max(history, key=lambda item: (item.timestamp, item.model_dump_json()), default=None)
    metrics = PaperMetrics(
        citations=current.citations if current else enrichment.citations,
        influential_citations=enrichment.influential_citations,
        github_stars=current.github_stars if current else enrichment.github_stars,
        github_forks=enrichment.github_forks,
        github_contributors=enrichment.github_contributors,
        github_releases=enrichment.github_releases,
        hf_upvotes=current.hf_upvotes if current else enrichment.hf_upvotes,
        observed_at=current.timestamp if current else now,
    )
    if current is None:
        return metrics
    metrics.github_star_velocity_24h = _daily_velocity(
        current,
        history,
        count_field="github_stars",
        source_field="github_url",
        horizon=timedelta(days=1),
        tolerance=timedelta(hours=12),
    )
    metrics.github_star_velocity_7d = _daily_velocity(
        current,
        history,
        count_field="github_stars",
        source_field="github_url",
        horizon=timedelta(days=7),
        tolerance=timedelta(days=1),
    )
    metrics.citation_velocity = _daily_velocity(
        current,
        history,
        count_field="citations",
        source_field="citation_source",
        horizon=timedelta(days=1),
        tolerance=timedelta(hours=12),
    )
    metrics.hf_velocity = _daily_velocity(
        current,
        history,
        count_field="hf_upvotes",
        source_field="hf_paper_url",
        horizon=timedelta(days=1),
        tolerance=timedelta(hours=12),
    )
    return metrics
