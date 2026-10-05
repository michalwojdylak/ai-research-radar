from datetime import UTC, date, datetime

from research_radar.models.analysis import AnalysisResult
from research_radar.models.history import BackfillCheckpoint, RelevanceObservation
from research_radar.models.metrics import MetricsSnapshot
from research_radar.models.paper import Paper
from research_radar.models.radars import MonthlyRadar, RankingMode, WeeklyRadar, YearlyRadar
from research_radar.models.ranking import PaperRecord, Rankings
from research_radar.models.run import RunReport
from research_radar.periods import utc, week_for
from research_radar.ranking.deduplication import (
    canonical_arxiv_id,
    deduplicate_groups,
    normalize_doi,
    normalize_title,
)
from research_radar.storage.database import Database


class Repository:
    def __init__(self, database: Database) -> None:
        self.connection = database.connection

    def save_record(self, record: PaperRecord) -> None:
        paper = record.paper
        with self.connection:
            previous = self.get_record(paper.id)
            if previous is not None:
                self._freeze_relevance(previous)
            self.connection.execute(
                """INSERT INTO papers (id, arxiv_id, doi, title, published_at, record_json,
                canonical_arxiv_id, normalized_doi, normalized_title)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET arxiv_id=excluded.arxiv_id, doi=excluded.doi,
                title=excluded.title, published_at=excluded.published_at,
                record_json=excluded.record_json, canonical_arxiv_id=excluded.canonical_arxiv_id,
                normalized_doi=excluded.normalized_doi,
                normalized_title=excluded.normalized_title""",
                (
                    paper.id,
                    paper.arxiv_id,
                    paper.doi,
                    paper.title,
                    paper.published_at.isoformat(),
                    record.model_dump_json(),
                    canonical_arxiv_id(paper.arxiv_id or paper.id),
                    normalize_doi(paper.doi),
                    normalize_title(paper.title),
                ),
            )

    def _freeze_relevance(self, record: PaperRecord) -> None:
        moment = record.relevance_observed_at
        if moment is None or (record.paper.updated_at or record.paper.published_at) > moment:
            return
        observation = RelevanceObservation(
            observed_at=moment,
            paper=record.paper,
            relevance=record.relevance,
            primary_only=record.relevance_primary_only,
            model=record.relevance_model,
            prompt_version=record.relevance_prompt_version,
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO relevance_observations VALUES (?, ?, ?)",
            (record.paper.id, moment.isoformat(), observation.model_dump_json()),
        )

    def list_records(self) -> list[PaperRecord]:
        rows = self.connection.execute("SELECT record_json FROM papers ORDER BY id").fetchall()
        return [PaperRecord.model_validate_json(row["record_json"]) for row in rows]

    def get_record(self, paper_id: str) -> PaperRecord | None:
        row = self.connection.execute(
            "SELECT record_json FROM papers WHERE id=?", (paper_id,)
        ).fetchone()
        return PaperRecord.model_validate_json(row["record_json"]) if row else None

    def merge_identity(self, old_id: str, new_id: str) -> None:
        """Move operational history before removing an aliased discovery record."""
        if old_id == new_id:
            return
        with self.connection:
            for table in ("metrics_snapshots", "analyses", "relevance_observations"):
                self.connection.execute(
                    f"UPDATE OR IGNORE {table} SET paper_id=? WHERE paper_id=?", (new_id, old_id)
                )
            for row in self.connection.execute(
                "SELECT observed_at, observation_json FROM relevance_observations WHERE paper_id=?",
                (new_id,),
            ).fetchall():
                observation = RelevanceObservation.model_validate_json(row["observation_json"])
                if observation.paper.id == old_id:
                    observation.paper.id = new_id
                    self.connection.execute(
                        "UPDATE relevance_observations SET observation_json=? "
                        "WHERE paper_id=? AND observed_at=?",
                        (observation.model_dump_json(), new_id, row["observed_at"]),
                    )
            self.connection.execute("DELETE FROM papers WHERE id=?", (old_id,))

    def save_snapshot(self, paper_id: str, snapshot: MetricsSnapshot) -> None:
        timestamp = snapshot.timestamp.astimezone(UTC).isoformat()
        with self.connection:
            self.connection.execute(
                """INSERT INTO metrics_snapshots VALUES (?, ?, ?)
                ON CONFLICT(paper_id, timestamp)
                DO UPDATE SET snapshot_json=excluded.snapshot_json""",
                (paper_id, timestamp, snapshot.model_dump_json()),
            )

    def snapshots(self, paper_id: str, *, as_of: datetime | None = None) -> list[MetricsSnapshot]:
        query = "SELECT snapshot_json FROM metrics_snapshots WHERE paper_id=?"
        arguments: list[str] = [paper_id]
        if as_of is not None:
            query += " AND timestamp<=?"
            arguments.append(utc(as_of).isoformat())
        rows = self.connection.execute(query + " ORDER BY timestamp", arguments).fetchall()
        values = [MetricsSnapshot.model_validate_json(row["snapshot_json"]) for row in rows]
        return values

    def records_between(self, start: datetime, end: datetime) -> list[PaperRecord]:
        rows = self.connection.execute(
            "SELECT record_json FROM papers WHERE published_at>=? AND published_at<? ORDER BY id",
            (utc(start).isoformat(), utc(end).isoformat()),
        ).fetchall()
        return [PaperRecord.model_validate_json(row["record_json"]) for row in rows]

    def save_discovered(self, papers: list[Paper]) -> list[str]:
        """Merge only one discovery batch and indexed potential aliases, not the corpus."""
        if not papers:
            return []
        start = min(paper.published_at for paper in papers)
        end = max(paper.published_at for paper in papers)

        old = {
            record.paper.id: record
            for record in self.records_between(week_for(start).start, week_for(end).end)
        }
        for offset in range(0, len(papers), 150):
            batch = papers[offset : offset + 150]
            clauses: list[str] = []
            arguments: list[str] = []
            for column, values in (
                ("id", [paper.id for paper in batch]),
                (
                    "canonical_arxiv_id",
                    [x for p in batch if (x := canonical_arxiv_id(p.arxiv_id or p.id))],
                ),
                ("normalized_doi", [x for p in batch if (x := normalize_doi(p.doi))]),
                ("normalized_title", [normalize_title(p.title) for p in batch]),
            ):
                if values:
                    clauses.append(f"{column} IN ({','.join('?' for _ in values)})")
                    arguments.extend(values)
            rows = self.connection.execute(
                "SELECT record_json FROM papers WHERE " + " OR ".join(clauses), arguments
            ).fetchall()
            for row in rows:
                record = PaperRecord.model_validate_json(row["record_json"])
                old[record.paper.id] = record
        incoming_objects = {id(paper) for paper in papers}
        touched: list[str] = []
        for group in deduplicate_groups([record.paper for record in old.values()] + papers):
            if not any(id(member) in incoming_objects for member in group.members):
                continue
            paper = group.survivor
            aliases = sorted(
                [old[member.id] for member in group.members if member.id in old],
                key=lambda record: (
                    record.paper.id == paper.id,
                    record.paper.updated_at or record.paper.published_at,
                ),
                reverse=True,
            )
            with self.connection:
                for alias in aliases:
                    self._freeze_relevance(alias)
            record = aliases[0].model_copy(deep=True) if aliases else PaperRecord(paper=paper)
            if record.paper.title != paper.title or record.paper.abstract != paper.abstract:
                record.relevance = None
                record.relevance_observed_at = None
                record.relevance_primary_only = False
            record.paper = paper
            self.save_record(record)
            for alias in aliases:
                self.merge_identity(alias.paper.id, paper.id)
            touched.append(paper.id)
        return sorted(set(touched))

    def save_relevance_observation(self, paper_id: str, observation: RelevanceObservation) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO relevance_observations VALUES (?, ?, ?)
                ON CONFLICT(paper_id, observed_at) DO UPDATE SET
                observation_json=excluded.observation_json""",
                (paper_id, observation.observed_at.isoformat(), observation.model_dump_json()),
            )

    def relevance_observations(
        self, paper_id: str, *, as_of: datetime | None = None
    ) -> list[RelevanceObservation]:
        query = "SELECT observation_json FROM relevance_observations WHERE paper_id=?"
        arguments = [paper_id]
        if as_of is not None:
            query += " AND observed_at<=?"
            arguments.append(utc(as_of).isoformat())
        rows = self.connection.execute(query + " ORDER BY observed_at", arguments).fetchall()
        return [RelevanceObservation.model_validate_json(row["observation_json"]) for row in rows]

    def save_checkpoint(self, checkpoint: BackfillCheckpoint) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO backfill_runs VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id, period_id) DO UPDATE SET start_date=excluded.start_date,
                end_date=excluded.end_date, checkpoint_json=excluded.checkpoint_json""",
                (
                    checkpoint.job_id,
                    checkpoint.period_id,
                    checkpoint.start_date.isoformat(),
                    checkpoint.end_date.isoformat(),
                    checkpoint.model_dump_json(),
                ),
            )

    def get_checkpoint(self, job_id: str, period_id: str) -> BackfillCheckpoint | None:
        row = self.connection.execute(
            "SELECT checkpoint_json FROM backfill_runs WHERE job_id=? AND period_id=?",
            (job_id, period_id),
        ).fetchone()
        return BackfillCheckpoint.model_validate_json(row["checkpoint_json"]) if row else None

    def save_weekly(self, radar: WeeklyRadar) -> WeeklyRadar:
        existing = self.get_weekly(radar.id, mode=radar.mode, preview=radar.is_preview)
        if existing and existing.status in {"approved", "published"}:
            return existing
        anchor_year, anchor_month = week_for(radar.start_date).month_key
        with self.connection:
            self.connection.execute(
                """INSERT INTO weekly_radars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id, mode, preview) DO UPDATE SET radar_json=excluded.radar_json""",
                (
                    radar.id,
                    radar.mode,
                    int(radar.is_preview),
                    radar.year,
                    radar.week,
                    radar.start_date.isoformat(),
                    radar.end_date.isoformat(),
                    anchor_year,
                    anchor_month,
                    radar.model_dump_json(),
                ),
            )
        return radar

    def get_weekly(
        self, period_id: str, *, mode: RankingMode = "publication", preview: bool = False
    ) -> WeeklyRadar | None:
        row = self.connection.execute(
            "SELECT radar_json FROM weekly_radars WHERE id=? AND mode=? AND preview=?",
            (period_id, mode, int(preview)),
        ).fetchone()
        return WeeklyRadar.model_validate_json(row["radar_json"]) if row else None

    def list_weekly(
        self,
        *,
        mode: RankingMode = "publication",
        year: int | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[WeeklyRadar]:
        query = "SELECT radar_json FROM weekly_radars WHERE mode=? AND preview=0"
        arguments: list[str | int] = [mode]
        if year is not None:
            query += " AND year=?"
            arguments.append(year)
        if start_date is not None:
            query += " AND end_date>=?"
            arguments.append(start_date.isoformat())
        if end_date is not None:
            query += " AND start_date<=?"
            arguments.append(end_date.isoformat())
        rows = self.connection.execute(query + " ORDER BY id", arguments).fetchall()
        return [WeeklyRadar.model_validate_json(row["radar_json"]) for row in rows]

    def weeks_for_month(
        self, year: int, month: int, *, mode: RankingMode = "publication"
    ) -> list[WeeklyRadar]:
        rows = self.connection.execute(
            "SELECT radar_json FROM weekly_radars WHERE anchor_year=? AND anchor_month=? "
            "AND mode=? AND preview=0 ORDER BY id",
            (year, month, mode),
        ).fetchall()
        return [WeeklyRadar.model_validate_json(row["radar_json"]) for row in rows]

    def save_monthly(self, radar: MonthlyRadar) -> MonthlyRadar:
        existing = self.get_monthly(radar.id, mode=radar.mode)
        if existing and existing.status in {"approved", "published"}:
            return existing
        with self.connection:
            self.connection.execute(
                """INSERT INTO monthly_radars VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id, mode) DO UPDATE SET radar_json=excluded.radar_json""",
                (radar.id, radar.mode, radar.year, radar.month, radar.model_dump_json()),
            )
        return radar

    def get_monthly(
        self, period_id: str, *, mode: RankingMode = "publication"
    ) -> MonthlyRadar | None:
        row = self.connection.execute(
            "SELECT radar_json FROM monthly_radars WHERE id=? AND mode=?", (period_id, mode)
        ).fetchone()
        return MonthlyRadar.model_validate_json(row["radar_json"]) if row else None

    def list_monthly(
        self, *, mode: RankingMode = "publication", year: int | None = None
    ) -> list[MonthlyRadar]:
        query = "SELECT radar_json FROM monthly_radars WHERE mode=?"
        arguments: list[str | int] = [mode]
        if year is not None:
            query += " AND year=?"
            arguments.append(year)
        rows = self.connection.execute(query + " ORDER BY id", arguments).fetchall()
        return [MonthlyRadar.model_validate_json(row["radar_json"]) for row in rows]

    def save_yearly(self, radar: YearlyRadar) -> YearlyRadar:
        existing = self.get_yearly(radar.id, mode=radar.mode)
        if existing and existing.status in {"approved", "published"}:
            return existing
        with self.connection:
            self.connection.execute(
                """INSERT INTO yearly_radars VALUES (?, ?, ?, ?)
                ON CONFLICT(id, mode) DO UPDATE SET radar_json=excluded.radar_json""",
                (radar.id, radar.mode, radar.year, radar.model_dump_json()),
            )
        return radar

    def get_yearly(
        self, period_id: str, *, mode: RankingMode = "publication"
    ) -> YearlyRadar | None:
        row = self.connection.execute(
            "SELECT radar_json FROM yearly_radars WHERE id=? AND mode=?", (period_id, mode)
        ).fetchone()
        return YearlyRadar.model_validate_json(row["radar_json"]) if row else None

    def list_yearly(self, *, mode: RankingMode = "publication") -> list[YearlyRadar]:
        rows = self.connection.execute(
            "SELECT radar_json FROM yearly_radars WHERE mode=? ORDER BY id", (mode,)
        ).fetchall()
        return [YearlyRadar.model_validate_json(row["radar_json"]) for row in rows]

    def save_analysis(self, analysis: AnalysisResult) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO analyses VALUES (?, ?, ?, ?)
                ON CONFLICT(paper_id, cache_key) DO UPDATE SET
                generated_at=excluded.generated_at, analysis_json=excluded.analysis_json""",
                (
                    analysis.paper_id,
                    analysis.cache_key,
                    analysis.generated_at.isoformat(),
                    analysis.model_dump_json(),
                ),
            )

    def latest_analysis(self, paper_id: str) -> AnalysisResult | None:
        row = self.connection.execute(
            "SELECT analysis_json FROM analyses WHERE paper_id=? "
            "ORDER BY generated_at DESC, rowid DESC LIMIT 1",
            (paper_id,),
        ).fetchone()
        return AnalysisResult.model_validate_json(row["analysis_json"]) if row else None

    def save_rankings(self, rankings: Rankings) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO rankings VALUES (?, ?)",
                (rankings.timestamp.isoformat(), rankings.model_dump_json()),
            )

    def latest_rankings(self) -> Rankings | None:
        row = self.connection.execute(
            "SELECT ranking_json FROM rankings ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        return Rankings.model_validate_json(row["ranking_json"]) if row else None

    def save_run(self, report: RunReport) -> None:
        with self.connection:
            self.connection.execute(
                """INSERT INTO pipeline_runs VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET finished_at=excluded.finished_at,
                status=excluded.status, run_json=excluded.run_json""",
                (
                    report.id,
                    report.started_at.isoformat(),
                    report.finished_at.isoformat() if report.finished_at else None,
                    report.stage,
                    report.status,
                    report.model_dump_json(),
                ),
            )
