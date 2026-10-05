from datetime import UTC, datetime

from research_radar.models.analysis import AnalysisResult
from research_radar.models.metrics import MetricsSnapshot
from research_radar.models.ranking import PaperRecord, Rankings
from research_radar.models.run import RunReport
from research_radar.storage.database import Database


class Repository:
    def __init__(self, database: Database) -> None:
        self.connection = database.connection

    def save_record(self, record: PaperRecord) -> None:
        paper = record.paper
        with self.connection:
            self.connection.execute(
                """INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET arxiv_id=excluded.arxiv_id, doi=excluded.doi,
                title=excluded.title, published_at=excluded.published_at,
                record_json=excluded.record_json""",
                (
                    paper.id,
                    paper.arxiv_id,
                    paper.doi,
                    paper.title,
                    paper.published_at.isoformat(),
                    record.model_dump_json(),
                ),
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
            for table in ("metrics_snapshots", "analyses"):
                self.connection.execute(
                    f"UPDATE OR IGNORE {table} SET paper_id=? WHERE paper_id=?", (new_id, old_id)
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
        rows = self.connection.execute(
            "SELECT snapshot_json FROM metrics_snapshots WHERE paper_id=? ORDER BY timestamp",
            (paper_id,),
        ).fetchall()
        values = [MetricsSnapshot.model_validate_json(row["snapshot_json"]) for row in rows]
        return [x for x in values if as_of is None or x.timestamp <= as_of]

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
