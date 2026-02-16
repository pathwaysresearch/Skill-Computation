from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_CANCELLED = "cancelled"

UPLOAD_STATUS_INITIATED = "initiated"
UPLOAD_STATUS_UPLOADING = "uploading"
UPLOAD_STATUS_COMPLETED = "completed"


@dataclass(slots=True)
class UploadRecord:
    upload_id: str
    file_name: str
    total_parts: int
    checksum_sha256: str | None
    status: str
    assembled_path: str | None
    bytes_received: int
    created_at: str
    completed_at: str | None


@dataclass(slots=True)
class JobRecord:
    job_id: str
    upload_id: str | None
    input_path: str
    output_dir: str
    status: str
    options_json: str
    stage: str | None
    progress: float
    error_message: str | None
    run_metadata_json: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS uploads (
                    upload_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    total_parts INTEGER NOT NULL,
                    checksum_sha256 TEXT,
                    status TEXT NOT NULL,
                    assembled_path TEXT,
                    bytes_received INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    upload_id TEXT,
                    input_path TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    status TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    stage TEXT,
                    progress REAL NOT NULL DEFAULT 0.0,
                    error_message TEXT,
                    run_metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    FOREIGN KEY(upload_id) REFERENCES uploads(upload_id)
                );

                CREATE TABLE IF NOT EXISTS job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    progress REAL,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );

                CREATE TABLE IF NOT EXISTS job_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    artifact_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    mime_type TEXT,
                    size_bytes INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_events_job_id ON job_events(job_id, id);
                CREATE INDEX IF NOT EXISTS idx_artifacts_job_id ON job_artifacts(job_id, id);
                """
            )

    @staticmethod
    def _utcnow() -> str:
        return datetime.now(UTC).isoformat()

    def create_upload(self, *, file_name: str, total_parts: int, checksum_sha256: str | None) -> UploadRecord:
        upload_id = str(uuid.uuid4())
        now = self._utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO uploads(upload_id, file_name, total_parts, checksum_sha256, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (upload_id, file_name, int(total_parts), checksum_sha256, UPLOAD_STATUS_INITIATED, now),
            )
        return self.get_upload(upload_id)

    def add_uploaded_bytes(self, upload_id: str, num_bytes: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE uploads
                SET bytes_received = bytes_received + ?, status = ?
                WHERE upload_id = ?
                """,
                (int(num_bytes), UPLOAD_STATUS_UPLOADING, upload_id),
            )

    def mark_upload_completed(self, upload_id: str, assembled_path: Path) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE uploads
                SET status = ?, assembled_path = ?, completed_at = ?
                WHERE upload_id = ?
                """,
                (UPLOAD_STATUS_COMPLETED, str(assembled_path), self._utcnow(), upload_id),
            )

    def get_upload(self, upload_id: str) -> UploadRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM uploads WHERE upload_id = ?", (upload_id,)).fetchone()
        if row is None:
            raise KeyError(f"upload not found: {upload_id}")
        return UploadRecord(**dict(row))

    def create_job(
        self,
        *,
        input_path: Path | str,
        output_dir: Path,
        options: dict[str, Any],
        upload_id: str | None = None,
        job_id: str | None = None,
    ) -> JobRecord:
        if not job_id:
            job_id = str(uuid.uuid4())
        now = self._utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(
                    job_id, upload_id, input_path, output_dir, status, options_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    upload_id,
                    str(input_path),
                    str(output_dir),
                    JOB_STATUS_QUEUED,
                    json.dumps(options, ensure_ascii=True),
                    now,
                ),
            )
        self.add_event(job_id, level="info", message="Job queued", progress=0.0)
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> JobRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"job not found: {job_id}")
        return JobRecord(**dict(row))

    def get_job_status(self, job_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"job not found: {job_id}")
        return str(row["status"])

    def add_event(self, job_id: str, *, level: str, message: str, progress: float | None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO job_events(job_id, created_at, level, message, progress)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, self._utcnow(), level, message, progress),
            )

    def list_events(self, job_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, job_id, created_at, level, message, progress
                FROM job_events
                WHERE job_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (job_id, int(limit)),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def list_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT artifact_name, file_path, mime_type, size_bytes, created_at
                FROM job_artifacts
                WHERE job_id = ?
                ORDER BY id ASC
                """,
                (job_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def replace_artifacts(self, job_id: str, artifacts: list[dict[str, Any]]) -> None:
        now = self._utcnow()
        with self._connect() as conn:
            conn.execute("DELETE FROM job_artifacts WHERE job_id = ?", (job_id,))
            for artifact in artifacts:
                conn.execute(
                    """
                    INSERT INTO job_artifacts(job_id, artifact_name, file_path, mime_type, size_bytes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        str(artifact.get("artifact_name")),
                        str(artifact.get("file_path")),
                        artifact.get("mime_type"),
                        int(artifact.get("size_bytes", 0)),
                        now,
                    ),
                )

    def update_job_progress(self, job_id: str, *, stage: str, progress: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET stage = ?, progress = ?
                WHERE job_id = ?
                """,
                (stage, float(progress), job_id),
            )

    def set_job_running(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, started_at = ?, stage = ?, progress = ?
                WHERE job_id = ?
                """,
                (JOB_STATUS_RUNNING, self._utcnow(), "starting", 0.0, job_id),
            )

    def set_job_completed(self, job_id: str, run_metadata: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, finished_at = ?, stage = ?, progress = ?, run_metadata_json = ?
                WHERE job_id = ?
                """,
                (
                    JOB_STATUS_COMPLETED,
                    self._utcnow(),
                    "completed",
                    1.0,
                    json.dumps(run_metadata, ensure_ascii=True),
                    job_id,
                ),
            )

    def set_job_failed(self, job_id: str, error_message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, finished_at = ?, stage = ?, error_message = ?
                WHERE job_id = ?
                """,
                (JOB_STATUS_FAILED, self._utcnow(), "failed", error_message, job_id),
            )

    def cancel_job(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, finished_at = ?, stage = ?
                WHERE job_id = ? AND status IN (?, ?)
                """,
                (
                    JOB_STATUS_CANCELLED,
                    self._utcnow(),
                    "cancelled",
                    job_id,
                    JOB_STATUS_QUEUED,
                    JOB_STATUS_RUNNING,
                ),
            )

    def claim_next_queued_job(self) -> JobRecord | None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT *
                FROM jobs
                WHERE status = ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (JOB_STATUS_QUEUED,),
            ).fetchone()
            if row is None:
                return None

            job_id = str(row["job_id"])
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, started_at = ?, stage = ?, progress = ?
                WHERE job_id = ?
                """,
                (JOB_STATUS_RUNNING, self._utcnow(), "claimed", 0.0, job_id),
            )

        return self.get_job(job_id)
