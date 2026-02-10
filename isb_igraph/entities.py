from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .models import ParseError, SkillItem
from .normalization import (
    clip_similarity,
    compute_edge_weight,
    deterministic_id,
    deterministic_job_id,
    normalize_bucket,
    normalize_skill_text,
    parse_numeric,
    pretty_skill_label,
)
from .skill_parser import parse_skills_cell


@dataclass(slots=True)
class EntityBuildStats:
    total_rows: int = 0
    valid_rows: int = 0
    dropped_rows_missing_title: int = 0
    dropped_rows_missing_skills: int = 0
    duplicate_job_rows: int = 0
    duplicate_job_skill_edges: int = 0
    parse_failure_count: int = 0


class EntityBuilder:
    def __init__(self, default_similarity: float = 0.5) -> None:
        self.default_similarity = default_similarity
        self.jobs: dict[str, dict[str, Any]] = {}
        self.skills_by_norm: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple[str, str], dict[str, Any]] = {}
        self.parse_errors: list[ParseError] = []
        self.stats = EntityBuildStats()
        self.validation_error_counts: dict[str, int] = {}

    def _increment_error(self, reason: str) -> None:
        self.validation_error_counts[reason] = self.validation_error_counts.get(reason, 0) + 1

    @staticmethod
    def _clean(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        return text

    def _job_id(
        self,
        row: dict[str, Any],
        row_idx: int,
    ) -> str:
        existing = self._clean(row.get("job_id"))
        if existing:
            return existing
        return deterministic_job_id(
            row.get("job_title"),
            row.get("company_name"),
            row.get("posted_at"),
            row_idx,
        )

    def _upsert_job(self, job: dict[str, Any]) -> None:
        job_id = job["job_id"]
        if job_id in self.jobs:
            self.stats.duplicate_job_rows += 1
            existing = self.jobs[job_id]
            for key, value in job.items():
                if existing.get(key) in (None, "") and value not in (None, ""):
                    existing[key] = value
            return
        self.jobs[job_id] = job

    def _skill_id(self, normalized: str) -> str:
        return deterministic_id("skill", normalized)

    def _upsert_skill(self, normalized: str, raw: str) -> dict[str, Any]:
        existing = self.skills_by_norm.get(normalized)
        if existing:
            return existing
        record = {
            "skill_id": self._skill_id(normalized),
            "skill_text_normalized": normalized,
            "label": pretty_skill_label(normalized),
            "skill_text_raw_example": raw,
        }
        self.skills_by_norm[normalized] = record
        return record

    def process_row(self, row: dict[str, Any], row_idx: int) -> None:
        self.stats.total_rows += 1

        job_title = self._clean(row.get("job_title"))
        if not job_title:
            self.stats.dropped_rows_missing_title += 1
            self._increment_error("missing_job_title")
            return

        skills_raw = row.get("skills")
        if skills_raw is None or str(skills_raw).strip() == "":
            self.stats.dropped_rows_missing_skills += 1
            self._increment_error("missing_skills")
            return

        parsed_skills, parse_error_reason = parse_skills_cell(skills_raw, row_idx)
        if parse_error_reason:
            self.stats.parse_failure_count += 1
            self.parse_errors.append(
                ParseError(
                    row_id=row_idx,
                    reason=parse_error_reason,
                    raw_skills_snippet=str(skills_raw)[:500],
                )
            )

        job_id = self._job_id(row, row_idx)
        salary = parse_numeric(row.get("salary_mean_inr_month"))

        job_record = {
            "job_id": job_id,
            "job_title": job_title,
            "company_name": self._clean(row.get("company_name")),
            "posted_at": self._clean(row.get("posted_at")),
            "group": self._clean(row.get("group")),
            "assigned_occupation_group": self._clean(row.get("assigned_occupation_group")),
            "district": self._clean(row.get("district")),
            "industry": self._clean(row.get("industry")),
            "nco_code": self._clean(row.get("nco_code")),
            "salary_mean_inr_month": salary,
        }
        self._upsert_job(job_record)

        if not parsed_skills:
            self.stats.valid_rows += 1
            return

        # Deduplicate skill mentions inside one job row by maximum computed edge weight.
        per_job_skill_best: dict[str, tuple[SkillItem, float, int, str]] = {}
        for skill in parsed_skills:
            normalized = normalize_skill_text(skill.skill_text_raw)
            if not normalized:
                continue
            normalized_bucket, bucket_score = normalize_bucket(skill.bucket_raw)
            similarity = clip_similarity(skill.mapping_similarity)
            if similarity != similarity:
                similarity_value = None
            else:
                similarity_value = similarity
            edge_weight = compute_edge_weight(
                mapping_similarity=similarity_value,
                bucket_score=bucket_score,
                default_similarity=self.default_similarity,
            )
            existing = per_job_skill_best.get(normalized)
            if existing and existing[1] >= edge_weight:
                self.stats.duplicate_job_skill_edges += 1
                continue
            per_job_skill_best[normalized] = (skill, edge_weight, bucket_score, normalized_bucket)

        for normalized_skill, (skill, edge_weight, bucket_score, normalized_bucket) in per_job_skill_best.items():
            skill_record = self._upsert_skill(normalized_skill, skill.skill_text_raw)
            edge_key = (job_id, skill_record["skill_id"])
            edge_record = {
                "job_id": job_id,
                "skill_id": skill_record["skill_id"],
                "skill_text_raw": skill.skill_text_raw,
                "skill_text_normalized": normalized_skill,
                "bucket_raw": skill.bucket_raw,
                "bucket_score": bucket_score,
                "bucket_normalized": normalized_bucket,
                "mapping_similarity": skill.mapping_similarity,
                "edge_weight": edge_weight,
                "source_row_index": row_idx,
            }
            existing = self.edges.get(edge_key)
            if existing:
                if edge_weight > float(existing["edge_weight"]):
                    self.edges[edge_key] = edge_record
                self.stats.duplicate_job_skill_edges += 1
            else:
                self.edges[edge_key] = edge_record

        self.stats.valid_rows += 1

    def process_chunk(self, chunk: pd.DataFrame, row_offset: int = 0) -> None:
        records = chunk.to_dict(orient="records")
        for idx, row in enumerate(records):
            self.process_row(row=row, row_idx=row_offset + idx)

    def to_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        jobs_df = pd.DataFrame(self.jobs.values())
        skills_df = pd.DataFrame(self.skills_by_norm.values())
        edges_df = pd.DataFrame(self.edges.values())
        parse_errors_df = pd.DataFrame(
            [
                {
                    "row_id": err.row_id,
                    "reason": err.reason,
                    "raw_skills_snippet": err.raw_skills_snippet,
                }
                for err in self.parse_errors
            ]
        )
        if parse_errors_df.empty:
            parse_errors_df = pd.DataFrame(columns=["row_id", "reason", "raw_skills_snippet"])
        return jobs_df, skills_df, edges_df, parse_errors_df

    def dedup_stats(self) -> dict[str, int]:
        return {
            "duplicate_job_rows": self.stats.duplicate_job_rows,
            "duplicate_job_skill_edges": self.stats.duplicate_job_skill_edges,
            "dropped_rows_missing_title": self.stats.dropped_rows_missing_title,
            "dropped_rows_missing_skills": self.stats.dropped_rows_missing_skills,
        }
