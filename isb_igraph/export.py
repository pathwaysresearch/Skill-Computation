from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_FILENAMES = {
    "nodes": "nodes.csv",
    "edges": "edges.csv",
    "job_similarity_topk": "job_similarity_topk.csv",
    "graph_summary": "graph_summary.json",
    "parse_errors": "parse_errors.csv",
    "job_node_metrics": "job_node_metrics.csv",
    "skill_node_metrics": "skill_node_metrics.csv",
    "projected_job_metrics": "projected_job_metrics.csv",
    "profiling": "profiling_summary.csv",
    "validation": "validation_report.json",
    "qa_report": "qa_report.json",
}



def _compact_json(value: dict[str, Any]) -> str:
    clean: dict[str, Any] = {}
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, float) and math.isnan(item):
            continue
        clean[key] = item
    return json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


def build_nodes_export(jobs_df: pd.DataFrame, skills_df: pd.DataFrame) -> pd.DataFrame:
    job_rows: list[dict[str, Any]] = []
    for row in jobs_df.to_dict(orient="records"):
        attrs = {
            "job_id": row.get("job_id"),
            "job_title": row.get("job_title"),
            "company_name": row.get("company_name"),
            "posted_at": row.get("posted_at"),
            "group": row.get("group"),
            "assigned_occupation_group": row.get("assigned_occupation_group"),
            "district": row.get("district"),
            "industry": row.get("industry"),
            "nco_code": row.get("nco_code"),
            "salary_mean_inr_month": row.get("salary_mean_inr_month"),
        }
        job_rows.append(
            {
                "node_id": row.get("job_id"),
                "node_type": "job",
                "label": row.get("job_title") or "Unknown Job",
                "attributes_json": _compact_json(attrs),
            }
        )

    skill_rows: list[dict[str, Any]] = []
    for row in skills_df.to_dict(orient="records"):
        attrs = {
            "skill_id": row.get("skill_id"),
            "skill_text_normalized": row.get("skill_text_normalized"),
            "skill_text_raw_example": row.get("skill_text_raw_example"),
        }
        skill_rows.append(
            {
                "node_id": row.get("skill_id"),
                "node_type": "skill",
                "label": row.get("label") or row.get("skill_text_normalized") or "Unknown Skill",
                "attributes_json": _compact_json(attrs),
            }
        )

    nodes_df = pd.DataFrame(job_rows + skill_rows)
    return nodes_df


def build_edges_export(edges_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in edges_df.to_dict(orient="records"):
        attrs = {
            "skill_text_raw": row.get("skill_text_raw"),
            "skill_text_normalized": row.get("skill_text_normalized"),
            "bucket_raw": row.get("bucket_raw"),
            "bucket_score": row.get("bucket_score"),
            "bucket_normalized": row.get("bucket_normalized"),
            "mapping_similarity": row.get("mapping_similarity"),
            "source_row_index": row.get("source_row_index"),
        }
        rows.append(
            {
                "source": row.get("job_id"),
                "target": row.get("skill_id"),
                "relation": "requires_skill",
                "weight": row.get("edge_weight"),
                "attributes_json": _compact_json(attrs),
            }
        )
    return pd.DataFrame(rows)


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    if df is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def prefixed_filename(name: str, prefix: str = "") -> str:
    base = OUTPUT_FILENAMES[name]
    if not prefix:
        return base
    return f"{prefix}{base}"
