from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SimilarityMethod = Literal["unweighted", "weighted", "both"]
ComputeProfile = Literal["fast", "balanced", "deep", "quick", "standard", "full"]


@dataclass(slots=True)
class PipelineConfig:
    input_csv: Path
    output_dir: Path
    chunksize: int = 25_000
    top_k: int = 20
    similarity_threshold: float = 0.0
    similarity_method: SimilarityMethod = "both"
    directed: bool = False
    compute_betweenness_enabled: bool = True
    compute_betweenness_max_vertices: int = 15_000
    compute_betweenness_max_edges: int = 1_000_000
    compute_closeness_enabled: bool = True
    compute_closeness_max_vertices: int = 30_000
    compute_closeness_max_edges: int = 1_500_000
    projection_max_skill_degree: int = 2_000
    projection_max_pairs_per_skill: int = 1_500_000
    projection_max_total_pairs: int = 25_000_000
    memory_warning_edge_threshold: int = 1_000_000
    default_similarity: float = 0.5
    subset_mode: bool = False
    subset_target_rows: int | None = None
    subset_target_size_mb: int = 100
    subset_seed: int = 42
    profile_stages: bool = True
    export_metrics: bool = True

    # Extended runtime options for async/deployment mode.
    compute_profile: ComputeProfile = "balanced"
    similarity_mode: SimilarityMethod = "both"
    centrality_enabled: bool = True
    betweenness_max_nodes: int = 15_000
    closeness_max_nodes: int = 30_000
    seed: int = 42

    encoding_candidates: tuple[str, ...] = field(
        default_factory=lambda: ("utf-8-sig", "utf-8", "latin-1")
    )


DEFAULT_JOB_COLUMNS: dict[str, tuple[str, ...]] = {
    "job_id": ("job_id", "job id", "jobid", "id"),
    "job_title": ("job_title", "job title", "title"),
    "skills": ("skills", "skill"),
    "company_name": ("company_name", "company name", "firm", "company"),
    "posted_at": ("posted_at", "posted at", "posting_date", "date_posted"),
    "group": ("group", "occupation_group", "occupation group"),
    "assigned_occupation_group": (
        "assigned_occupation_group",
        "assigned occupation group",
        "category",
        "occupation_category",
    ),
    "district": ("district", "location", "city", "district/state"),
    "industry": ("industry",),
    "nco_code": ("nco_code", "nco code"),
    "salary_mean_inr_month": (
        "salary_mean_inr_month",
        "salary",
        "salary_mean",
        "salary_monthly",
    ),
}


BUCKET_SCORE_MAP: dict[str, int] = {
    "mission-critical": 4,
    "mission critical": 4,
    "critical": 4,
    "advanced": 3,
    "4: advanced": 3,
    "3: proficient": 2,
    "proficient": 2,
    "working knowledge": 1,
    "1: familiarity": 0,
    "familiarity": 0,
}
