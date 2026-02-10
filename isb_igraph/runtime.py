from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import PipelineConfig


COMPUTE_PROFILES: dict[str, dict[str, Any]] = {
    "fast": {
        "compute_betweenness_enabled": False,
        "compute_betweenness_max_vertices": 5_000,
        "compute_betweenness_max_edges": 200_000,
        "compute_closeness_enabled": True,
        "compute_closeness_max_vertices": 12_000,
        "compute_closeness_max_edges": 500_000,
    },
    "balanced": {
        "compute_betweenness_enabled": True,
        "compute_betweenness_max_vertices": 15_000,
        "compute_betweenness_max_edges": 1_000_000,
        "compute_closeness_enabled": True,
        "compute_closeness_max_vertices": 30_000,
        "compute_closeness_max_edges": 1_500_000,
    },
    "deep": {
        "compute_betweenness_enabled": True,
        "compute_betweenness_max_vertices": 30_000,
        "compute_betweenness_max_edges": 2_000_000,
        "compute_closeness_enabled": True,
        "compute_closeness_max_vertices": 60_000,
        "compute_closeness_max_edges": 4_000_000,
    },
}

# Aliases used by UI/API wording.
COMPUTE_PROFILE_ALIASES: dict[str, str] = {
    "quick": "fast",
    "standard": "balanced",
    "full": "deep",
}


def runtime_root() -> Path:
    return Path(os.getenv("ISB_IGRAPH_RUNTIME_ROOT", "runtime")).resolve()


def uploads_root() -> Path:
    return Path(os.getenv("ISB_IGRAPH_UPLOADS_ROOT", str(runtime_root() / "uploads"))).resolve()


def artifacts_root() -> Path:
    return Path(os.getenv("ISB_IGRAPH_ARTIFACTS_ROOT", str(runtime_root() / "artifacts"))).resolve()


def jobs_db_path() -> Path:
    return Path(os.getenv("ISB_IGRAPH_JOBS_DB", str(runtime_root() / "jobs.db"))).resolve()


def ensure_runtime_dirs() -> None:
    runtime_root().mkdir(parents=True, exist_ok=True)
    uploads_root().mkdir(parents=True, exist_ok=True)
    artifacts_root().mkdir(parents=True, exist_ok=True)


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _normalize_profile(profile: str) -> str:
    key = str(profile).strip().lower()
    key = COMPUTE_PROFILE_ALIASES.get(key, key)
    if key not in COMPUTE_PROFILES:
        return "balanced"
    return key


def build_pipeline_config_dict(
    *,
    input_csv: Path,
    output_dir: Path,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = options or {}
    profile = _normalize_profile(str(options.get("compute_profile", "balanced")))
    profile_defaults = COMPUTE_PROFILES[profile]

    config = PipelineConfig(
        input_csv=input_csv,
        output_dir=output_dir,
        chunksize=_int(options.get("chunksize"), 25_000),
        top_k=_int(options.get("top_k"), 20),
        similarity_threshold=_float(options.get("similarity_threshold"), 0.0),
        similarity_method=str(options.get("similarity_method", options.get("similarity_mode", "both"))),
        projection_max_skill_degree=_int(options.get("projection_max_skill_degree"), 2_000),
        projection_max_pairs_per_skill=_int(options.get("projection_max_pairs_per_skill"), 1_500_000),
        projection_max_total_pairs=_int(options.get("projection_max_total_pairs"), 25_000_000),
        compute_betweenness_enabled=_bool(
            options.get("compute_betweenness_enabled"),
            profile_defaults["compute_betweenness_enabled"],
        ),
        compute_betweenness_max_vertices=_int(
            options.get("compute_betweenness_max_vertices"),
            profile_defaults["compute_betweenness_max_vertices"],
        ),
        compute_betweenness_max_edges=_int(
            options.get("compute_betweenness_max_edges"),
            profile_defaults["compute_betweenness_max_edges"],
        ),
        compute_closeness_enabled=_bool(
            options.get("compute_closeness_enabled"),
            profile_defaults["compute_closeness_enabled"],
        ),
        compute_closeness_max_vertices=_int(
            options.get("compute_closeness_max_vertices"),
            profile_defaults["compute_closeness_max_vertices"],
        ),
        compute_closeness_max_edges=_int(
            options.get("compute_closeness_max_edges"),
            profile_defaults["compute_closeness_max_edges"],
        ),
        subset_mode=_bool(options.get("subset_mode"), False),
        subset_target_rows=_optional_int(options.get("subset_target_rows")),
        subset_target_size_mb=_int(options.get("subset_target_size_mb"), 100),
        subset_seed=_int(options.get("subset_seed", options.get("seed")), 42),
        compute_profile=profile,
        similarity_mode=str(options.get("similarity_mode", options.get("similarity_method", "both"))),
        centrality_enabled=_bool(options.get("centrality_enabled"), True),
        betweenness_max_nodes=_int(
            options.get("betweenness_max_nodes"),
            profile_defaults["compute_betweenness_max_vertices"],
        ),
        closeness_max_nodes=_int(
            options.get("closeness_max_nodes"),
            profile_defaults["compute_closeness_max_vertices"],
        ),
        seed=_int(options.get("seed", options.get("subset_seed")), 42),
    )

    payload = asdict(config)
    payload["input_csv"] = str(input_csv)
    payload["output_dir"] = str(output_dir)
    payload["compute_profile"] = profile
    return payload


def pipeline_config_from_dict(payload: dict[str, Any]) -> PipelineConfig:
    profile = _normalize_profile(str(payload.get("compute_profile", "balanced")))
    return PipelineConfig(
        input_csv=Path(str(payload["input_csv"])),
        output_dir=Path(str(payload["output_dir"])),
        chunksize=_int(payload.get("chunksize"), 25_000),
        top_k=_int(payload.get("top_k"), 20),
        similarity_threshold=_float(payload.get("similarity_threshold"), 0.0),
        similarity_method=str(payload.get("similarity_method", "both")),
        projection_max_skill_degree=_int(payload.get("projection_max_skill_degree"), 2_000),
        projection_max_pairs_per_skill=_int(payload.get("projection_max_pairs_per_skill"), 1_500_000),
        projection_max_total_pairs=_int(payload.get("projection_max_total_pairs"), 25_000_000),
        compute_betweenness_enabled=_bool(payload.get("compute_betweenness_enabled"), True),
        compute_betweenness_max_vertices=_int(payload.get("compute_betweenness_max_vertices"), 15_000),
        compute_betweenness_max_edges=_int(payload.get("compute_betweenness_max_edges"), 1_000_000),
        compute_closeness_enabled=_bool(payload.get("compute_closeness_enabled"), True),
        compute_closeness_max_vertices=_int(payload.get("compute_closeness_max_vertices"), 30_000),
        compute_closeness_max_edges=_int(payload.get("compute_closeness_max_edges"), 1_500_000),
        subset_mode=_bool(payload.get("subset_mode"), False),
        subset_target_rows=_optional_int(payload.get("subset_target_rows")),
        subset_target_size_mb=_int(payload.get("subset_target_size_mb"), 100),
        subset_seed=_int(payload.get("subset_seed"), 42),
        compute_profile=profile,
        similarity_mode=str(payload.get("similarity_mode", payload.get("similarity_method", "both"))),
        centrality_enabled=_bool(payload.get("centrality_enabled"), True),
        betweenness_max_nodes=_int(payload.get("betweenness_max_nodes"), 15_000),
        closeness_max_nodes=_int(payload.get("closeness_max_nodes"), 30_000),
        seed=_int(payload.get("seed", payload.get("subset_seed")), 42),
    )
