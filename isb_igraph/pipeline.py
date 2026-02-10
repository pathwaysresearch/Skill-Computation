from __future__ import annotations

import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .config import PipelineConfig
from .entities import EntityBuilder
from .export import (
    build_edges_export,
    build_nodes_export,
    prefixed_filename,
    write_dataframe,
    write_json,
)
from .graph import (
    build_bipartite_graph,
    compute_bipartite_metrics,
    graph_component_summary,
    safe_projected_centrality,
)
from .ingest import canonicalize_chunk, detect_encoding, read_csv_chunks, resolve_columns
from .projection import build_job_similarity, build_projected_job_graph
from .qa import run_data_quality_checks
from .subset import SubsetResult, create_deterministic_subset

ProgressFn = Callable[[str, float], None] | None


@dataclass(slots=True)
class PipelineResult:
    output_dir: Path
    output_files: dict[str, Path]
    graph_summary: dict[str, Any]
    validation_report: dict[str, Any]
    qa_report: dict[str, Any]
    profile_records: list[dict[str, Any]]
    subset_result: SubsetResult | None = None



def _progress(progress_fn: ProgressFn, message: str, fraction: float) -> None:
    if progress_fn:
        progress_fn(message, max(0.0, min(1.0, fraction)))



def _stage(profile: list[dict[str, Any]], name: str):
    class _StageCtx:
        def __enter__(self_nonlocal):
            self_nonlocal.start = time.perf_counter()
            return self_nonlocal

        def __exit__(self_nonlocal, exc_type, exc, tb):
            elapsed = time.perf_counter() - self_nonlocal.start
            profile.append({"stage": name, "seconds": round(elapsed, 4)})

    return _StageCtx()



def run_pipeline_file(
    config: PipelineConfig,
    input_csv: Path,
    output_dir: Path,
    output_prefix: str = "",
    progress_fn: ProgressFn = None,
) -> PipelineResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_records: list[dict[str, Any]] = []
    memory_warnings: list[str] = []

    output_files: dict[str, Path] = {
        "nodes": output_dir / prefixed_filename("nodes", output_prefix),
        "edges": output_dir / prefixed_filename("edges", output_prefix),
        "job_similarity_topk": output_dir / prefixed_filename("job_similarity_topk", output_prefix),
        "graph_summary": output_dir / prefixed_filename("graph_summary", output_prefix),
        "parse_errors": output_dir / prefixed_filename("parse_errors", output_prefix),
        "job_node_metrics": output_dir / prefixed_filename("job_node_metrics", output_prefix),
        "skill_node_metrics": output_dir / prefixed_filename("skill_node_metrics", output_prefix),
        "projected_job_metrics": output_dir / prefixed_filename("projected_job_metrics", output_prefix),
        "profiling": output_dir / prefixed_filename("profiling", output_prefix),
        "validation": output_dir / prefixed_filename("validation", output_prefix),
        "qa_report": output_dir / prefixed_filename("qa_report", output_prefix),
    }

    builder = EntityBuilder(default_similarity=config.default_similarity)
    missing_required_fields: list[str] = []

    with _stage(profile_records, "ingest_parse_canonicalize"):
        encoding = detect_encoding(input_csv, config.encoding_candidates)
        _progress(progress_fn, f"Reading CSV in chunks ({encoding})", 0.05)

        row_offset = 0
        first_chunk = True
        warned = False
        for chunk_idx, raw_chunk in enumerate(
            read_csv_chunks(input_csv, chunksize=config.chunksize, encoding=encoding),
            start=1,
        ):
            if first_chunk:
                resolution = resolve_columns([str(c) for c in raw_chunk.columns])
                missing_required_fields = resolution.missing_required
                first_chunk = False

            canonical_chunk = canonicalize_chunk(raw_chunk)
            if "skills" in missing_required_fields:
                # Skills may be inferred from a non-standard column on canonicalization.
                skills_blank = canonical_chunk["skills"].fillna("").astype(str).str.strip().eq("").all()
                if not skills_blank:
                    missing_required_fields = [field for field in missing_required_fields if field != "skills"]

            builder.process_chunk(canonical_chunk, row_offset=row_offset)
            row_offset += len(canonical_chunk)

            if not warned and len(builder.edges) >= config.memory_warning_edge_threshold:
                warned = True
                memory_warnings.append(
                    (
                        "Edge count has crossed memory warning threshold; "
                        "consider subset mode or a higher projection_max_skill_degree filter."
                    )
                )

            if chunk_idx % 4 == 0:
                _progress(
                    progress_fn,
                    f"Parsed {row_offset:,} rows",
                    min(0.4, 0.06 + min(0.32, row_offset / max(1, row_offset + config.chunksize))),
                )

    jobs_df, skills_df, edges_df, parse_errors_df = builder.to_dataframes()

    if jobs_df.empty:
        raise ValueError("No valid job rows found after validation/parsing. Check input schema and required fields.")

    _progress(progress_fn, "Building igraph bipartite graph", 0.45)
    with _stage(profile_records, "build_bipartite_graph_and_metrics"):
        graph_artifacts = build_bipartite_graph(
            jobs_df=jobs_df,
            skills_df=skills_df,
            edges_df=edges_df,
            directed=config.directed,
        )
        bipartite_graph = graph_artifacts.graph
        job_node_metrics_df, skill_node_metrics_df = compute_bipartite_metrics(bipartite_graph)
        component_stats = graph_component_summary(bipartite_graph)

    _progress(progress_fn, "Projecting Job-Job similarities", 0.58)
    with _stage(profile_records, "projection_and_topk"):
        projection = build_job_similarity(
            edges_df=edges_df,
            method=config.similarity_method,
            top_k=config.top_k,
            similarity_threshold=config.similarity_threshold,
            max_skill_degree=config.projection_max_skill_degree,
            max_pairs_per_skill=config.projection_max_pairs_per_skill,
            max_total_pairs=config.projection_max_total_pairs,
        )

    weighted_for_projection = (
        projection.pair_scores_weighted
        if config.similarity_method in {"weighted", "both"}
        else projection.pair_scores_unweighted
    )

    _progress(progress_fn, "Computing projected graph centrality", 0.72)
    with _stage(profile_records, "projected_graph_metrics"):
        projected_graph = build_projected_job_graph(
            all_job_ids=jobs_df["job_id"].astype(str).tolist(),
            pair_scores=weighted_for_projection,
            similarity_threshold=config.similarity_threshold,
        )
        projected_job_metrics_df, projected_notes = safe_projected_centrality(
            projected_graph,
            betweenness_threshold=config.compute_betweenness_max_vertices,
            betweenness_max_edges=config.compute_betweenness_max_edges,
            closeness_threshold=config.compute_closeness_max_vertices,
            closeness_max_edges=config.compute_closeness_max_edges,
            compute_betweenness=config.compute_betweenness_enabled,
            compute_closeness=config.compute_closeness_enabled,
        )

    _progress(progress_fn, "Preparing exports and QA", 0.82)
    with _stage(profile_records, "exports_and_qa"):
        nodes_export_df = build_nodes_export(jobs_df, skills_df)
        edges_export_df = build_edges_export(edges_df)
        qa_report = run_data_quality_checks(nodes_export_df, edges_export_df, projection.topk_df)

        validation_report = {
            "missing_required_fields": missing_required_fields,
            "validation_error_counts": builder.validation_error_counts,
            "total_rows": builder.stats.total_rows,
            "valid_rows": builder.stats.valid_rows,
        }

        num_jobs = int(len(jobs_df))
        num_skills = int(len(skills_df))
        num_edges = int(len(edges_df))
        density = None
        if num_jobs > 0 and num_skills > 0:
            density = round(num_edges / float(num_jobs * num_skills), 8)

        graph_summary = {
            "num_jobs": num_jobs,
            "num_skills": num_skills,
            "num_edges": num_edges,
            "density": density,
            "num_components": component_stats["num_components"],
            "largest_component_size": component_stats["largest_component_size"],
            "parse_failure_count": builder.stats.parse_failure_count,
            "dedup_stats": builder.dedup_stats(),
            "projection_stats": projection.stats,
            "projected_graph": {
                "num_job_nodes": projected_graph.vcount(),
                "num_job_edges": projected_graph.ecount(),
                **projected_notes,
            },
            "memory_warnings": memory_warnings,
        }

        write_dataframe(nodes_export_df, output_files["nodes"])
        write_dataframe(edges_export_df, output_files["edges"])
        write_dataframe(projection.topk_df, output_files["job_similarity_topk"])
        write_dataframe(parse_errors_df, output_files["parse_errors"])
        write_dataframe(job_node_metrics_df, output_files["job_node_metrics"])
        write_dataframe(skill_node_metrics_df, output_files["skill_node_metrics"])
        write_dataframe(projected_job_metrics_df, output_files["projected_job_metrics"])

        profiling_df = pd.DataFrame(profile_records)
        write_dataframe(profiling_df, output_files["profiling"])

        write_json(graph_summary, output_files["graph_summary"])
        write_json(validation_report, output_files["validation"])
        write_json(qa_report, output_files["qa_report"])

    _progress(progress_fn, "Pipeline completed", 1.0)

    return PipelineResult(
        output_dir=output_dir,
        output_files=output_files,
        graph_summary=graph_summary,
        validation_report=validation_report,
        qa_report=qa_report,
        profile_records=profile_records,
    )



def run_pipeline(config: PipelineConfig, progress_fn: ProgressFn = None) -> PipelineResult:
    output_dir = Path(config.output_dir)
    subset_result: SubsetResult | None = None

    if config.subset_mode:
        _progress(progress_fn, "Running deterministic subset builder", 0.01)
        subset_result = create_deterministic_subset(
            config=config,
            output_dir=output_dir,
            progress_fn=lambda message, fraction: _progress(progress_fn, message, min(0.35, fraction * 0.35)),
        )

        sample_config = PipelineConfig(**asdict(config))
        sample_config.input_csv = subset_result.sample_input_path
        result = run_pipeline_file(
            config=sample_config,
            input_csv=subset_result.sample_input_path,
            output_dir=output_dir,
            output_prefix="sample_",
            progress_fn=lambda message, fraction: _progress(progress_fn, message, 0.35 + (fraction * 0.65)),
        )

        # Save subset summary separately as required for fast-testing outputs.
        subset_summary_path = output_dir / "sample_summary.json"
        subset_summary = {
            **subset_result.summary,
            "pipeline_graph_summary": result.graph_summary,
            "sample_input_path": str(subset_result.sample_input_path),
        }
        write_json(subset_summary, subset_summary_path)

        result.output_files["sample_summary"] = subset_summary_path
        result.subset_result = subset_result
        return result

    return run_pipeline_file(
        config=config,
        input_csv=Path(config.input_csv),
        output_dir=output_dir,
        output_prefix="",
        progress_fn=progress_fn,
    )
