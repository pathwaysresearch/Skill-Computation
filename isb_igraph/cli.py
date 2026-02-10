from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .environment import assert_runtime_compatibility
from .pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isb-igraph",
        description="Build job-skill bipartite graphs and job similarity outputs using igraph.",
    )
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for generated outputs")
    parser.add_argument("--chunksize", type=int, default=25_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--similarity-method", choices=["unweighted", "weighted", "both"], default="both")
    parser.add_argument("--similarity-threshold", type=float, default=0.0)
    parser.add_argument("--projection-max-skill-degree", type=int, default=2_000)
    parser.add_argument("--projection-max-pairs-per-skill", type=int, default=1_500_000)
    parser.add_argument("--projection-max-total-pairs", type=int, default=25_000_000)
    parser.add_argument("--memory-warning-edge-threshold", type=int, default=1_000_000)

    parser.add_argument("--disable-betweenness", action="store_true")
    parser.add_argument("--compute-betweenness-max-vertices", type=int, default=15_000)
    parser.add_argument("--compute-betweenness-max-edges", type=int, default=1_000_000)

    parser.add_argument("--disable-closeness", action="store_true")
    parser.add_argument("--compute-closeness-max-vertices", type=int, default=30_000)
    parser.add_argument("--compute-closeness-max-edges", type=int, default=1_500_000)

    parser.add_argument("--subset-mode", action="store_true", help="Run deterministic subset mode")
    parser.add_argument("--subset-target-rows", type=int, default=None)
    parser.add_argument("--subset-target-size-mb", type=int, default=100)
    parser.add_argument("--subset-seed", type=int, default=42)
    return parser


def main() -> None:
    assert_runtime_compatibility()
    parser = _build_parser()
    args = parser.parse_args()

    config = PipelineConfig(
        input_csv=Path(args.input),
        output_dir=Path(args.output_dir),
        chunksize=args.chunksize,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        similarity_method=args.similarity_method,
        projection_max_skill_degree=args.projection_max_skill_degree,
        projection_max_pairs_per_skill=args.projection_max_pairs_per_skill,
        projection_max_total_pairs=args.projection_max_total_pairs,
        memory_warning_edge_threshold=args.memory_warning_edge_threshold,
        compute_betweenness_enabled=not args.disable_betweenness,
        compute_betweenness_max_vertices=args.compute_betweenness_max_vertices,
        compute_betweenness_max_edges=args.compute_betweenness_max_edges,
        compute_closeness_enabled=not args.disable_closeness,
        compute_closeness_max_vertices=args.compute_closeness_max_vertices,
        compute_closeness_max_edges=args.compute_closeness_max_edges,
        subset_mode=args.subset_mode,
        subset_target_rows=args.subset_target_rows,
        subset_target_size_mb=args.subset_target_size_mb,
        subset_seed=args.subset_seed,
    )

    def progress(message: str, fraction: float) -> None:
        pct = round(fraction * 100, 1)
        print(f"[{pct:>5}%] {message}")

    result = run_pipeline(config, progress_fn=progress)

    print("\n=== Pipeline complete ===")
    print(json.dumps(result.graph_summary, indent=2))
    print("\nGenerated files:")
    for key, path in result.output_files.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
