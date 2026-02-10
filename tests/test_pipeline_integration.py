from __future__ import annotations

from pathlib import Path

import pandas as pd

from isb_igraph.config import PipelineConfig
from isb_igraph.pipeline import run_pipeline



def test_pipeline_end_to_end(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    input_csv = Path("sample_data/jobs_sample.csv").resolve()

    config = PipelineConfig(
        input_csv=input_csv,
        output_dir=output_dir,
        chunksize=4,
        top_k=5,
        similarity_method="both",
        similarity_threshold=0.0,
    )

    result = run_pipeline(config)

    assert result.output_files["nodes"].exists()
    assert result.output_files["edges"].exists()
    assert result.output_files["job_similarity_topk"].exists()
    assert result.output_files["graph_summary"].exists()
    assert result.output_files["parse_errors"].exists()

    nodes_df = pd.read_csv(result.output_files["nodes"])
    edges_df = pd.read_csv(result.output_files["edges"])
    topk_df = pd.read_csv(result.output_files["job_similarity_topk"])

    assert {"node_id", "node_type", "label", "attributes_json"}.issubset(nodes_df.columns)
    assert {"source", "target", "relation", "weight", "attributes_json"}.issubset(edges_df.columns)
    assert {"job_id", "neighbor_job_id", "similarity_score", "rank", "method"}.issubset(topk_df.columns)

    assert result.qa_report["all_passed"] is True
