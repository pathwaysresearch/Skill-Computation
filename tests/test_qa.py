from __future__ import annotations

import pandas as pd

from isb_igraph.qa import run_data_quality_checks



def test_data_quality_checks_pass() -> None:
    nodes_df = pd.DataFrame(
        [
            {"node_id": "job_1", "node_type": "job", "label": "Job 1", "attributes_json": "{}"},
            {"node_id": "skill_1", "node_type": "skill", "label": "Skill 1", "attributes_json": "{}"},
        ]
    )
    edges_df = pd.DataFrame(
        [
            {
                "source": "job_1",
                "target": "skill_1",
                "relation": "requires_skill",
                "weight": 0.8,
                "attributes_json": "{}",
            }
        ]
    )
    topk_df = pd.DataFrame(
        [
            {
                "job_id": "job_1",
                "neighbor_job_id": "job_2",
                "similarity_score": 1.0,
                "rank": 1,
                "method": "unweighted",
            }
        ]
    )

    report = run_data_quality_checks(nodes_df, edges_df, topk_df)
    assert report["all_passed"] is True
