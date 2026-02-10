from __future__ import annotations

import pandas as pd

from isb_igraph.lookup import (
    _build_parser,
    load_edges_for_explainability,
    lookup_closest_jobs,
    resolve_query_job,
)


def _jobs_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "job_id": "job_1",
                "job_title": "4 Wheeler Mechanic",
                "company_name": "Alpha",
                "district": "Delhi",
                "assigned_occupation_group": "Vehicle Mechanics and Technicians",
                "job_title_norm": "4 wheeler mechanic",
                "company_name_norm": "alpha",
            },
            {
                "job_id": "job_2",
                "job_title": "4 Wheeler Mechanic",
                "company_name": "Beta",
                "district": "Tamil Nadu",
                "assigned_occupation_group": "Vehicle Mechanics and Technicians",
                "job_title_norm": "4 wheeler mechanic",
                "company_name_norm": "beta",
            },
            {
                "job_id": "job_3",
                "job_title": "Hotel Desk Clerk",
                "company_name": "Gamma",
                "district": "Kerala",
                "assigned_occupation_group": "Hospitality and Travel",
                "job_title_norm": "hotel desk clerk",
                "company_name_norm": "gamma",
            },
        ]
    )


def _sim_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "job_id": "job_1",
                "neighbor_job_id": "job_2",
                "similarity_score": 1.0,
                "rank": 1,
                "method": "unweighted",
            },
            {
                "job_id": "job_1",
                "neighbor_job_id": "job_2",
                "similarity_score": 0.83,
                "rank": 1,
                "method": "weighted",
            },
            {
                "job_id": "job_2",
                "neighbor_job_id": "job_1",
                "similarity_score": 1.0,
                "rank": 1,
                "method": "unweighted",
            },
        ]
    )


def _edges_export_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": "job_1",
                "target": "skill_a",
                "weight": 0.9,
                "attributes_json": '{"skill_text_normalized":"mechanical diagnosis"}',
            },
            {
                "source": "job_1",
                "target": "skill_b",
                "weight": 0.7,
                "attributes_json": '{"skill_text_normalized":"vehicle maintenance"}',
            },
            {
                "source": "job_2",
                "target": "skill_b",
                "weight": 0.6,
                "attributes_json": '{"skill_text_normalized":"vehicle maintenance"}',
            },
            {
                "source": "job_2",
                "target": "skill_c",
                "weight": 0.6,
                "attributes_json": '{"skill_text_normalized":"customer service"}',
            },
        ]
    )


def test_parser_default_method_is_unweighted() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--nodes-csv",
            "nodes.csv",
            "--similarity-csv",
            "job_similarity_topk.csv",
            "--job-id",
            "job_1",
        ]
    )
    assert args.method == "unweighted"


def test_resolve_query_by_job_id() -> None:
    resolved_id, rows, reason = resolve_query_job(_jobs_df(), job_id="job_3", title_query=None)
    assert resolved_id == "job_3"
    assert reason == "resolved_by_job_id"
    assert len(rows) == 1


def test_resolve_query_by_partial_title_unique() -> None:
    resolved_id, rows, reason = resolve_query_job(_jobs_df(), job_id=None, title_query="hotel desk")
    assert resolved_id == "job_3"
    assert reason == "resolved_by_partial_title"
    assert len(rows) == 1


def test_resolve_query_ambiguous_title() -> None:
    resolved_id, rows, reason = resolve_query_job(_jobs_df(), job_id=None, title_query="4 wheeler mechanic")
    assert resolved_id is None
    assert reason == "ambiguous_exact_title"
    assert len(rows) == 2


def test_lookup_closest_jobs_method_filter() -> None:
    result = lookup_closest_jobs(
        similarity_df=_sim_df(),
        jobs_df=_jobs_df(),
        query_job_id="job_1",
        method="weighted",
        limit=10,
    )
    assert len(result) == 1
    row = result.iloc[0]
    assert row["method"] == "weighted"
    assert row["neighbor_job_title"] == "4 Wheeler Mechanic"


def test_lookup_closest_jobs_all_methods() -> None:
    result = lookup_closest_jobs(
        similarity_df=_sim_df(),
        jobs_df=_jobs_df(),
        query_job_id="job_1",
        method="all",
        limit=10,
    )
    methods = set(result["method"].tolist())
    assert methods == {"unweighted", "weighted"}


def test_lookup_closest_jobs_with_explainability_columns() -> None:
    explain_edges = load_edges_for_explainability(_edges_export_df())
    result = lookup_closest_jobs(
        similarity_df=_sim_df(),
        jobs_df=_jobs_df(),
        query_job_id="job_1",
        method="unweighted",
        limit=10,
        edges_df=explain_edges,
    )

    assert "shared_skill_count" in result.columns
    assert "shared_skills_preview" in result.columns
    assert int(result.iloc[0]["shared_skill_count"]) == 1
    assert "vehicle maintenance" in str(result.iloc[0]["shared_skills_preview"])
