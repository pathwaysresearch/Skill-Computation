from __future__ import annotations

import pandas as pd

from isb_igraph.skill_lookup import (
    get_job_top_skills,
    load_edges_catalog,
    load_skills_catalog,
    parse_skill_query,
    rank_jobs_for_skills_or,
    resolve_skills,
)


def _nodes_export_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "node_id": "job_1",
                "node_type": "job",
                "label": "4 Wheeler Mechanic",
                "attributes_json": '{"company_name":"Alpha"}',
            },
            {
                "node_id": "job_2",
                "node_type": "job",
                "label": "Automotive Technician",
                "attributes_json": '{"company_name":"Beta"}',
            },
            {
                "node_id": "job_3",
                "node_type": "job",
                "label": "Customer Support Associate",
                "attributes_json": '{"company_name":"Gamma"}',
            },
            {
                "node_id": "skill_1",
                "node_type": "skill",
                "label": "Vehicle Maintenance",
                "attributes_json": '{"skill_text_normalized":"vehicle maintenance"}',
            },
            {
                "node_id": "skill_2",
                "node_type": "skill",
                "label": "Mechanical Diagnosis",
                "attributes_json": '{"skill_text_normalized":"mechanical diagnosis"}',
            },
            {
                "node_id": "skill_3",
                "node_type": "skill",
                "label": "Customer Service",
                "attributes_json": '{"skill_text_normalized":"customer service"}',
            },
        ]
    )


def _edges_export_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": "job_1",
                "target": "skill_1",
                "weight": 0.4,
                "attributes_json": '{"skill_text_normalized":"vehicle maintenance","bucket_normalized":"proficient","mapping_similarity":0.7}',
            },
            {
                "source": "job_1",
                "target": "skill_2",
                "weight": 0.2,
                "attributes_json": '{"skill_text_normalized":"mechanical diagnosis","bucket_normalized":"working knowledge","mapping_similarity":0.6}',
            },
            {
                "source": "job_2",
                "target": "skill_1",
                "weight": 0.3,
                "attributes_json": '{"skill_text_normalized":"vehicle maintenance","bucket_normalized":"proficient","mapping_similarity":0.75}',
            },
            {
                "source": "job_2",
                "target": "skill_2",
                "weight": 0.8,
                "attributes_json": '{"skill_text_normalized":"mechanical diagnosis","bucket_normalized":"mission critical","mapping_similarity":0.9}',
            },
            {
                "source": "job_3",
                "target": "skill_1",
                "weight": 0.9,
                "attributes_json": '{"skill_text_normalized":"vehicle maintenance","bucket_normalized":"advanced","mapping_similarity":0.8}',
            },
            {
                "source": "job_3",
                "target": "skill_3",
                "weight": 0.8,
                "attributes_json": '{"skill_text_normalized":"customer service","bucket_normalized":"mission critical","mapping_similarity":0.95}',
            },
        ]
    )


def _jobs_catalog_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "job_id": "job_1",
                "job_title": "4 Wheeler Mechanic",
            },
            {
                "job_id": "job_2",
                "job_title": "Automotive Technician",
            },
            {
                "job_id": "job_3",
                "job_title": "Customer Support Associate",
            },
        ]
    )


def test_parse_skill_query_deduplicates() -> None:
    parsed = parse_skill_query("vehicle maintenance, mechanical diagnosis, vehicle maintenance")
    assert parsed == ["vehicle maintenance", "mechanical diagnosis"]


def test_resolve_skills_exact_and_suggestions() -> None:
    skills_df = load_skills_catalog(_nodes_export_df())
    resolution = resolve_skills(
        ["Vehicle Maintenance", "mechanical diagnos"],
        skills_df,
        fuzzy_cutoff=70,
        suggestion_limit=3,
    )

    resolved = resolution["resolved_skills"]
    unresolved = resolution["unresolved_skills"]
    suggestions = resolution["suggestions"]

    assert len(resolved) == 1
    assert resolved.iloc[0]["skill_text_normalized"] == "vehicle maintenance"
    assert len(unresolved) == 1
    assert "mechanical diagnos" in suggestions
    assert len(suggestions["mechanical diagnos"]) >= 1


def test_rank_jobs_for_skills_or_prioritizes_count_then_weight() -> None:
    edges_df = load_edges_catalog(_edges_export_df())
    ranked = rank_jobs_for_skills_or(
        edges_df=edges_df,
        jobs_df=_jobs_catalog_df(),
        resolved_skill_ids=["skill_1", "skill_2"],
        total_query_skills=2,
        limit=10,
    )

    assert ranked.iloc[0]["job_id"] == "job_2"
    assert int(ranked.iloc[0]["matched_skill_count"]) == 2
    assert ranked.iloc[1]["job_id"] == "job_1"
    assert int(ranked.iloc[2]["matched_skill_count"]) == 1
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_get_job_top_skills_sorted_desc_by_weight() -> None:
    edges_df = load_edges_catalog(_edges_export_df())
    skills_df = load_skills_catalog(_nodes_export_df())

    top_skills = get_job_top_skills(
        edges_df=edges_df,
        skills_df=skills_df,
        jobs_df=_jobs_catalog_df(),
        query_job_id="job_2",
        limit=10,
    )

    assert len(top_skills) == 2
    assert top_skills.iloc[0]["skill_text_normalized"] == "mechanical diagnosis"
    assert float(top_skills.iloc[0]["edge_weight"]) >= float(top_skills.iloc[1]["edge_weight"])
    assert top_skills["rank"].tolist() == [1, 2]


def test_rank_jobs_for_skills_or_deterministic_repeated_runs() -> None:
    edges_df = load_edges_catalog(_edges_export_df())
    jobs_df = _jobs_catalog_df()

    first = rank_jobs_for_skills_or(
        edges_df=edges_df,
        jobs_df=jobs_df,
        resolved_skill_ids=["skill_1", "skill_2", "skill_3"],
        total_query_skills=3,
        limit=10,
    )
    second = rank_jobs_for_skills_or(
        edges_df=edges_df,
        jobs_df=jobs_df,
        resolved_skill_ids=["skill_1", "skill_2", "skill_3"],
        total_query_skills=3,
        limit=10,
    )

    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
