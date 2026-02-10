from __future__ import annotations

from isb_igraph.entities import EntityBuilder


def test_dedup_within_job_uses_highest_edge_weight() -> None:
    builder = EntityBuilder(default_similarity=0.5)
    row = {
        "job_id": "job_1",
        "job_title": "Mechanic",
        "company_name": "Acme",
        "posted_at": "2025-02-01",
        "skills": "[{\"skill\":\"Machine Operation\",\"bucket\":\"Working Knowledge\",\"mapping_similarity\":0.4},{\"skill\":\"Machine operation\",\"bucket\":\"Mission-Critical\",\"mapping_similarity\":0.9}]",
    }
    builder.process_row(row=row, row_idx=0)
    _, _, edges_df, _ = builder.to_dataframes()

    assert len(edges_df) == 1
    edge = edges_df.iloc[0]
    assert edge["skill_text_normalized"] == "machine operation"
    assert float(edge["edge_weight"]) > 0.9


def test_duplicate_job_rows_collapsed() -> None:
    builder = EntityBuilder(default_similarity=0.5)
    base = {
        "job_id": "job_dup",
        "job_title": "Technician",
        "company_name": "Acme",
        "posted_at": "2025-02-01",
        "skills": "[{\"skill\":\"Service Accounts\",\"bucket\":\"Familiarity\",\"mapping_similarity\":0.6}]",
    }
    builder.process_row(row=base, row_idx=0)
    builder.process_row(row=base, row_idx=1)

    jobs_df, _, edges_df, _ = builder.to_dataframes()
    assert len(jobs_df) == 1
    assert len(edges_df) == 1
    assert builder.stats.duplicate_job_rows >= 1


def test_missing_job_title_is_dropped() -> None:
    builder = EntityBuilder(default_similarity=0.5)
    row = {
        "job_id": "job_2",
        "job_title": "",
        "skills": "[{\"skill\":\"Driving\",\"bucket\":\"Proficient\",\"mapping_similarity\":0.7}]",
    }
    builder.process_row(row=row, row_idx=0)
    jobs_df, _, _, _ = builder.to_dataframes()
    assert jobs_df.empty
    assert builder.stats.dropped_rows_missing_title == 1
