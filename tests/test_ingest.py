from __future__ import annotations

import pandas as pd

from isb_igraph.ingest import canonicalize_chunk, infer_skills_column



def test_infer_skills_column_from_shifted_schema() -> None:
    df = pd.DataFrame(
        {
            "Job ID": ["j1"],
            "Job Title": ["Mechanic"],
            "importance_standardised": [
                '[{"skill": "Problem Solving", "bucket": "Mission-Critical", "mapping_similarity": 0.9}]'
            ],
        }
    )
    canonical = canonicalize_chunk(df)
    assert canonical["skills"].iloc[0].startswith('[{"skill": "Problem Solving"')



def test_infer_skills_column_returns_none_for_non_skill_text() -> None:
    df = pd.DataFrame(
        {
            "col_a": ["hello", "world"],
            "col_b": ["just text", "no json"],
        }
    )
    assert infer_skills_column(df) is None
