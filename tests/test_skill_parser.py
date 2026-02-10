from __future__ import annotations

from isb_igraph.skill_parser import parse_skills_cell


def test_parse_valid_json_array() -> None:
    raw = '[{"skill": "Problem Solving", "bucket": "Mission-Critical", "mapping_similarity": 0.9}]'
    skills, err = parse_skills_cell(raw, row_id=1)
    assert err is None
    assert len(skills) == 1
    assert skills[0].skill_text_raw == "Problem Solving"


def test_parse_single_quote_array() -> None:
    raw = "[{'skill':'Machine Operation','bucket':'Proficient','mapping_similarity':0.7}]"
    skills, err = parse_skills_cell(raw, row_id=2)
    assert err is None
    assert len(skills) == 1
    assert skills[0].bucket_raw == "Proficient"


def test_parse_recovery_from_fragment() -> None:
    raw = "prefix {'skill':'Payment Processing','bucket':'Working Knowledge','mapping_similarity':0.71} suffix"
    skills, err = parse_skills_cell(raw, row_id=3)
    assert err is None
    assert len(skills) == 1
    assert skills[0].skill_text_raw == "Payment Processing"


def test_parse_failure_reported() -> None:
    raw = "not-json and no objects"
    skills, err = parse_skills_cell(raw, row_id=4)
    assert skills == []
    assert err == "unable_to_parse_skills"
