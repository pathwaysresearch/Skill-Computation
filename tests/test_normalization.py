from __future__ import annotations

from isb_igraph.normalization import (
    compute_edge_weight,
    normalize_bucket,
    normalize_skill_text,
)


def test_skill_normalization_deterministic() -> None:
    assert normalize_skill_text(" Machine Operation  ") == "machine operation"
    assert normalize_skill_text("Machine operation!!!") == "machine operation"


def test_bucket_normalization_map() -> None:
    assert normalize_bucket("Mission-Critical")[1] == 4
    assert normalize_bucket("mission critical")[1] == 4
    assert normalize_bucket("critical")[1] == 4
    assert normalize_bucket("4: Advanced")[1] == 3
    assert normalize_bucket("Advanced")[1] == 3
    assert normalize_bucket("3: Proficient")[1] == 2
    assert normalize_bucket("Proficient")[1] == 2
    assert normalize_bucket("Working Knowledge")[1] == 1
    assert normalize_bucket("1: Familiarity")[1] == 0
    assert normalize_bucket("Familiarity")[1] == 0


def test_edge_weight_formula_with_similarity() -> None:
    weight = compute_edge_weight(mapping_similarity=0.8, bucket_score=4)
    # 0.7 * 0.8 + 0.3 * (4/4) = 0.56 + 0.3 = 0.86
    assert abs(weight - 0.86) < 1e-9


def test_edge_weight_formula_without_similarity() -> None:
    weight = compute_edge_weight(mapping_similarity=None, bucket_score=2, default_similarity=0.5)
    # 0.5 * (2/4) + 0.5 * 0.5 = 0.25 + 0.25 = 0.5
    assert abs(weight - 0.5) < 1e-9
