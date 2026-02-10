from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

import igraph as ig
import pandas as pd


@dataclass(slots=True)
class ProjectionResult:
    topk_df: pd.DataFrame
    pair_scores_unweighted: dict[tuple[str, str], float]
    pair_scores_weighted: dict[tuple[str, str], float]
    stats: dict[str, Any]


def _ordered_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def compute_projection_pair_scores(
    edges_df: pd.DataFrame,
    max_skill_degree: int,
    max_pairs_per_skill: int,
    max_total_pairs: int,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], dict[str, Any]]:
    pair_unweighted: dict[tuple[str, str], float] = {}
    pair_weighted: dict[tuple[str, str], float] = {}

    skipped_skills_high_degree = 0
    skipped_skill_edges_high_degree = 0
    skipped_skills_pair_cap = 0
    skipped_skill_edges_pair_cap = 0
    skipped_skills_total_cap = 0
    skipped_skill_edges_total_cap = 0
    processed_skills = 0
    pair_updates = 0

    required = {"job_id", "skill_id", "edge_weight"}
    missing = required - set(edges_df.columns)
    if missing:
        raise ValueError(f"edges_df missing columns: {sorted(missing)}")

    for _, group in edges_df.groupby("skill_id", sort=False):
        processed_skills += 1
        degree = len(group)
        if degree < 2:
            continue

        if degree > max_skill_degree:
            skipped_skills_high_degree += 1
            skipped_skill_edges_high_degree += degree
            continue

        pair_count_for_skill = (degree * (degree - 1)) // 2
        if max_pairs_per_skill > 0 and pair_count_for_skill > max_pairs_per_skill:
            skipped_skills_pair_cap += 1
            skipped_skill_edges_pair_cap += degree
            continue

        if max_total_pairs > 0 and (pair_updates + pair_count_for_skill) > max_total_pairs:
            skipped_skills_total_cap += 1
            skipped_skill_edges_total_cap += degree
            continue

        rows = list(zip(group["job_id"].astype(str).tolist(), group["edge_weight"].astype(float).tolist()))
        for (job_a, weight_a), (job_b, weight_b) in combinations(rows, 2):
            pair_updates += 1
            pair = _ordered_pair(job_a, job_b)
            pair_unweighted[pair] = pair_unweighted.get(pair, 0.0) + 1.0
            overlap = min(weight_a, weight_b)
            pair_weighted[pair] = pair_weighted.get(pair, 0.0) + overlap

    stats = {
        "processed_skills": processed_skills,
        "skipped_skills_high_degree": skipped_skills_high_degree,
        "skipped_skill_edges_high_degree": skipped_skill_edges_high_degree,
        "skipped_skills_pair_cap": skipped_skills_pair_cap,
        "skipped_skill_edges_pair_cap": skipped_skill_edges_pair_cap,
        "skipped_skills_total_cap": skipped_skills_total_cap,
        "skipped_skill_edges_total_cap": skipped_skill_edges_total_cap,
        "pair_count_unweighted": len(pair_unweighted),
        "pair_count_weighted": len(pair_weighted),
        "pair_updates": pair_updates,
        "max_pairs_per_skill": max_pairs_per_skill,
        "max_total_pairs": max_total_pairs,
    }
    return pair_unweighted, pair_weighted, stats


def _build_topk_df(
    pair_scores: dict[tuple[str, str], float],
    method: str,
    top_k: int,
    similarity_threshold: float,
) -> pd.DataFrame:
    neighbors: dict[str, list[tuple[str, float]]] = {}
    for (job_a, job_b), score in pair_scores.items():
        if score < similarity_threshold:
            continue
        neighbors.setdefault(job_a, []).append((job_b, float(score)))
        neighbors.setdefault(job_b, []).append((job_a, float(score)))

    rows: list[dict[str, Any]] = []
    for job_id, candidate_list in neighbors.items():
        candidate_list.sort(key=lambda x: (-x[1], x[0]))
        if top_k > 0:
            candidate_list = candidate_list[:top_k]
        for rank, (neighbor_id, score) in enumerate(candidate_list, start=1):
            rows.append(
                {
                    "job_id": job_id,
                    "neighbor_job_id": neighbor_id,
                    "similarity_score": round(score, 6),
                    "rank": rank,
                    "method": method,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["job_id", "neighbor_job_id", "similarity_score", "rank", "method"])
    return pd.DataFrame(rows)


def build_job_similarity(
    edges_df: pd.DataFrame,
    method: str,
    top_k: int,
    similarity_threshold: float,
    max_skill_degree: int,
    max_pairs_per_skill: int,
    max_total_pairs: int,
) -> ProjectionResult:
    pair_unweighted, pair_weighted, stats = compute_projection_pair_scores(
        edges_df=edges_df,
        max_skill_degree=max_skill_degree,
        max_pairs_per_skill=max_pairs_per_skill,
        max_total_pairs=max_total_pairs,
    )

    frames: list[pd.DataFrame] = []
    if method in {"unweighted", "both"}:
        frames.append(
            _build_topk_df(
                pair_scores=pair_unweighted,
                method="unweighted",
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
        )
    if method in {"weighted", "both"}:
        frames.append(
            _build_topk_df(
                pair_scores=pair_weighted,
                method="weighted",
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
        )

    if frames:
        topk_df = pd.concat(frames, ignore_index=True)
    else:
        topk_df = pd.DataFrame(columns=["job_id", "neighbor_job_id", "similarity_score", "rank", "method"])

    return ProjectionResult(
        topk_df=topk_df,
        pair_scores_unweighted=pair_unweighted,
        pair_scores_weighted=pair_weighted,
        stats=stats,
    )


def build_projected_job_graph(
    all_job_ids: Iterable[str],
    pair_scores: dict[tuple[str, str], float],
    similarity_threshold: float = 0.0,
) -> ig.Graph:
    job_ids = sorted(set(str(job_id) for job_id in all_job_ids))
    index_by_job = {job_id: idx for idx, job_id in enumerate(job_ids)}

    edges: list[tuple[int, int]] = []
    weights: list[float] = []

    for (job_a, job_b), score in pair_scores.items():
        if score < similarity_threshold:
            continue
        idx_a = index_by_job.get(job_a)
        idx_b = index_by_job.get(job_b)
        if idx_a is None or idx_b is None or idx_a == idx_b:
            continue
        edges.append((idx_a, idx_b))
        weights.append(float(score))

    graph = ig.Graph(n=len(job_ids), edges=edges, directed=False)
    graph.vs["name"] = job_ids
    graph.es["weight"] = weights
    return graph
