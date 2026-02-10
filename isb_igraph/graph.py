from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import igraph as ig
import numpy as np
import pandas as pd


@dataclass(slots=True)
class GraphArtifacts:
    graph: ig.Graph
    node_index: dict[str, int]


def build_bipartite_graph(
    jobs_df: pd.DataFrame,
    skills_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    directed: bool = False,
) -> GraphArtifacts:
    job_nodes = jobs_df[["job_id", "job_title"]].copy()
    job_nodes["node_type"] = "job"
    job_nodes = job_nodes.rename(columns={"job_id": "node_id", "job_title": "label"})

    skill_nodes = skills_df[["skill_id", "label"]].copy()
    skill_nodes["node_type"] = "skill"
    skill_nodes = skill_nodes.rename(columns={"skill_id": "node_id"})

    nodes_df = pd.concat([job_nodes, skill_nodes], ignore_index=True)
    nodes_df = nodes_df.drop_duplicates(subset=["node_id"], keep="first")

    node_index = {node_id: idx for idx, node_id in enumerate(nodes_df["node_id"].tolist())}

    edge_tuples: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    for row in edges_df.itertuples(index=False):
        source_idx = node_index.get(row.job_id)
        target_idx = node_index.get(row.skill_id)
        if source_idx is None or target_idx is None:
            continue
        edge_tuples.append((source_idx, target_idx))
        edge_weights.append(float(row.edge_weight))

    graph = ig.Graph(
        n=len(nodes_df),
        edges=edge_tuples,
        directed=directed,
    )
    graph.vs["name"] = nodes_df["node_id"].tolist()
    graph.vs["label"] = nodes_df["label"].fillna("").tolist()
    graph.vs["node_type"] = nodes_df["node_type"].tolist()
    graph.vs["is_job"] = [node_type == "job" for node_type in graph.vs["node_type"]]
    graph.es["edge_weight"] = edge_weights

    return GraphArtifacts(graph=graph, node_index=node_index)


def compute_bipartite_metrics(graph: ig.Graph) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_ids = graph.vs["name"]
    node_types = graph.vs["node_type"]
    degrees = graph.degree()
    strengths = graph.strength(weights=graph.es["edge_weight"])

    metrics_df = pd.DataFrame(
        {
            "node_id": node_ids,
            "node_type": node_types,
            "degree": degrees,
            "weighted_degree": strengths,
        }
    )
    job_metrics = metrics_df[metrics_df["node_type"] == "job"].copy()
    skill_metrics = metrics_df[metrics_df["node_type"] == "skill"].copy()
    return job_metrics, skill_metrics


def graph_component_summary(graph: ig.Graph) -> dict[str, Any]:
    comps = graph.connected_components(mode="weak")
    sizes = comps.sizes()
    largest = int(max(sizes)) if sizes else 0
    return {
        "num_components": int(len(comps)),
        "largest_component_size": largest,
    }


def safe_projected_centrality(
    job_graph: ig.Graph,
    betweenness_threshold: int,
    betweenness_max_edges: int,
    closeness_threshold: int,
    closeness_max_edges: int,
    compute_betweenness: bool = True,
    compute_closeness: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    node_ids = job_graph.vs["name"]
    degree = job_graph.degree()
    strength = job_graph.strength(weights=job_graph.es["weight"] if job_graph.ecount() else None)

    centrality_df = pd.DataFrame(
        {
            "job_id": node_ids,
            "degree": degree,
            "weighted_degree": strength,
            "betweenness": np.nan,
            "closeness": np.nan,
        }
    )

    notes: dict[str, Any] = {
        "betweenness_computed": False,
        "closeness_computed": False,
    }

    edge_count = job_graph.ecount()
    if not compute_betweenness:
        notes["betweenness_skipped_reason"] = "disabled_by_config"
    elif job_graph.vcount() <= betweenness_threshold and edge_count <= betweenness_max_edges:
        centrality_df["betweenness"] = job_graph.betweenness(weights=None)
        notes["betweenness_computed"] = True
    else:
        notes["betweenness_skipped_reason"] = (
            f"guarded_by_size(v={job_graph.vcount()},e={edge_count},"
            f"v_limit={betweenness_threshold},e_limit={betweenness_max_edges})"
        )

    comps = job_graph.connected_components(mode="weak")
    if len(comps) == 0:
        if not compute_closeness:
            notes["closeness_skipped_reason"] = "disabled_by_config"
        return centrality_df, notes

    giant = comps.giant()
    giant_v = giant.vcount()
    giant_e = giant.ecount()

    if not compute_closeness:
        notes["closeness_skipped_reason"] = "disabled_by_config"
    elif giant_v <= closeness_threshold and giant_e <= closeness_max_edges:
        closeness_values = giant.closeness(weights=None)
        closeness_by_node = dict(zip(giant.vs["name"], closeness_values))
        centrality_df["closeness"] = centrality_df["job_id"].map(closeness_by_node)
        notes["closeness_computed"] = True
    else:
        notes["closeness_skipped_reason"] = (
            f"guarded_by_size(giant_v={giant_v},giant_e={giant_e},"
            f"v_limit={closeness_threshold},e_limit={closeness_max_edges})"
        )

    return centrality_df, notes
