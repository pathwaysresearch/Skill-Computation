from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import igraph as ig
import pandas as pd


@dataclass(slots=True)
class ExportGraph:
    graph: ig.Graph
    node_index: dict[str, int]


REQUIRED_NODES_COLS = {"node_id", "node_type", "label"}
REQUIRED_EDGES_COLS = {"source", "target", "weight"}


def build_graph_from_exports(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> ExportGraph:
    missing_nodes = REQUIRED_NODES_COLS - set(nodes_df.columns)
    if missing_nodes:
        raise ValueError(f"nodes.csv missing columns: {sorted(missing_nodes)}")

    missing_edges = REQUIRED_EDGES_COLS - set(edges_df.columns)
    if missing_edges:
        raise ValueError(f"edges.csv missing columns: {sorted(missing_edges)}")

    nodes = nodes_df[["node_id", "node_type", "label"]].copy()
    nodes["node_id"] = nodes["node_id"].astype(str)
    nodes = nodes.drop_duplicates(subset=["node_id"], keep="first")

    node_index = {node_id: idx for idx, node_id in enumerate(nodes["node_id"].tolist())}

    edge_tuples: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    valid_edge_rows = edges_df[["source", "target", "weight"]].copy()
    valid_edge_rows["source"] = valid_edge_rows["source"].astype(str)
    valid_edge_rows["target"] = valid_edge_rows["target"].astype(str)
    valid_edge_rows["weight"] = pd.to_numeric(valid_edge_rows["weight"], errors="coerce").fillna(1.0)

    for row in valid_edge_rows.itertuples(index=False):
        src_idx = node_index.get(row.source)
        tgt_idx = node_index.get(row.target)
        if src_idx is None or tgt_idx is None or src_idx == tgt_idx:
            continue
        edge_tuples.append((src_idx, tgt_idx))
        edge_weights.append(float(row.weight))

    graph = ig.Graph(n=len(nodes), edges=edge_tuples, directed=False)
    graph.vs["name"] = nodes["node_id"].tolist()
    graph.vs["node_type"] = nodes["node_type"].astype(str).tolist()
    graph.vs["label"] = nodes["label"].astype(str).tolist()
    graph.es["weight"] = edge_weights

    return ExportGraph(graph=graph, node_index=node_index)


def find_job_matches(
    nodes_df: pd.DataFrame,
    *,
    job_id_query: str | None,
    title_query: str | None,
    limit: int = 50,
) -> pd.DataFrame:
    jobs = nodes_df[nodes_df["node_type"].astype(str) == "job"].copy()
    jobs["node_id"] = jobs["node_id"].astype(str)
    jobs["label"] = jobs["label"].astype(str)

    if job_id_query:
        q = job_id_query.strip()
        if q:
            return jobs[jobs["node_id"] == q].head(limit)

    if title_query:
        q = title_query.strip().lower()
        if q:
            return jobs[jobs["label"].str.lower().str.contains(q, na=False)].head(limit)

    return jobs.head(limit)


def bfs_ego_nodes(
    graph: ig.Graph,
    center_idx: int,
    *,
    max_hops: int,
    max_nodes: int,
) -> list[int]:
    if center_idx < 0 or center_idx >= graph.vcount():
        return []
    max_nodes = max(1, max_nodes)
    max_hops = max(0, max_hops)

    visited: set[int] = {center_idx}
    queue: deque[tuple[int, int]] = deque([(center_idx, 0)])
    out: list[int] = [center_idx]

    while queue and len(out) < max_nodes:
        node, hops = queue.popleft()
        if hops >= max_hops:
            continue

        for nbr in graph.neighbors(node):
            if nbr in visited:
                continue
            visited.add(nbr)
            out.append(nbr)
            if len(out) >= max_nodes:
                break
            queue.append((nbr, hops + 1))

    return out


def top_components(graph: ig.Graph, limit: int = 20) -> list[tuple[int, int]]:
    comps = graph.connected_components(mode="weak")
    comp_sizes = [(idx, len(comp)) for idx, comp in enumerate(comps)]
    comp_sizes.sort(key=lambda x: x[1], reverse=True)
    return comp_sizes[:limit]


def component_nodes(
    graph: ig.Graph,
    component_index: int,
    *,
    max_nodes: int,
) -> list[int]:
    comps = graph.connected_components(mode="weak")
    if component_index < 0 or component_index >= len(comps):
        return []

    nodes = list(comps[component_index])
    if len(nodes) <= max_nodes:
        return nodes

    # Keep the most connected nodes if clipping is needed.
    sub = graph.induced_subgraph(nodes)
    degrees = sub.degree()
    ranked = sorted(zip(nodes, degrees), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:max_nodes]]


def sample_global_nodes(
    graph: ig.Graph,
    *,
    max_nodes: int,
    seed: int = 42,
    high_degree_ratio: float = 0.35,
) -> list[int]:
    max_nodes = max(1, int(max_nodes))
    if graph.vcount() <= max_nodes:
        return list(range(graph.vcount()))

    degrees = graph.degree()
    names = graph.vs["name"] if "name" in graph.vs.attributes() else [str(i) for i in range(graph.vcount())]
    node_types = (
        graph.vs["node_type"]
        if "node_type" in graph.vs.attributes()
        else ["unknown"] * graph.vcount()
    )

    by_type: dict[str, list[int]] = {}
    for idx, node_type in enumerate(node_types):
        by_type.setdefault(str(node_type), []).append(idx)

    total_nodes = graph.vcount()
    allocations: dict[str, int] = {}
    for node_type, idxs in by_type.items():
        allocations[node_type] = max(1, int(round(max_nodes * len(idxs) / total_nodes)))

    allocated_total = sum(allocations.values())
    while allocated_total > max_nodes:
        largest = max(allocations, key=allocations.get)
        if allocations[largest] <= 1:
            break
        allocations[largest] -= 1
        allocated_total -= 1
    while allocated_total < max_nodes:
        largest_group = max(by_type, key=lambda t: len(by_type[t]))
        allocations[largest_group] += 1
        allocated_total += 1

    rng = random.Random(seed)
    selected: list[int] = []

    for node_type, idxs in by_type.items():
        target = min(len(idxs), allocations.get(node_type, 0))
        if target <= 0:
            continue
        ranked = sorted(idxs, key=lambda i: (-degrees[i], str(names[i])))
        high_n = min(target, max(1, int(round(target * high_degree_ratio))))
        chosen = ranked[:high_n]

        remaining_pool = ranked[high_n:]
        need_more = target - len(chosen)
        if need_more > 0 and remaining_pool:
            pool_sorted = sorted(remaining_pool)
            if len(pool_sorted) <= need_more:
                chosen.extend(pool_sorted)
            else:
                chosen.extend(rng.sample(pool_sorted, need_more))

        selected.extend(chosen)

    selected = sorted(set(selected))
    if len(selected) < max_nodes:
        missing = max_nodes - len(selected)
        remaining = [idx for idx in range(graph.vcount()) if idx not in set(selected)]
        ranked_remaining = sorted(remaining, key=lambda i: (-degrees[i], str(names[i])))
        selected.extend(ranked_remaining[:missing])
    elif len(selected) > max_nodes:
        ranked_selected = sorted(selected, key=lambda i: (-degrees[i], str(names[i])))
        selected = ranked_selected[:max_nodes]

    return selected


def degree_distribution_frame(graph: ig.Graph) -> pd.DataFrame:
    node_types = (
        graph.vs["node_type"]
        if "node_type" in graph.vs.attributes()
        else ["unknown"] * graph.vcount()
    )
    names = graph.vs["name"] if "name" in graph.vs.attributes() else [str(i) for i in range(graph.vcount())]
    return pd.DataFrame(
        {
            "node_id": names,
            "node_type": [str(x) for x in node_types],
            "degree": graph.degree(),
        }
    )


def component_size_frame(graph: ig.Graph, limit: int = 50) -> pd.DataFrame:
    rows = [
        {"component_index": idx, "size": size}
        for idx, size in top_components(graph, limit=limit)
    ]
    return pd.DataFrame(rows, columns=["component_index", "size"])


def run_visualization_diagnostics(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    graph: ig.Graph,
    *,
    max_nodes: int = 300,
    max_edges: int = 1000,
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []

    def add_check(name: str, passed: bool, details: str) -> None:
        checks.append(
            {
                "check": name,
                "status": "pass" if passed else "fail",
                "details": details,
            }
        )

    missing_nodes = REQUIRED_NODES_COLS - set(nodes_df.columns)
    add_check(
        "nodes_columns",
        not missing_nodes,
        "ok" if not missing_nodes else f"missing: {sorted(missing_nodes)}",
    )

    missing_edges = REQUIRED_EDGES_COLS - set(edges_df.columns)
    add_check(
        "edges_columns",
        not missing_edges,
        "ok" if not missing_edges else f"missing: {sorted(missing_edges)}",
    )

    if {"node_id"}.issubset(nodes_df.columns):
        null_nodes = int(nodes_df["node_id"].isna().sum())
        add_check("null_node_ids", null_nodes == 0, f"null_count={null_nodes}")

    if {"source", "target"}.issubset(edges_df.columns):
        null_src = int(edges_df["source"].isna().sum())
        null_tgt = int(edges_df["target"].isna().sum())
        add_check("null_edge_endpoints", (null_src + null_tgt) == 0, f"source_null={null_src}, target_null={null_tgt}")

        node_set = set(nodes_df["node_id"].astype(str).tolist()) if "node_id" in nodes_df.columns else set()
        if node_set:
            missing_src = int((~edges_df["source"].astype(str).isin(node_set)).sum())
            missing_tgt = int((~edges_df["target"].astype(str).isin(node_set)).sum())
            add_check(
                "edge_endpoints_in_nodes",
                (missing_src + missing_tgt) == 0,
                f"missing_source={missing_src}, missing_target={missing_tgt}",
            )

    try:
        center_idx = 0 if graph.vcount() > 0 else -1
        ego_nodes = bfs_ego_nodes(graph, center_idx, max_hops=2, max_nodes=min(max_nodes, 100))
        ego_plot = subgraph_to_plot_data(graph, ego_nodes, max_edges=min(max_edges, 300))
        add_check("ego_plot_generation", not ego_plot["nodes"].empty, f"nodes={len(ego_plot['nodes'])}")
    except Exception as exc:
        add_check("ego_plot_generation", False, str(exc))

    try:
        comps = top_components(graph, limit=1)
        if not comps:
            add_check("component_plot_generation", False, "no components")
        else:
            comp_nodes = component_nodes(graph, component_index=comps[0][0], max_nodes=min(max_nodes, 200))
            comp_plot = subgraph_to_plot_data(graph, comp_nodes, max_edges=min(max_edges, 500))
            add_check("component_plot_generation", not comp_plot["nodes"].empty, f"nodes={len(comp_plot['nodes'])}")
    except Exception as exc:
        add_check("component_plot_generation", False, str(exc))

    try:
        global_nodes = sample_global_nodes(graph, max_nodes=max_nodes, seed=42)
        global_plot = subgraph_to_plot_data(graph, global_nodes, max_edges=max_edges)
        add_check("global_plot_generation", not global_plot["nodes"].empty, f"nodes={len(global_plot['nodes'])}")
    except Exception as exc:
        add_check("global_plot_generation", False, str(exc))

    return pd.DataFrame(checks)


def subgraph_to_plot_data(
    graph: ig.Graph,
    node_ids: list[int],
    *,
    max_edges: int,
) -> dict[str, Any]:
    if not node_ids:
        return {
            "nodes": pd.DataFrame(columns=["node_id", "label", "node_type", "x", "y", "degree"]),
            "edges": pd.DataFrame(columns=["source", "target", "weight"]),
            "clipped_edges": 0,
        }

    sub = graph.induced_subgraph(node_ids)

    if sub.vcount() == 1:
        layout_points = [(0.0, 0.0)]
    else:
        layout = sub.layout_fruchterman_reingold(weights=sub.es["weight"] if sub.ecount() else None, niter=300)
        layout_points = [tuple(point) for point in layout]

    node_rows: list[dict[str, Any]] = []
    for idx, vertex in enumerate(sub.vs):
        x, y = layout_points[idx]
        node_rows.append(
            {
                "node_id": vertex["name"],
                "label": vertex["label"],
                "node_type": vertex["node_type"],
                "x": float(x),
                "y": float(y),
                "degree": int(sub.degree(idx)),
            }
        )

    edge_rows: list[dict[str, Any]] = []
    for edge in sub.es:
        source = sub.vs[edge.source]["name"]
        target = sub.vs[edge.target]["name"]
        edge_rows.append(
            {
                "source": source,
                "target": target,
                "weight": float(edge["weight"]) if "weight" in edge.attributes() else 1.0,
            }
        )

    clipped = 0
    if len(edge_rows) > max_edges:
        edge_rows = sorted(edge_rows, key=lambda x: x["weight"], reverse=True)[:max_edges]
        clipped = len(sub.es) - max_edges

    return {
        "nodes": pd.DataFrame(node_rows),
        "edges": pd.DataFrame(edge_rows),
        "clipped_edges": clipped,
    }
