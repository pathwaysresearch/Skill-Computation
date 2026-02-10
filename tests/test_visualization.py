from __future__ import annotations

import igraph as ig
import pandas as pd

from isb_igraph.visualization import (
    bfs_ego_nodes,
    build_graph_from_exports,
    component_nodes,
    component_size_frame,
    degree_distribution_frame,
    find_job_matches,
    run_visualization_diagnostics,
    sample_global_nodes,
    subgraph_to_plot_data,
    top_components,
)


def _sample_exports() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes_df = pd.DataFrame(
        [
            {"node_id": "job_1", "node_type": "job", "label": "Mechanic"},
            {"node_id": "job_2", "node_type": "job", "label": "Technician"},
            {"node_id": "skill_1", "node_type": "skill", "label": "Problem Solving"},
            {"node_id": "skill_2", "node_type": "skill", "label": "Driving"},
            {"node_id": "job_3", "node_type": "job", "label": "Clerk"},
        ]
    )
    edges_df = pd.DataFrame(
        [
            {"source": "job_1", "target": "skill_1", "weight": 0.9},
            {"source": "job_2", "target": "skill_1", "weight": 0.8},
            {"source": "job_2", "target": "skill_2", "weight": 0.7},
        ]
    )
    return nodes_df, edges_df


def test_build_graph_from_exports() -> None:
    nodes_df, edges_df = _sample_exports()
    export_graph = build_graph_from_exports(nodes_df, edges_df)

    assert export_graph.graph.vcount() == 5
    assert export_graph.graph.ecount() == 3
    assert "job_1" in export_graph.node_index


def test_find_job_matches() -> None:
    nodes_df, _ = _sample_exports()
    by_id = find_job_matches(nodes_df, job_id_query="job_1", title_query=None)
    by_title = find_job_matches(nodes_df, job_id_query=None, title_query="tech")

    assert len(by_id) == 1
    assert by_id.iloc[0]["label"] == "Mechanic"
    assert len(by_title) == 1
    assert by_title.iloc[0]["node_id"] == "job_2"


def test_bfs_ego_nodes_hops() -> None:
    nodes_df, edges_df = _sample_exports()
    export_graph = build_graph_from_exports(nodes_df, edges_df)

    center = export_graph.node_index["job_1"]
    hop1 = bfs_ego_nodes(export_graph.graph, center, max_hops=1, max_nodes=10)
    hop2 = bfs_ego_nodes(export_graph.graph, center, max_hops=2, max_nodes=10)

    names_hop1 = {export_graph.graph.vs[idx]["name"] for idx in hop1}
    names_hop2 = {export_graph.graph.vs[idx]["name"] for idx in hop2}

    assert names_hop1 == {"job_1", "skill_1"}
    assert "job_2" in names_hop2


def test_top_components_and_component_nodes() -> None:
    nodes_df, edges_df = _sample_exports()
    export_graph = build_graph_from_exports(nodes_df, edges_df)

    comps = top_components(export_graph.graph, limit=5)
    assert len(comps) >= 2
    assert comps[0][1] >= comps[1][1]

    comp_nodes = component_nodes(export_graph.graph, component_index=comps[0][0], max_nodes=2)
    assert len(comp_nodes) == 2


def test_subgraph_plot_data_edge_clipping() -> None:
    graph = ig.Graph.Full(5)
    graph.vs["name"] = [f"n{i}" for i in range(5)]
    graph.vs["label"] = [f"Node {i}" for i in range(5)]
    graph.vs["node_type"] = ["job", "job", "skill", "skill", "job"]
    graph.es["weight"] = [1.0] * graph.ecount()

    plot_data = subgraph_to_plot_data(graph, [0, 1, 2, 3, 4], max_edges=3)

    assert len(plot_data["nodes"]) == 5
    assert len(plot_data["edges"]) == 3
    assert int(plot_data["clipped_edges"]) == 7


def test_sample_global_nodes_is_deterministic_and_capped() -> None:
    graph = ig.Graph.Full(20)
    graph.vs["name"] = [f"n{i}" for i in range(20)]
    graph.vs["label"] = [f"Node {i}" for i in range(20)]
    graph.vs["node_type"] = ["job" if i % 2 == 0 else "skill" for i in range(20)]
    graph.es["weight"] = [1.0] * graph.ecount()

    first = sample_global_nodes(graph, max_nodes=8, seed=42)
    second = sample_global_nodes(graph, max_nodes=8, seed=42)

    assert len(first) == 8
    assert first == second


def test_degree_and_component_frames() -> None:
    nodes_df, edges_df = _sample_exports()
    export_graph = build_graph_from_exports(nodes_df, edges_df)

    degree_df = degree_distribution_frame(export_graph.graph)
    comps_df = component_size_frame(export_graph.graph, limit=5)

    assert len(degree_df) == export_graph.graph.vcount()
    assert {"node_id", "node_type", "degree"}.issubset(degree_df.columns)
    assert {"component_index", "size"}.issubset(comps_df.columns)


def test_run_visualization_diagnostics() -> None:
    nodes_df, edges_df = _sample_exports()
    export_graph = build_graph_from_exports(nodes_df, edges_df)

    diagnostics_df = run_visualization_diagnostics(
        nodes_df,
        edges_df,
        export_graph.graph,
        max_nodes=20,
        max_edges=40,
    )

    assert not diagnostics_df.empty
    assert {"check", "status", "details"}.issubset(diagnostics_df.columns)
    assert (diagnostics_df["status"] == "fail").sum() == 0
