from __future__ import annotations

import igraph as ig

from isb_igraph.graph import safe_projected_centrality


def _small_graph() -> ig.Graph:
    g = ig.Graph(edges=[(0, 1), (1, 2), (2, 3)], directed=False)
    g.vs["name"] = ["j0", "j1", "j2", "j3"]
    g.es["weight"] = [1.0, 1.0, 1.0]
    return g


def test_centrality_computed_when_under_limits() -> None:
    g = _small_graph()
    df, notes = safe_projected_centrality(
        g,
        betweenness_threshold=10,
        betweenness_max_edges=10,
        closeness_threshold=10,
        closeness_max_edges=10,
    )
    assert notes["betweenness_computed"] is True
    assert notes["closeness_computed"] is True
    assert df["betweenness"].notna().all()


def test_centrality_skipped_when_edges_over_limit() -> None:
    g = _small_graph()
    df, notes = safe_projected_centrality(
        g,
        betweenness_threshold=10,
        betweenness_max_edges=2,
        closeness_threshold=10,
        closeness_max_edges=2,
    )
    assert notes["betweenness_computed"] is False
    assert notes["closeness_computed"] is False
    assert "betweenness_skipped_reason" in notes
    assert "closeness_skipped_reason" in notes
    assert df["betweenness"].isna().all()


def test_centrality_disabled_by_config() -> None:
    g = _small_graph()
    df, notes = safe_projected_centrality(
        g,
        betweenness_threshold=10,
        betweenness_max_edges=10,
        closeness_threshold=10,
        closeness_max_edges=10,
        compute_betweenness=False,
        compute_closeness=False,
    )
    assert notes["betweenness_computed"] is False
    assert notes["closeness_computed"] is False
    assert notes["betweenness_skipped_reason"] == "disabled_by_config"
    assert notes["closeness_skipped_reason"] == "disabled_by_config"
    assert df["betweenness"].isna().all()
    assert df["closeness"].isna().all()
