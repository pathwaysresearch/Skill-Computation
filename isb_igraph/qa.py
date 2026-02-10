from __future__ import annotations

from typing import Any

import pandas as pd



def _check_no_null_endpoints(edges_export_df: pd.DataFrame) -> tuple[bool, str]:
    if edges_export_df.empty:
        return True, "edges empty"
    null_count = int(edges_export_df[["source", "target"]].isna().sum().sum())
    if null_count:
        return False, f"found {null_count} null source/target values"
    blank_mask = (
        edges_export_df["source"].astype(str).str.strip().eq("")
        | edges_export_df["target"].astype(str).str.strip().eq("")
    )
    blanks = int(blank_mask.sum())
    if blanks:
        return False, f"found {blanks} blank source/target values"
    return True, "ok"



def _check_edge_nodes_exist(nodes_export_df: pd.DataFrame, edges_export_df: pd.DataFrame) -> tuple[bool, str]:
    if edges_export_df.empty:
        return True, "edges empty"
    node_ids = set(nodes_export_df["node_id"].astype(str))
    missing_source = (~edges_export_df["source"].astype(str).isin(node_ids)).sum()
    missing_target = (~edges_export_df["target"].astype(str).isin(node_ids)).sum()
    missing_total = int(missing_source + missing_target)
    if missing_total:
        return False, f"{missing_total} edge endpoints missing from nodes"
    return True, "ok"



def _check_topk_rank_continuity(topk_df: pd.DataFrame) -> tuple[bool, str]:
    if topk_df.empty:
        return True, "top_k empty"
    for (job_id, method), group in topk_df.groupby(["job_id", "method"], sort=False):
        ranks = sorted(int(x) for x in group["rank"].tolist())
        expected = list(range(1, len(ranks) + 1))
        if ranks != expected:
            return False, f"rank gap for job_id={job_id}, method={method}"
    return True, "ok"



def run_data_quality_checks(
    nodes_export_df: pd.DataFrame,
    edges_export_df: pd.DataFrame,
    topk_df: pd.DataFrame,
) -> dict[str, Any]:
    checks = {
        "no_null_edge_endpoints": _check_no_null_endpoints(edges_export_df),
        "edge_endpoints_exist_in_nodes": _check_edge_nodes_exist(nodes_export_df, edges_export_df),
        "topk_rank_continuity": _check_topk_rank_continuity(topk_df),
    }

    details = {
        key: {"passed": passed, "message": message}
        for key, (passed, message) in checks.items()
    }
    all_passed = all(item["passed"] for item in details.values())

    return {
        "all_passed": all_passed,
        "checks": details,
    }
