from __future__ import annotations

import argparse
import json
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_NODES_COLS = {"node_id", "node_type", "label", "attributes_json"}
REQUIRED_SIM_COLS = {"job_id", "neighbor_job_id", "similarity_score", "rank", "method"}
REQUIRED_EDGES_COLS = {"source", "target", "weight", "attributes_json"}


def _safe_json_loads(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        if isinstance(value, str) and value.strip():
            return json.loads(value)
    except Exception:
        return {}
    return {}


def _ensure_dataframe(data: Path | pd.DataFrame, name: str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, Path):
        return pd.read_csv(data)
    raise TypeError(f"Unsupported input type for {name}: {type(data)}")


def load_jobs_catalog(nodes_source: Path | pd.DataFrame) -> pd.DataFrame:
    nodes_df = _ensure_dataframe(nodes_source, "nodes")
    missing = REQUIRED_NODES_COLS - set(nodes_df.columns)
    if missing:
        raise ValueError(f"nodes.csv missing columns: {sorted(missing)}")

    jobs = nodes_df[nodes_df["node_type"].astype(str) == "job"].copy()
    jobs = jobs.rename(columns={"node_id": "job_id", "label": "job_title"})

    attrs = jobs["attributes_json"].apply(_safe_json_loads)
    jobs["company_name"] = attrs.apply(lambda x: x.get("company_name"))
    jobs["district"] = attrs.apply(lambda x: x.get("district"))
    jobs["assigned_occupation_group"] = attrs.apply(lambda x: x.get("assigned_occupation_group"))
    jobs["job_title_norm"] = jobs["job_title"].astype(str).str.strip().str.lower()
    jobs["company_name_norm"] = jobs["company_name"].astype(str).str.strip().str.lower()

    return jobs[
        [
            "job_id",
            "job_title",
            "company_name",
            "district",
            "assigned_occupation_group",
            "job_title_norm",
            "company_name_norm",
        ]
    ]


def load_similarity(similarity_source: Path | pd.DataFrame) -> pd.DataFrame:
    similarity_df = _ensure_dataframe(similarity_source, "similarity")
    missing = REQUIRED_SIM_COLS - set(similarity_df.columns)
    if missing:
        raise ValueError(f"job_similarity_topk.csv missing columns: {sorted(missing)}")
    similarity_df["rank"] = similarity_df["rank"].astype(int)
    similarity_df["similarity_score"] = similarity_df["similarity_score"].astype(float)
    similarity_df["method"] = similarity_df["method"].astype(str)
    return similarity_df


def load_edges_for_explainability(edges_source: Path | pd.DataFrame) -> pd.DataFrame:
    edges_df = _ensure_dataframe(edges_source, "edges")
    missing = REQUIRED_EDGES_COLS - set(edges_df.columns)
    if missing:
        raise ValueError(f"edges.csv missing columns: {sorted(missing)}")

    out = edges_df[["source", "target", "weight", "attributes_json"]].copy()
    out["job_id"] = out["source"].astype(str)
    out["skill_id"] = out["target"].astype(str)
    out["edge_weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    attrs = out["attributes_json"].apply(_safe_json_loads)
    out["skill_text_normalized"] = attrs.apply(
        lambda x: str(x.get("skill_text_normalized", "")).strip().lower()
    )
    out = out[out["job_id"].str.len() > 0].copy()
    out = out[out["skill_id"].str.len() > 0].copy()
    out = out[out["skill_text_normalized"].str.len() > 0].copy()
    return out[["job_id", "skill_id", "skill_text_normalized", "edge_weight"]]


def _build_job_skill_sets(edges_df: pd.DataFrame) -> dict[str, set[str]]:
    skill_sets: dict[str, set[str]] = {}
    for row in edges_df.itertuples(index=False):
        job_id = str(row.job_id)
        skill = str(row.skill_text_normalized).strip().lower()
        if not job_id or not skill:
            continue
        skill_sets.setdefault(job_id, set()).add(skill)
    return skill_sets


def resolve_query_job(
    jobs_df: pd.DataFrame,
    *,
    job_id: str | None,
    title_query: str | None,
    max_candidates: int = 10,
) -> tuple[str | None, pd.DataFrame, str]:
    if job_id:
        match = jobs_df[jobs_df["job_id"].astype(str) == str(job_id)].copy()
        if not match.empty:
            return str(job_id), match.head(1), "resolved_by_job_id"
        return None, pd.DataFrame(), f"job_id_not_found:{job_id}"

    if not title_query:
        return None, pd.DataFrame(), "no_query"

    query_norm = str(title_query).strip().lower()
    exact = jobs_df[jobs_df["job_title_norm"] == query_norm].copy()
    if len(exact) == 1:
        return str(exact.iloc[0]["job_id"]), exact.head(1), "resolved_by_exact_title"
    if len(exact) > 1:
        cols = ["job_id", "job_title", "company_name", "district"]
        return None, exact[cols].head(max_candidates), "ambiguous_exact_title"

    contains = jobs_df[jobs_df["job_title_norm"].str.contains(re.escape(query_norm), na=False)].copy()
    if len(contains) == 1:
        return str(contains.iloc[0]["job_id"]), contains.head(1), "resolved_by_partial_title"
    if len(contains) > 1:
        cols = ["job_id", "job_title", "company_name", "district"]
        return None, contains[cols].head(max_candidates), "ambiguous_partial_title"

    titles = jobs_df["job_title"].astype(str).tolist()
    fuzzy = get_close_matches(title_query, titles, n=max_candidates, cutoff=0.5)
    if fuzzy:
        fuzzy_df = jobs_df[jobs_df["job_title"].isin(fuzzy)][
            ["job_id", "job_title", "company_name", "district"]
        ].copy()
        return None, fuzzy_df, "title_not_found_fuzzy_suggestions"

    return None, pd.DataFrame(), "title_not_found"


def lookup_closest_jobs(
    similarity_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    *,
    query_job_id: str,
    method: str = "unweighted",
    limit: int = 20,
    edges_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = similarity_df[similarity_df["job_id"].astype(str) == str(query_job_id)].copy()
    if method != "all":
        rows = rows[rows["method"] == method].copy()

    if rows.empty:
        columns = [
            "job_id",
            "job_title",
            "neighbor_job_id",
            "neighbor_job_title",
            "neighbor_company_name",
            "similarity_score",
            "rank",
            "method",
        ]
        if edges_df is not None:
            columns.extend(["shared_skill_count", "shared_skills_preview"])
        return pd.DataFrame(columns=columns)

    if method == "all":
        rows = rows.sort_values(by=["method", "rank", "neighbor_job_id"], kind="stable")
        rows = rows.groupby("method", as_index=False, sort=False).head(limit)
    else:
        rows = rows.sort_values(by=["rank", "neighbor_job_id"], kind="stable").head(limit)

    job_meta = jobs_df[["job_id", "job_title"]].rename(
        columns={"job_id": "query_job_id", "job_title": "job_title"}
    )
    neighbor_meta = jobs_df[["job_id", "job_title", "company_name"]].rename(
        columns={
            "job_id": "neighbor_job_id",
            "job_title": "neighbor_job_title",
            "company_name": "neighbor_company_name",
        }
    )

    rows = rows.merge(job_meta, left_on="job_id", right_on="query_job_id", how="left")
    rows = rows.merge(neighbor_meta, on="neighbor_job_id", how="left")
    rows = rows.drop(columns=["query_job_id"])

    if edges_df is not None:
        skill_sets = _build_job_skill_sets(edges_df)
        query_set = skill_sets.get(str(query_job_id), set())
        shared_counts: list[int] = []
        shared_previews: list[str] = []

        for row in rows.itertuples(index=False):
            neighbor_set = skill_sets.get(str(row.neighbor_job_id), set())
            shared = sorted(query_set & neighbor_set)
            shared_counts.append(len(shared))
            shared_previews.append(", ".join(shared[:5]))

        rows["shared_skill_count"] = shared_counts
        rows["shared_skills_preview"] = shared_previews

    output_columns = [
        "job_id",
        "job_title",
        "neighbor_job_id",
        "neighbor_job_title",
        "neighbor_company_name",
        "similarity_score",
        "rank",
        "method",
    ]
    if edges_df is not None:
        output_columns.extend(["shared_skill_count", "shared_skills_preview"])

    return rows[output_columns]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isb-igraph-lookup",
        description="Lookup closest jobs from job_similarity_topk.csv by job_id or title.",
    )
    parser.add_argument("--nodes-csv", required=True, help="Path to nodes.csv")
    parser.add_argument("--similarity-csv", required=True, help="Path to job_similarity_topk.csv")
    parser.add_argument(
        "--edges-csv",
        default=None,
        help="Optional path to edges.csv for shared-skill explainability columns",
    )

    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--job-id", help="Query job_id")
    query_group.add_argument("--title", help="Query by job title (exact/partial/fuzzy suggestion)")

    parser.add_argument("--method", choices=["all", "weighted", "unweighted"], default="unweighted")
    parser.add_argument("--limit", type=int, default=20, help="Neighbors per method")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument("--max-candidates", type=int, default=10)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    jobs_df = load_jobs_catalog(Path(args.nodes_csv))
    sim_df = load_similarity(Path(args.similarity_csv))
    edge_df = load_edges_for_explainability(Path(args.edges_csv)) if args.edges_csv else None

    resolved_job_id, resolved_rows, reason = resolve_query_job(
        jobs_df,
        job_id=args.job_id,
        title_query=args.title,
        max_candidates=args.max_candidates,
    )

    if resolved_job_id is None:
        if not resolved_rows.empty:
            print(f"Query not uniquely resolved ({reason}). Candidate jobs:")
            print(resolved_rows.to_string(index=False))
        else:
            print(f"Query not resolved ({reason}).")
        raise SystemExit(2)

    lookup_df = lookup_closest_jobs(
        similarity_df=sim_df,
        jobs_df=jobs_df,
        query_job_id=resolved_job_id,
        method=args.method,
        limit=args.limit,
        edges_df=edge_df,
    )

    if lookup_df.empty:
        print(f"No neighbors found for job_id={resolved_job_id} with method={args.method}.")
        return

    if args.json:
        print(lookup_df.to_json(orient="records", indent=2))
    else:
        print(lookup_df.to_string(index=False))


if __name__ == "__main__":
    main()
