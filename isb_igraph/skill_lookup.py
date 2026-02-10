from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

from .lookup import load_jobs_catalog, resolve_query_job
from .normalization import normalize_skill_text

REQUIRED_NODES_COLS = {"node_id", "node_type", "label", "attributes_json"}
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


def parse_skill_query(raw_skills: str) -> list[str]:
    parts = re.split(r"[,;\n]+", raw_skills or "")
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = part.strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(text)
    return out


def load_skills_catalog(nodes_source: Path | pd.DataFrame) -> pd.DataFrame:
    nodes_df = _ensure_dataframe(nodes_source, "nodes")
    missing = REQUIRED_NODES_COLS - set(nodes_df.columns)
    if missing:
        raise ValueError(f"nodes.csv missing columns: {sorted(missing)}")

    skills = nodes_df[nodes_df["node_type"].astype(str) == "skill"].copy()
    attrs = skills["attributes_json"].apply(_safe_json_loads)
    skills["skill_id"] = skills["node_id"].astype(str)
    skills["label"] = skills["label"].astype(str)
    skills["skill_text_normalized"] = attrs.apply(
        lambda x: normalize_skill_text(x.get("skill_text_normalized"))
    )
    missing_norm = skills["skill_text_normalized"].str.len() == 0
    skills.loc[missing_norm, "skill_text_normalized"] = skills.loc[missing_norm, "label"].apply(normalize_skill_text)

    skills = skills[["skill_id", "label", "skill_text_normalized"]].drop_duplicates(subset=["skill_id"], keep="first")
    skills = skills[skills["skill_text_normalized"].str.len() > 0].copy()
    return skills


def load_edges_catalog(edges_source: Path | pd.DataFrame) -> pd.DataFrame:
    edges_df = _ensure_dataframe(edges_source, "edges")
    missing = REQUIRED_EDGES_COLS - set(edges_df.columns)
    if missing:
        raise ValueError(f"edges.csv missing columns: {sorted(missing)}")

    out = edges_df[["source", "target", "weight", "attributes_json"]].copy()
    attrs = out["attributes_json"].apply(_safe_json_loads)

    out["job_id"] = out["source"].astype(str)
    out["skill_id"] = out["target"].astype(str)
    out["edge_weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    out["skill_text_normalized"] = attrs.apply(
        lambda x: normalize_skill_text(x.get("skill_text_normalized"))
    )
    out["bucket_normalized"] = attrs.apply(lambda x: x.get("bucket_normalized"))
    out["bucket_score"] = pd.to_numeric(
        attrs.apply(lambda x: x.get("bucket_score")), errors="coerce"
    )
    out["mapping_similarity"] = pd.to_numeric(
        attrs.apply(lambda x: x.get("mapping_similarity")), errors="coerce"
    )

    out = out[out["job_id"].str.len() > 0].copy()
    out = out[out["skill_id"].str.len() > 0].copy()

    return out[
        [
            "job_id",
            "skill_id",
            "skill_text_normalized",
            "edge_weight",
            "bucket_normalized",
            "bucket_score",
            "mapping_similarity",
        ]
    ]


def resolve_skills(
    query_skills: list[str],
    skills_df: pd.DataFrame,
    *,
    fuzzy_cutoff: int = 85,
    suggestion_limit: int = 5,
) -> dict[str, Any]:
    normalized_map: dict[str, dict[str, str]] = {}
    for row in skills_df.itertuples(index=False):
        normalized = str(row.skill_text_normalized)
        if normalized not in normalized_map:
            normalized_map[normalized] = {
                "skill_id": str(row.skill_id),
                "label": str(row.label),
                "skill_text_normalized": normalized,
            }

    catalog_norm_values = sorted(normalized_map.keys())

    resolved_rows: list[dict[str, str]] = []
    unresolved_rows: list[dict[str, str]] = []
    suggestions: dict[str, list[str]] = {}
    seen_skill_ids: set[str] = set()

    for raw_query in query_skills:
        normalized_query = normalize_skill_text(raw_query)
        if not normalized_query:
            continue

        hit = normalized_map.get(normalized_query)
        if hit:
            skill_id = hit["skill_id"]
            if skill_id in seen_skill_ids:
                continue
            seen_skill_ids.add(skill_id)
            resolved_rows.append(
                {
                    "query": raw_query,
                    "skill_id": skill_id,
                    "skill_text_normalized": hit["skill_text_normalized"],
                    "label": hit["label"],
                    "match_type": "exact_normalized",
                }
            )
            continue

        unresolved_rows.append({"query": raw_query, "query_normalized": normalized_query})
        fuzzy_hits = process.extract(
            normalized_query,
            catalog_norm_values,
            scorer=fuzz.ratio,
            limit=max(1, suggestion_limit),
        )
        candidate_norm = [cand for cand, score, _ in fuzzy_hits if score >= fuzzy_cutoff]
        suggestions[raw_query] = [normalized_map[cand]["label"] for cand in candidate_norm]

    resolved_df = pd.DataFrame(
        resolved_rows,
        columns=["query", "skill_id", "skill_text_normalized", "label", "match_type"],
    )
    unresolved_df = pd.DataFrame(unresolved_rows, columns=["query", "query_normalized"])

    return {
        "resolved_skills": resolved_df,
        "unresolved_skills": unresolved_df,
        "suggestions": suggestions,
    }


def rank_jobs_for_skills_or(
    edges_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    resolved_skill_ids: list[str],
    *,
    total_query_skills: int,
    limit: int = 20,
) -> pd.DataFrame:
    output_columns = [
        "job_id",
        "job_title",
        "matched_skill_count",
        "total_query_skills",
        "matched_ratio",
        "matched_weight_sum",
        "matched_skills",
        "rank",
    ]

    if not resolved_skill_ids:
        return pd.DataFrame(columns=output_columns)

    filtered = edges_df[edges_df["skill_id"].astype(str).isin([str(x) for x in resolved_skill_ids])].copy()
    if filtered.empty:
        return pd.DataFrame(columns=output_columns)

    grouped = (
        filtered.groupby("job_id", sort=False)
        .agg(
            matched_skill_count=("skill_id", "nunique"),
            matched_weight_sum=("edge_weight", "sum"),
        )
        .reset_index()
    )

    matched_skills = (
        filtered.groupby("job_id", sort=False)["skill_text_normalized"]
        .apply(lambda s: sorted({str(v) for v in s if str(v).strip()}))
        .reset_index(name="matched_skills_list")
    )

    ranked = grouped.merge(matched_skills, on="job_id", how="left")
    ranked["matched_skills"] = ranked["matched_skills_list"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    ranked = ranked.drop(columns=["matched_skills_list"])

    query_denominator = max(1, int(total_query_skills))
    ranked["total_query_skills"] = int(total_query_skills)
    ranked["matched_ratio"] = ranked["matched_skill_count"].astype(float) / float(query_denominator)

    jobs_meta = jobs_df[["job_id", "job_title"]].copy()
    ranked = ranked.merge(jobs_meta, on="job_id", how="left")

    ranked["matched_weight_sum"] = ranked["matched_weight_sum"].astype(float)
    ranked["matched_ratio"] = ranked["matched_ratio"].round(6)
    ranked = ranked.sort_values(
        by=["matched_skill_count", "matched_weight_sum", "job_id"],
        ascending=[False, False, True],
        kind="stable",
    )

    if limit > 0:
        ranked = ranked.head(limit)

    ranked = ranked.reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    return ranked[output_columns]


def get_job_top_skills(
    edges_df: pd.DataFrame,
    skills_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    *,
    query_job_id: str,
    limit: int = 20,
) -> pd.DataFrame:
    output_columns = [
        "job_id",
        "job_title",
        "skill_id",
        "skill_label",
        "skill_text_normalized",
        "edge_weight",
        "bucket_normalized",
        "mapping_similarity",
        "rank",
    ]

    rows = edges_df[edges_df["job_id"].astype(str) == str(query_job_id)].copy()
    if rows.empty:
        return pd.DataFrame(columns=output_columns)

    skills_meta = skills_df[["skill_id", "label", "skill_text_normalized"]].rename(
        columns={"label": "skill_label", "skill_text_normalized": "skill_text_normalized_catalog"}
    )
    rows = rows.merge(skills_meta, on="skill_id", how="left")

    missing_norm = rows["skill_text_normalized"].astype(str).str.len() == 0
    rows.loc[missing_norm, "skill_text_normalized"] = rows.loc[missing_norm, "skill_text_normalized_catalog"]

    rows = rows.sort_values(
        by=["edge_weight", "skill_text_normalized", "skill_id"],
        ascending=[False, True, True],
        kind="stable",
    )

    if limit > 0:
        rows = rows.head(limit)

    rows = rows.reset_index(drop=True)
    rows["rank"] = rows.index + 1

    jobs_meta = jobs_df[["job_id", "job_title"]].copy()
    rows = rows.merge(jobs_meta, on="job_id", how="left")

    return rows[output_columns]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isb-igraph-skill-lookup",
        description="Skill-to-job OR lookup and top-skills-by-job lookup from exported graph files.",
    )
    parser.add_argument("--nodes-csv", required=True, help="Path to nodes.csv")
    parser.add_argument("--edges-csv", required=True, help="Path to edges.csv")

    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--skills", help='Comma-separated skill query, e.g. "sql, python"')
    query_group.add_argument("--job-id", help="Query job_id for top skills")
    query_group.add_argument("--title", help="Query job title for top skills")

    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--fuzzy-cutoff", type=int, default=85)
    parser.add_argument("--suggestion-limit", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    jobs_df = load_jobs_catalog(Path(args.nodes_csv))
    skills_df = load_skills_catalog(Path(args.nodes_csv))
    edges_df = load_edges_catalog(Path(args.edges_csv))

    if args.skills:
        query_skills = parse_skill_query(args.skills)
        resolution = resolve_skills(
            query_skills,
            skills_df,
            fuzzy_cutoff=args.fuzzy_cutoff,
            suggestion_limit=args.suggestion_limit,
        )

        resolved_df = resolution["resolved_skills"]
        unresolved_df = resolution["unresolved_skills"]
        suggestions = resolution["suggestions"]

        if resolved_df.empty:
            payload = {
                "resolved_skills": [],
                "unresolved_skills": unresolved_df.to_dict(orient="records"),
                "suggestions": suggestions,
                "results": [],
            }
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print("No skills were resolved from the query.")
                if not unresolved_df.empty:
                    print("Unresolved:")
                    print(unresolved_df.to_string(index=False))
                if suggestions:
                    print("Suggestions:")
                    for query, options in suggestions.items():
                        print(f"- {query}: {', '.join(options) if options else 'none'}")
            raise SystemExit(2)

        ranked_df = rank_jobs_for_skills_or(
            edges_df=edges_df,
            jobs_df=jobs_df,
            resolved_skill_ids=resolved_df["skill_id"].astype(str).tolist(),
            total_query_skills=len(query_skills),
            limit=args.limit,
        )

        payload = {
            "resolved_skills": resolved_df.to_dict(orient="records"),
            "unresolved_skills": unresolved_df.to_dict(orient="records"),
            "suggestions": suggestions,
            "results": ranked_df.to_dict(orient="records"),
        }

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print("Resolved skills:")
            print(resolved_df[["query", "label", "skill_text_normalized"]].to_string(index=False))
            if not unresolved_df.empty:
                print("\nUnresolved skills:")
                print(unresolved_df.to_string(index=False))
                print("Suggestions:")
                for query, options in suggestions.items():
                    if options:
                        print(f"- {query}: {', '.join(options)}")
            print("\nClosest jobs by skill OR:")
            if ranked_df.empty:
                print("No jobs matched the resolved skills.")
            else:
                print(ranked_df.to_string(index=False))
        return

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

    top_skills_df = get_job_top_skills(
        edges_df=edges_df,
        skills_df=skills_df,
        jobs_df=jobs_df,
        query_job_id=resolved_job_id,
        limit=args.limit,
    )

    if top_skills_df.empty:
        print(f"No skills found for job_id={resolved_job_id}.")
        return

    if args.json:
        print(top_skills_df.to_json(orient="records", indent=2))
    else:
        print(top_skills_df.to_string(index=False))


if __name__ == "__main__":
    main()
