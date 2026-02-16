from __future__ import annotations

import json
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st

from isb_igraph.environment import assert_runtime_compatibility
from isb_igraph.jobs_client import (
    cancel_job,
    default_api_base_url,
    get_job,
    get_job_events,
    submit_job_from_gcs,
    submit_job_from_path,
    upload_and_submit_job,
)
from isb_igraph.lookup import (
    load_edges_for_explainability,
    load_jobs_catalog,
    load_similarity,
    lookup_closest_jobs,
    resolve_query_job,
)
from isb_igraph.skill_lookup import (
    get_job_top_skills,
    load_edges_catalog,
    load_skills_catalog,
    parse_skill_query,
    rank_jobs_for_skills_or,
    resolve_skills,
)
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


assert_runtime_compatibility()

st.set_page_config(page_title="ISB iGraph Deployment UI", layout="wide")
st.title(" iGraph UI")
st.caption("Async jobs API + worker for large-file graph computation, lookup, and visualization")

COMPUTE_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "compute_profile": "quick",
        "compute_betweenness_enabled": False,
        "compute_betweenness_max_vertices": 5_000,
        "compute_betweenness_max_edges": 200_000,
        "compute_closeness_enabled": True,
        "compute_closeness_max_vertices": 12_000,
        "compute_closeness_max_edges": 500_000,
    },
    "standard": {
        "compute_profile": "standard",
        "compute_betweenness_enabled": True,
        "compute_betweenness_max_vertices": 15_000,
        "compute_betweenness_max_edges": 1_000_000,
        "compute_closeness_enabled": True,
        "compute_closeness_max_vertices": 30_000,
        "compute_closeness_max_edges": 1_500_000,
    },
    "full": {
        "compute_profile": "full",
        "compute_betweenness_enabled": True,
        "compute_betweenness_max_vertices": 30_000,
        "compute_betweenness_max_edges": 2_000_000,
        "compute_closeness_enabled": True,
        "compute_closeness_max_vertices": 60_000,
        "compute_closeness_max_edges": 4_000_000,
    },
}


def _load_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(data))


def _load_artifact_dataframes(job_payload: dict[str, Any]) -> None:
    artifacts = job_payload.get("artifacts") or []
    artifact_by_name = {str(a.get("artifact_name")): a for a in artifacts if isinstance(a, dict)}

    def read_csv_artifact(name: str) -> pd.DataFrame | None:
        item = artifact_by_name.get(name)
        if not item:
            return None
        path = Path(str(item.get("file_path", "")))
        if not path.exists() or not path.is_file():
            return None
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    st.session_state["latest_nodes_df"] = read_csv_artifact("nodes")
    st.session_state["latest_edges_df"] = read_csv_artifact("edges")
    st.session_state["latest_similarity_df"] = read_csv_artifact("job_similarity_topk")


def _render_subgraph_chart(plot_data: dict[str, Any]) -> None:
    nodes_df: pd.DataFrame = plot_data["nodes"]
    edges_df: pd.DataFrame = plot_data["edges"]
    clipped_edges = int(plot_data.get("clipped_edges", 0))

    if nodes_df.empty:
        st.warning("No nodes available for the selected subgraph.")
        return

    node_pos = nodes_df.set_index("node_id")[["x", "y"]].to_dict(orient="index")

    edge_rows: list[dict[str, Any]] = []
    for row in edges_df.to_dict(orient="records"):
        src = row["source"]
        tgt = row["target"]
        src_pos = node_pos.get(src)
        tgt_pos = node_pos.get(tgt)
        if src_pos is None or tgt_pos is None:
            continue
        edge_rows.append(
            {
                "source": src,
                "target": tgt,
                "x": src_pos["x"],
                "y": src_pos["y"],
                "x2": tgt_pos["x"],
                "y2": tgt_pos["y"],
                "weight": float(row.get("weight", 1.0)),
            }
        )

    edge_plot_df = pd.DataFrame(edge_rows)

    base = alt.Chart(nodes_df)
    nodes_chart = (
        base.mark_circle(opacity=0.95)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color("node_type:N", legend=alt.Legend(title="Node Type")),
            size=alt.Size("degree_global:Q", legend=alt.Legend(title="Global Degree")),
            tooltip=[
                alt.Tooltip("node_id:N", title="Node ID"),
                alt.Tooltip("label:N", title="Label"),
                alt.Tooltip("node_type:N", title="Type"),
                alt.Tooltip("degree_global:Q", title="Global Degree"),
                alt.Tooltip("degree_local:Q", title="Local Degree (Subgraph)"),
                alt.Tooltip("degree_rendered:Q", title="Rendered Degree (After Edge Cap)"),
            ],
        )
        .properties(height=620)
    )

    chart = nodes_chart
    if not edge_plot_df.empty:
        edges_chart = (
            alt.Chart(edge_plot_df)
            .mark_rule(opacity=0.24, color="#8f8f8f")
            .encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                x2="x2:Q",
                y2="y2:Q",
                tooltip=[
                    alt.Tooltip("source:N", title="Source"),
                    alt.Tooltip("target:N", title="Target"),
                    alt.Tooltip("weight:Q", title="Weight", format=".4f"),
                ],
            )
            .properties(height=620)
        )
        chart = edges_chart + nodes_chart

    st.altair_chart(chart.interactive(), use_container_width=True)

    if clipped_edges > 0:
        st.info(f"Rendered top-weighted edges only. Clipped edges: {clipped_edges}")
        st.caption("Rendered degree can be lower than local/global degree when edge cap is active.")


def _render_global_overview(
    graph,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    *,
    default_max_nodes: int = 1000,
    default_max_edges: int = 3000,
) -> None:
    st.subheader("Global Graph Overview")

    degree_df = degree_distribution_frame(graph)
    components_df = component_size_frame(graph, limit=50)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Nodes", f"{graph.vcount():,}")
    c2.metric("Total Edges", f"{graph.ecount():,}")
    c3.metric("Components", f"{len(graph.connected_components(mode='weak')):,}")

    node_counts = (
        nodes_df["node_type"].astype(str).value_counts().rename_axis("node_type").reset_index(name="count")
    )
    node_count_chart = (
        alt.Chart(node_counts)
        .mark_bar()
        .encode(x=alt.X("node_type:N", title="Node Type"), y=alt.Y("count:Q", title="Count"), color="node_type:N")
        .properties(height=240)
    )

    degree_hist = (
        alt.Chart(degree_df)
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X("degree:Q", bin=alt.Bin(maxbins=40), title="Degree"),
            y=alt.Y("count():Q", title="Node Count"),
            color=alt.Color("node_type:N", title="Node Type"),
            tooltip=[alt.Tooltip("count():Q", title="Nodes")],
        )
        .properties(height=240)
    )

    st.altair_chart(node_count_chart, use_container_width=True)
    st.altair_chart(degree_hist, use_container_width=True)

    if not components_df.empty:
        comp_view = components_df.head(20).copy()
        comp_view["component_label"] = comp_view["component_index"].apply(lambda x: f"C{x}")
        comp_chart = (
            alt.Chart(comp_view)
            .mark_bar(color="#4E79A7")
            .encode(
                x=alt.X("component_label:N", sort="-y", title="Component"),
                y=alt.Y("size:Q", title="Size"),
                tooltip=["component_index:Q", "size:Q"],
            )
            .properties(height=240)
        )
        st.altair_chart(comp_chart, use_container_width=True)

    st.markdown("### Full/Global Render")
    st.caption("For large graphs, render sampled global view instead of forcing full browser rendering.")

    col1, col2, col3 = st.columns(3)
    with col1:
        render_mode = st.radio(
            "Render mode",
            ["Global sample (recommended)", "Attempt full graph render"],
            horizontal=False,
            key="viz_global_render_mode",
        )
    with col2:
        global_max_nodes = st.slider(
            "Max nodes",
            min_value=50,
            max_value=5000,
            value=int(default_max_nodes),
            step=50,
            key="viz_global_max_nodes",
        )
    with col3:
        global_max_edges = st.slider(
            "Max edges",
            min_value=100,
            max_value=15000,
            value=int(default_max_edges),
            step=100,
            key="viz_global_max_edges",
        )

    seed = st.number_input("Global sample seed", min_value=1, max_value=10_000_000, value=42, step=1)

    if render_mode == "Attempt full graph render":
        if graph.vcount() > 8000 or graph.ecount() > 60000:
            st.warning("Graph too large for safe full render. Falling back to sampled view.")
            selected_nodes = sample_global_nodes(graph, max_nodes=int(global_max_nodes), seed=int(seed))
        else:
            selected_nodes = list(range(graph.vcount()))
    else:
        selected_nodes = sample_global_nodes(graph, max_nodes=int(global_max_nodes), seed=int(seed))

    plot_data = subgraph_to_plot_data(graph, selected_nodes, max_edges=int(global_max_edges))

    cols = st.columns(3)
    cols[0].metric("Rendered Nodes", f"{len(plot_data['nodes']):,}")
    cols[1].metric("Rendered Edges", f"{len(plot_data['edges']):,}")
    cols[2].metric("Clipped Edges", f"{int(plot_data['clipped_edges']):,}")

    _render_subgraph_chart(plot_data)

    st.markdown("### One-Click Diagnostics")
    if st.button("Run visualization + computation diagnostics", key="viz_run_diagnostics"):
        diag_df = run_visualization_diagnostics(
            nodes_df,
            edges_df,
            graph,
            max_nodes=int(global_max_nodes),
            max_edges=int(global_max_edges),
        )
        st.dataframe(diag_df, use_container_width=True, hide_index=True)
        failed = int((diag_df["status"] == "fail").sum())
        if failed > 0:
            st.error(f"Diagnostics found {failed} failing checks.")
        else:
            st.success("All diagnostics passed.")


def _render_sidebar() -> tuple[str, dict[str, Any]]:
    with st.sidebar:
        st.header("Execution")
        api_base_url = st.text_input("Jobs API base URL", value=default_api_base_url())

        st.markdown("---")
        st.header("Computation")
        profile = st.selectbox("Compute profile", ["quick", "standard", "full"], index=1)
        preset = COMPUTE_PRESETS[profile]

        similarity_mode = st.selectbox("Similarity mode", ["both", "unweighted", "weighted"], index=0)
        top_k = st.number_input("Top-K neighbors", min_value=1, max_value=200, value=20, step=1)
        similarity_threshold = st.number_input("Similarity threshold", min_value=0.0, value=0.0, step=0.1)
        chunksize = st.number_input("Chunk size", min_value=1_000, max_value=200_000, value=25_000, step=1_000)
        projection_max_skill_degree = st.number_input(
            "Projection max skill degree", min_value=50, max_value=50_000, value=2_000, step=50
        )

        with st.expander("Advanced metric options", expanded=False):
            compute_betweenness_enabled = st.toggle("Compute betweenness", value=preset["compute_betweenness_enabled"])
            compute_betweenness_max_vertices = st.number_input(
                "Betweenness max vertices",
                min_value=100,
                max_value=500_000,
                value=int(preset["compute_betweenness_max_vertices"]),
                step=100,
            )
            compute_betweenness_max_edges = st.number_input(
                "Betweenness max edges",
                min_value=1_000,
                max_value=20_000_000,
                value=int(preset["compute_betweenness_max_edges"]),
                step=10_000,
            )

            compute_closeness_enabled = st.toggle("Compute closeness", value=preset["compute_closeness_enabled"])
            compute_closeness_max_vertices = st.number_input(
                "Closeness max vertices",
                min_value=100,
                max_value=500_000,
                value=int(preset["compute_closeness_max_vertices"]),
                step=100,
            )
            compute_closeness_max_edges = st.number_input(
                "Closeness max edges",
                min_value=1_000,
                max_value=20_000_000,
                value=int(preset["compute_closeness_max_edges"]),
                step=10_000,
            )

        st.markdown("---")
        st.header("Subset")
        subset_mode = st.toggle("Subset mode", value=False)
        subset_target_rows = st.number_input("Subset target rows (0 to disable)", min_value=0, value=0, step=1_000)
        subset_target_size_mb = st.number_input("Subset target size MB", min_value=10, value=100, step=10)
        subset_seed = st.number_input("Subset seed", min_value=1, value=42, step=1)

    options = {
        "compute_profile": profile,
        "similarity_method": similarity_mode,
        "similarity_mode": similarity_mode,
        "top_k": int(top_k),
        "similarity_threshold": float(similarity_threshold),
        "chunksize": int(chunksize),
        "projection_max_skill_degree": int(projection_max_skill_degree),
        "projection_max_pairs_per_skill": (1_000_000 if profile == "quick" else (1_500_000 if profile == "standard" else 3_000_000)),
        "projection_max_total_pairs": (8_000_000 if profile == "quick" else (25_000_000 if profile == "standard" else 70_000_000)),
        "compute_betweenness_enabled": bool(compute_betweenness_enabled),
        "compute_betweenness_max_vertices": int(compute_betweenness_max_vertices),
        "compute_betweenness_max_edges": int(compute_betweenness_max_edges),
        "compute_closeness_enabled": bool(compute_closeness_enabled),
        "compute_closeness_max_vertices": int(compute_closeness_max_vertices),
        "compute_closeness_max_edges": int(compute_closeness_max_edges),
        "subset_mode": bool(subset_mode),
        "subset_target_rows": int(subset_target_rows) if int(subset_target_rows) > 0 else None,
        "subset_target_size_mb": int(subset_target_size_mb),
        "subset_seed": int(subset_seed),
        "seed": int(subset_seed),
    }
    return api_base_url, options

def _run_tab(api_base_url: str, options: dict[str, Any]) -> None:
    st.subheader("Run (Async Jobs)")
    st.caption("Submit jobs to API/worker. Streamlit does not run heavy compute in-process.")

    mode = st.radio("Input mode", ["Server file path", "GCS URI", "Upload file"], horizontal=True)
    submit_col1, submit_col2 = st.columns(2)

    with submit_col1:
        if mode == "Server file path":
            input_path = st.text_input(
                "Server input path",
                value="/Users/srijan26/ISB_Work/isb-igraph /Merged_All_Sheets_post6_skill_salary_merged_skills_non_empty_2_from_xlsx.csv",
            )
            if st.button("Submit path job", type="primary"):
                try:
                    payload = submit_job_from_path(
                        input_path=Path(input_path),
                        options=options,
                        base_url=api_base_url,
                    )
                except Exception as exc:
                    st.error(f"Failed to submit job: {exc}")
                else:
                    st.session_state["latest_job_id"] = payload.get("job_id")
                    st.session_state["latest_job_payload"] = payload
                    st.success(f"Submitted job: {payload.get('job_id')}")
        elif mode == "GCS URI":
            st.info("Recommended for Cloud Run large files. Example: gs://my-bucket/path/jobs.csv")
            gcs_uri = st.text_input("GCS URI", value="gs://my-bucket/path/to/jobs.csv")
            if st.button("Submit GCS job", type="primary"):
                if not gcs_uri.strip():
                    st.error("Enter a gs:// URI")
                else:
                    try:
                        payload = submit_job_from_gcs(
                            gcs_uri=gcs_uri.strip(),
                            options=options,
                            base_url=api_base_url,
                        )
                    except Exception as exc:
                        st.error(f"Failed to submit GCS job: {exc}")
                    else:
                        st.session_state["latest_job_id"] = payload.get("job_id")
                        st.session_state["latest_job_payload"] = payload
                        st.success(f"Submitted job: {payload.get('job_id')}")
        else:
            uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
            part_size_mb = st.number_input("Upload chunk size MB", min_value=2, max_value=128, value=8, step=2)
            st.caption("For Cloud Run, use GCS URI mode for large files to avoid HTTP 413 limits.")
            if st.button("Upload + submit job", type="primary"):
                if uploaded is None:
                    st.error("Upload a file first.")
                else:
                    with tempfile.TemporaryDirectory(prefix="isb_upload_") as tmp_dir:
                        tmp_path = Path(tmp_dir) / uploaded.name
                        tmp_path.write_bytes(uploaded.getvalue())
                        try:
                            payload = upload_and_submit_job(
                                file_path=tmp_path,
                                options=options,
                                base_url=api_base_url,
                                part_size_mb=int(part_size_mb),
                            )
                        except Exception as exc:
                            st.error(f"Upload/submit failed: {exc}")
                        else:
                            st.session_state["latest_job_id"] = payload.get("job_id")
                            st.session_state["latest_job_payload"] = payload
                            st.success(f"Submitted job: {payload.get('job_id')}")

    with submit_col2:
        job_id = st.text_input("Track job ID", value=st.session_state.get("latest_job_id", ""))
        c1, c2, c3 = st.columns(3)
        if c1.button("Refresh status"):
            if not job_id.strip():
                st.error("Enter job_id")
            else:
                try:
                    payload = get_job(job_id.strip(), base_url=api_base_url)
                    st.session_state["latest_job_payload"] = payload
                    st.session_state["latest_job_id"] = payload.get("job_id")
                    if payload.get("status") == "completed":
                        _load_artifact_dataframes(payload)
                except Exception as exc:
                    st.error(f"Failed to fetch job: {exc}")
        if c2.button("Load events") and job_id.strip():
            try:
                events_payload = get_job_events(job_id.strip(), base_url=api_base_url, limit=300)
                st.session_state["latest_job_events"] = events_payload.get("events", [])
            except Exception as exc:
                st.error(f"Failed to fetch events: {exc}")
        if c3.button("Cancel") and job_id.strip():
            try:
                _ = cancel_job(job_id.strip(), base_url=api_base_url)
                st.warning(f"Cancellation requested for {job_id}")
            except Exception as exc:
                st.error(f"Failed to cancel job: {exc}")

    payload = st.session_state.get("latest_job_payload")
    if payload:
        st.markdown("### Latest Job")
        st.json(payload)

    events = st.session_state.get("latest_job_events")
    if events:
        st.markdown("### Recent Events")
        st.dataframe(pd.DataFrame(events), use_container_width=True)



def _metrics_tab() -> None:
    st.subheader("Metrics")
    latest_job = st.session_state.get("latest_job_payload")
    if not latest_job:
        st.info("No job loaded yet. Submit and refresh a job first.")
        return

    run_metadata = latest_job.get("run_metadata")
    if not run_metadata:
        st.warning("Run metadata not available yet. Job may still be running.")
        return

    st.markdown("### Graph Summary")
    st.json(run_metadata.get("graph_summary", {}))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Validation Report")
        st.json(run_metadata.get("validation_report", {}))
    with col2:
        st.markdown("### QA Report")
        st.json(run_metadata.get("qa_report", {}))

    profile_records = run_metadata.get("profile_records") or []
    if profile_records:
        st.markdown("### Profiling")
        st.dataframe(pd.DataFrame(profile_records), use_container_width=True, hide_index=True)


def _lookup_tab() -> None:
    st.subheader("Lookup")
    st.caption("Deterministic shared-skill based lookup.")

    source_mode = st.radio("Lookup source", ["Latest completed job", "Upload exported files"], horizontal=True)

    nodes_df: pd.DataFrame | None = None
    edges_df: pd.DataFrame | None = None
    sim_df: pd.DataFrame | None = None

    if source_mode == "Latest completed job":
        nodes_df = st.session_state.get("latest_nodes_df")
        edges_df = st.session_state.get("latest_edges_df")
        sim_df = st.session_state.get("latest_similarity_df")
        if nodes_df is None or edges_df is None:
            st.info("No latest artifacts loaded. Refresh a completed job first.")
            return
    else:
        upload_nodes = st.file_uploader("Upload nodes.csv", type=["csv"], key="lookup_nodes_upload")
        upload_edges = st.file_uploader("Upload edges.csv", type=["csv"], key="lookup_edges_upload")
        upload_similarity = st.file_uploader("Upload job_similarity_topk.csv", type=["csv"], key="lookup_sim_upload")
        if upload_nodes is None or upload_edges is None:
            st.info("Upload nodes.csv and edges.csv.")
            return
        nodes_df = _load_csv_bytes(upload_nodes.getvalue())
        edges_df = _load_csv_bytes(upload_edges.getvalue())
        if upload_similarity is not None:
            sim_df = _load_csv_bytes(upload_similarity.getvalue())

    jobs_catalog = load_jobs_catalog(nodes_df)
    skills_catalog = load_skills_catalog(nodes_df)
    edges_catalog = load_edges_catalog(edges_df)
    edges_explain = load_edges_for_explainability(edges_df)

    tab_a, tab_b, tab_c = st.tabs(["Job -> Closest Jobs", "Skill(s) -> Closest Jobs", "Job -> Top Skills"])

    with tab_a:
        if sim_df is None:
            st.info("Similarity file not available.")
        else:
            similarity_catalog = load_similarity(sim_df)
            method = st.selectbox("Method", ["unweighted", "weighted", "all"], index=0)
            limit = st.number_input("Limit", min_value=1, max_value=200, value=20, step=1)
            include_explainability = st.toggle("Include shared skill explainability", value=True)
            q1, q2 = st.columns(2)
            with q1:
                job_id_query = st.text_input("Job ID", key="lookup_job_id")
            with q2:
                title_query = st.text_input("Job title", key="lookup_job_title")

            if st.button("Run job lookup"):
                resolved_job_id, candidates_df, reason = resolve_query_job(
                    jobs_catalog,
                    job_id=job_id_query or None,
                    title_query=title_query or None,
                    max_candidates=20,
                )
                if resolved_job_id is None:
                    st.warning(f"Query not uniquely resolved ({reason}).")
                    if not candidates_df.empty:
                        st.dataframe(candidates_df, use_container_width=True, hide_index=True)
                else:
                    out = lookup_closest_jobs(
                        similarity_df=similarity_catalog,
                        jobs_df=jobs_catalog,
                        query_job_id=resolved_job_id,
                        method=str(method),
                        limit=int(limit),
                        edges_df=edges_explain if include_explainability else None,
                    )
                    st.success(f"Resolved job_id: {resolved_job_id}")
                    st.dataframe(out, use_container_width=True, hide_index=True)

    with tab_b:
        limit = st.number_input("Top jobs", min_value=1, max_value=500, value=20, step=1)
        fuzzy_cutoff = st.slider("Fuzzy cutoff", min_value=0, max_value=100, value=85, step=1)
        suggestion_limit = st.number_input("Suggestion limit", min_value=1, max_value=20, value=5)
        skills_input = st.text_area("Skills (comma-separated)", placeholder="problem solving, analytical thinking")

        if st.button("Run skill->job lookup"):
            queries = parse_skill_query(skills_input)
            if not queries:
                st.error("Enter at least one skill")
            else:
                resolution = resolve_skills(
                    queries,
                    skills_catalog,
                    fuzzy_cutoff=int(fuzzy_cutoff),
                    suggestion_limit=int(suggestion_limit),
                )
                resolved_df = resolution["resolved_skills"]
                unresolved_df = resolution["unresolved_skills"]
                suggestions = resolution["suggestions"]

                st.markdown("### Resolved")
                st.dataframe(resolved_df, use_container_width=True, hide_index=True)

                if not unresolved_df.empty:
                    st.markdown("### Unresolved")
                    st.dataframe(unresolved_df, use_container_width=True, hide_index=True)
                    rows = [{"query": q, "suggestions": ", ".join(vals) if vals else "none"} for q, vals in suggestions.items()]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                if not resolved_df.empty:
                    ranked = rank_jobs_for_skills_or(
                        edges_df=edges_catalog,
                        jobs_df=jobs_catalog,
                        resolved_skill_ids=resolved_df["skill_id"].astype(str).tolist(),
                        total_query_skills=len(queries),
                        limit=int(limit),
                    )
                    st.markdown("### Ranked Jobs")
                    st.dataframe(ranked, use_container_width=True, hide_index=True)

    with tab_c:
        q1, q2 = st.columns(2)
        with q1:
            job_id_query = st.text_input("Job ID", key="top_skills_job_id")
        with q2:
            title_query = st.text_input("Job title", key="top_skills_title")
        limit = st.number_input("Top skills", min_value=1, max_value=200, value=20, step=1)

        if st.button("Run job->skills lookup"):
            resolved_job_id, candidates_df, reason = resolve_query_job(
                jobs_catalog,
                job_id=job_id_query or None,
                title_query=title_query or None,
                max_candidates=20,
            )
            if resolved_job_id is None:
                st.warning(f"Query not resolved ({reason})")
                if not candidates_df.empty:
                    st.dataframe(candidates_df, use_container_width=True, hide_index=True)
            else:
                out = get_job_top_skills(
                    edges_df=edges_catalog,
                    skills_df=skills_catalog,
                    jobs_df=jobs_catalog,
                    query_job_id=resolved_job_id,
                    limit=int(limit),
                )
                st.dataframe(out, use_container_width=True, hide_index=True)


def _visualize_tab() -> None:
    st.subheader("Visualize")
    source_mode = st.radio("Graph source", ["Latest completed job", "Upload nodes + edges"], horizontal=True)

    nodes_df: pd.DataFrame | None = None
    edges_df: pd.DataFrame | None = None

    if source_mode == "Latest completed job":
        nodes_df = st.session_state.get("latest_nodes_df")
        edges_df = st.session_state.get("latest_edges_df")
        if nodes_df is None or edges_df is None:
            st.info("No latest artifacts loaded. Refresh a completed job first.")
            return
    else:
        upload_nodes = st.file_uploader("Upload nodes.csv", type=["csv"], key="viz_nodes_upload")
        upload_edges = st.file_uploader("Upload edges.csv", type=["csv"], key="viz_edges_upload")
        if not upload_nodes or not upload_edges:
            st.info("Upload nodes.csv and edges.csv")
            return
        nodes_df = _load_csv_bytes(upload_nodes.getvalue())
        edges_df = _load_csv_bytes(upload_edges.getvalue())

    export_graph = build_graph_from_exports(nodes_df=nodes_df, edges_df=edges_df)
    graph = export_graph.graph
    st.write(f"Graph loaded: **{graph.vcount():,}** nodes, **{graph.ecount():,}** edges")

    max_nodes = st.slider("Max nodes in rendered subgraph", min_value=20, max_value=5000, value=300, step=10)
    max_edges = st.slider("Max edges in rendered subgraph", min_value=50, max_value=15000, value=1500, step=50)

    mode = st.radio("Visualization mode", ["Ego graph", "Skill hub", "Top connected component", "Global overview"], horizontal=True)

    if mode == "Ego graph":
        c1, c2 = st.columns(2)
        with c1:
            job_id_query = st.text_input("Job ID filter")
        with c2:
            title_query = st.text_input("Job title filter")

        candidates = find_job_matches(nodes_df, job_id_query=job_id_query or None, title_query=title_query or None, limit=200)
        if candidates.empty:
            st.warning("No matching jobs found.")
            return

        options = [f"{row.node_id} | {row.label}" for row in candidates.itertuples(index=False)]
        selected = st.selectbox("Center job", options)
        selected_job_id = selected.split(" | ", 1)[0]
        hops = st.slider("Neighborhood hops", min_value=1, max_value=4, value=2)

        center_idx = export_graph.node_index.get(selected_job_id)
        if center_idx is None:
            st.error("Selected job not found in graph index")
            return

        sub_nodes = bfs_ego_nodes(graph, center_idx, max_hops=int(hops), max_nodes=int(max_nodes))
        plot_data = subgraph_to_plot_data(graph, sub_nodes, max_edges=int(max_edges))
        _render_subgraph_chart(plot_data)

    elif mode == "Skill hub":
        skill_nodes = nodes_df[nodes_df["node_type"].astype(str) == "skill"][["node_id", "label"]].copy()
        if skill_nodes.empty:
            st.warning("No skill nodes found.")
            return

        skill_query = st.text_input("Skill label filter", key="viz_skill_filter")
        if skill_query.strip():
            q = skill_query.strip().lower()
            skill_nodes = skill_nodes[skill_nodes["label"].astype(str).str.lower().str.contains(q, na=False)].copy()

        if skill_nodes.empty:
            st.warning("No skill matches the current filter.")
            return

        degree_map = dict(zip(graph.vs["name"], graph.degree()))
        skill_nodes["global_degree"] = (
            skill_nodes["node_id"].astype(str).map(degree_map).fillna(0).astype(int)
        )
        skill_nodes = skill_nodes.sort_values(
            by=["global_degree", "label", "node_id"],
            ascending=[False, True, True],
            kind="stable",
        ).head(300)

        options = [
            f"{row.node_id} | {row.label} | degree={row.global_degree}"
            for row in skill_nodes.itertuples(index=False)
        ]
        selected = st.selectbox("Center skill", options)
        selected_skill_id = selected.split(" | ", 1)[0]

        hops = st.slider("Skill neighborhood hops", min_value=1, max_value=3, value=1)

        center_idx = export_graph.node_index.get(selected_skill_id)
        if center_idx is None:
            st.error("Selected skill not found in graph index")
            return

        sub_nodes = bfs_ego_nodes(graph, center_idx, max_hops=int(hops), max_nodes=int(max_nodes))
        plot_data = subgraph_to_plot_data(graph, sub_nodes, max_edges=int(max_edges))
        _render_subgraph_chart(plot_data)

        linked_jobs = edges_df[edges_df["target"].astype(str) == str(selected_skill_id)][["source"]].copy()
        linked_jobs = linked_jobs.rename(columns={"source": "job_id"}).drop_duplicates(subset=["job_id"])
        job_labels = nodes_df[nodes_df["node_type"].astype(str) == "job"][["node_id", "label"]].rename(
            columns={"node_id": "job_id", "label": "job_title"}
        )
        linked_jobs = linked_jobs.merge(job_labels, on="job_id", how="left")
        st.markdown("### Jobs requiring selected skill")
        st.caption(f"Total linked jobs: {len(linked_jobs):,}")
        st.dataframe(linked_jobs.head(200), use_container_width=True, hide_index=True)

    elif mode == "Top connected component":
        comps = top_components(graph, limit=50)
        if not comps:
            st.warning("No components available")
            return
        labels = [f"Component {idx} (size={size:,})" for idx, size in comps]
        selected_label = st.selectbox("Component", labels)
        comp_idx = int(selected_label.split(" ")[1])
        sub_nodes = component_nodes(graph, component_index=comp_idx, max_nodes=int(max_nodes))
        plot_data = subgraph_to_plot_data(graph, sub_nodes, max_edges=int(max_edges))
        _render_subgraph_chart(plot_data)

    else:
        _render_global_overview(
            graph,
            nodes_df=nodes_df,
            edges_df=edges_df,
            default_max_nodes=min(1200, int(max_nodes)),
            default_max_edges=min(4000, int(max_edges)),
        )


api_base_url, options = _render_sidebar()

tab_run, tab_metrics, tab_lookup, tab_visualize = st.tabs(["Run", "Metrics", "Lookup", "Visualize"])

with tab_run:
    _run_tab(api_base_url, options)

with tab_metrics:
    _metrics_tab()

with tab_lookup:
    _lookup_tab()

with tab_visualize:
    _visualize_tab()

st.markdown("---")
st.caption(
    "Deployment mode: API + worker. For huge files, use 'Server file path' submit and keep UI focused on monitoring/results."
)
