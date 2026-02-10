from __future__ import annotations

import json
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from isb_igraph.config import PipelineConfig
from isb_igraph.lookup import (
    load_edges_for_explainability,
    load_jobs_catalog,
    load_similarity,
    lookup_closest_jobs,
    resolve_query_job,
)
from isb_igraph.pipeline import run_pipeline
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


st.set_page_config(page_title="ISB iGraph MVP", layout="wide")
st.title("Job-to-Job Closeness MVP (igraph)")
st.caption("Upload a job CSV, build Job-Skill bipartite graph, and export Job-Job similarity outputs.")


COMPUTE_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "compute_betweenness_enabled": False,
        "compute_betweenness_max_vertices": 5_000,
        "compute_betweenness_max_edges": 200_000,
        "compute_closeness_enabled": True,
        "compute_closeness_max_vertices": 12_000,
        "compute_closeness_max_edges": 500_000,
    },
    "balanced": {
        "compute_betweenness_enabled": True,
        "compute_betweenness_max_vertices": 15_000,
        "compute_betweenness_max_edges": 1_000_000,
        "compute_closeness_enabled": True,
        "compute_closeness_max_vertices": 30_000,
        "compute_closeness_max_edges": 1_500_000,
    },
    "deep": {
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
            size=alt.Size("degree:Q", legend=alt.Legend(title="Degree")),
            tooltip=[
                alt.Tooltip("node_id:N", title="Node ID"),
                alt.Tooltip("label:N", title="Label"),
                alt.Tooltip("node_type:N", title="Type"),
                alt.Tooltip("degree:Q", title="Degree"),
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
    st.caption(
        "For very large graphs, full rendering is unsafe. Use global sampled render for an 'all-graph' view."
    )

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

    seed = st.number_input(
        "Global sample seed",
        min_value=1,
        max_value=10_000_000,
        value=42,
        step=1,
        key="viz_global_seed",
    )

    if render_mode == "Attempt full graph render":
        if graph.vcount() > 8000 or graph.ecount() > 60000:
            st.warning(
                "Graph too large for safe full render in browser. Falling back to sampled global view."
            )
            selected_nodes = sample_global_nodes(
                graph,
                max_nodes=int(global_max_nodes),
                seed=int(seed),
            )
        else:
            selected_nodes = list(range(graph.vcount()))
    else:
        selected_nodes = sample_global_nodes(
            graph,
            max_nodes=int(global_max_nodes),
            seed=int(seed),
        )

    plot_data = subgraph_to_plot_data(
        graph,
        selected_nodes,
        max_edges=int(global_max_edges),
    )

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


def _render_pipeline_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.header("Pipeline Settings")
        similarity_method = st.selectbox("Similarity mode", ["both", "weighted", "unweighted"], index=0)
        top_k = st.number_input("Top-K neighbors", min_value=1, max_value=200, value=20, step=1)
        similarity_threshold = st.number_input(
            "Similarity threshold",
            min_value=0.0,
            max_value=99999.0,
            value=0.0,
            step=0.1,
        )
        chunksize = st.number_input(
            "Chunk size", min_value=1_000, max_value=200_000, value=25_000, step=1_000
        )
        projection_max_skill_degree = st.number_input(
            "Projection max skill degree",
            min_value=50,
            max_value=50_000,
            value=2_000,
            step=50,
        )

        st.markdown("---")
        st.header("Computation Controls")
        compute_profile = st.selectbox(
            "Computation profile",
            ["fast", "balanced", "deep"],
            index=1,
            help="Fast skips heavy metrics; deep attempts larger centrality computations.",
        )
        defaults = COMPUTE_PRESETS[compute_profile]

        with st.expander("Advanced metric options", expanded=False):
            compute_betweenness_enabled = st.toggle(
                "Compute betweenness",
                value=bool(defaults["compute_betweenness_enabled"]),
                key="sidebar_compute_betweenness_enabled",
            )
            compute_betweenness_max_vertices = st.number_input(
                "Betweenness max vertices",
                min_value=100,
                max_value=500_000,
                value=int(defaults["compute_betweenness_max_vertices"]),
                step=100,
                key="sidebar_compute_betweenness_max_vertices",
            )
            compute_betweenness_max_edges = st.number_input(
                "Betweenness max edges",
                min_value=1_000,
                max_value=20_000_000,
                value=int(defaults["compute_betweenness_max_edges"]),
                step=10_000,
                key="sidebar_compute_betweenness_max_edges",
            )

            compute_closeness_enabled = st.toggle(
                "Compute closeness",
                value=bool(defaults["compute_closeness_enabled"]),
                key="sidebar_compute_closeness_enabled",
            )
            compute_closeness_max_vertices = st.number_input(
                "Closeness max vertices",
                min_value=100,
                max_value=500_000,
                value=int(defaults["compute_closeness_max_vertices"]),
                step=100,
                key="sidebar_compute_closeness_max_vertices",
            )
            compute_closeness_max_edges = st.number_input(
                "Closeness max edges",
                min_value=1_000,
                max_value=20_000_000,
                value=int(defaults["compute_closeness_max_edges"]),
                step=10_000,
                key="sidebar_compute_closeness_max_edges",
            )

        st.markdown("---")
        subset_mode = st.toggle("Subset mode", value=False)
        subset_target_rows = st.number_input(
            "Subset target rows (0 to disable)",
            min_value=0,
            max_value=2_000_000,
            value=0,
            step=1_000,
        )
        subset_target_size_mb = st.number_input(
            "Subset target size MB",
            min_value=10,
            max_value=2_000,
            value=100,
            step=10,
        )
        subset_seed = st.number_input("Subset seed", min_value=1, max_value=10_000_000, value=42, step=1)

    return {
        "similarity_method": similarity_method,
        "top_k": int(top_k),
        "similarity_threshold": float(similarity_threshold),
        "chunksize": int(chunksize),
        "projection_max_skill_degree": int(projection_max_skill_degree),
        "compute_profile": compute_profile,
        "compute_betweenness_enabled": bool(compute_betweenness_enabled),
        "compute_betweenness_max_vertices": int(compute_betweenness_max_vertices),
        "compute_betweenness_max_edges": int(compute_betweenness_max_edges),
        "compute_closeness_enabled": bool(compute_closeness_enabled),
        "compute_closeness_max_vertices": int(compute_closeness_max_vertices),
        "compute_closeness_max_edges": int(compute_closeness_max_edges),
        "subset_mode": bool(subset_mode),
        "subset_target_rows": (int(subset_target_rows) if int(subset_target_rows) > 0 else None),
        "subset_target_size_mb": int(subset_target_size_mb),
        "subset_seed": int(subset_seed),
    }


def _render_lookup_tab() -> None:
    st.subheader("Deterministic Lookup")
    st.caption(
        "Shared-skill based ranking only. Default job closeness is unweighted shared-skill count, not random."
    )

    source_mode = st.radio(
        "Lookup source",
        ["Latest pipeline run", "Upload exported files"],
        horizontal=True,
        key="lookup_source_mode",
    )

    nodes_df: pd.DataFrame | None = None
    edges_df: pd.DataFrame | None = None
    sim_df: pd.DataFrame | None = None

    if source_mode == "Latest pipeline run":
        nodes_df = st.session_state.get("latest_nodes_df")
        edges_df = st.session_state.get("latest_edges_df")
        sim_df = st.session_state.get("latest_similarity_df")

        if nodes_df is None or edges_df is None:
            st.info("Run the pipeline first, or switch to upload mode.")
    else:
        upload_nodes = st.file_uploader("Upload nodes.csv", type=["csv"], key="lookup_nodes_upload")
        upload_edges = st.file_uploader("Upload edges.csv", type=["csv"], key="lookup_edges_upload")
        upload_similarity = st.file_uploader(
            "Upload job_similarity_topk.csv (optional for Job -> Closest Jobs)",
            type=["csv"],
            key="lookup_similarity_upload",
        )

        if upload_nodes is not None and upload_edges is not None:
            nodes_df = _load_csv_bytes(upload_nodes.getvalue())
            edges_df = _load_csv_bytes(upload_edges.getvalue())
        if upload_similarity is not None:
            sim_df = _load_csv_bytes(upload_similarity.getvalue())

        if nodes_df is None or edges_df is None:
            st.info("Upload nodes.csv and edges.csv to use lookup tools.")

    if nodes_df is None or edges_df is None:
        return

    try:
        jobs_catalog = load_jobs_catalog(nodes_df)
        skills_catalog = load_skills_catalog(nodes_df)
        edges_catalog = load_edges_catalog(edges_df)
        edges_explain = load_edges_for_explainability(edges_df)
    except Exception as exc:
        st.error(f"Failed to prepare lookup catalogs: {exc}")
        return

    st.write(
        f"Lookup catalogs loaded: **{len(jobs_catalog):,} jobs**, **{len(skills_catalog):,} skills**, "
        f"**{len(edges_catalog):,} job-skill edges**"
    )

    lookup_job_tab, lookup_skill_tab, lookup_top_skills_tab = st.tabs(
        ["Job -> Closest Jobs", "Skill(s) -> Closest Jobs", "Job -> Top Skills"]
    )

    with lookup_job_tab:
        if sim_df is None:
            st.info("Upload job_similarity_topk.csv or run pipeline to use this panel.")
        else:
            try:
                similarity_catalog = load_similarity(sim_df)
            except Exception as exc:
                st.error(f"Failed to parse similarity data: {exc}")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    method = st.selectbox(
                        "Method",
                        ["unweighted", "weighted", "all"],
                        index=0,
                        key="lookup_job_method",
                    )
                with col2:
                    limit = st.number_input(
                        "Neighbors per job",
                        min_value=1,
                        max_value=200,
                        value=20,
                        step=1,
                        key="lookup_job_limit",
                    )
                with col3:
                    include_explainability = st.toggle(
                        "Include shared-skill explainability",
                        value=True,
                        key="lookup_job_explain",
                    )

                jcol1, jcol2 = st.columns(2)
                with jcol1:
                    job_id_query = st.text_input("Job ID", key="lookup_job_id")
                with jcol2:
                    title_query = st.text_input("Job title", key="lookup_job_title")

                if st.button("Find closest jobs", key="lookup_job_run"):
                    resolved_job_id, candidates_df, reason = resolve_query_job(
                        jobs_catalog,
                        job_id=job_id_query or None,
                        title_query=title_query or None,
                        max_candidates=15,
                    )
                    if resolved_job_id is None:
                        if not candidates_df.empty:
                            st.warning(f"Query not uniquely resolved ({reason}). Pick a job_id from candidates.")
                            st.dataframe(candidates_df, use_container_width=True, hide_index=True)
                        else:
                            st.error(f"Query not resolved ({reason}).")
                    else:
                        closest_df = lookup_closest_jobs(
                            similarity_df=similarity_catalog,
                            jobs_df=jobs_catalog,
                            query_job_id=resolved_job_id,
                            method=method,
                            limit=int(limit),
                            edges_df=(edges_explain if include_explainability else None),
                        )
                        st.success(f"Resolved job_id: {resolved_job_id}")
                        if closest_df.empty:
                            st.info("No neighbors found.")
                        else:
                            st.dataframe(closest_df, use_container_width=True, hide_index=True)

    with lookup_skill_tab:
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            skill_limit = st.number_input(
                "Top jobs",
                min_value=1,
                max_value=500,
                value=20,
                step=1,
                key="lookup_skill_limit",
            )
        with scol2:
            fuzzy_cutoff = st.slider(
                "Fuzzy cutoff",
                min_value=0,
                max_value=100,
                value=85,
                step=1,
                key="lookup_skill_cutoff",
            )
        with scol3:
            suggestion_limit = st.number_input(
                "Suggestions per unresolved skill",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="lookup_skill_sugg_limit",
            )

        skill_query_text = st.text_area(
            "Skill query (comma-separated)",
            placeholder="problem solving, analytical thinking, machine operation",
            key="lookup_skill_query",
        )

        if st.button("Find jobs for skills", key="lookup_skill_run"):
            query_skills = parse_skill_query(skill_query_text)
            if not query_skills:
                st.error("Provide at least one skill.")
            else:
                resolution = resolve_skills(
                    query_skills,
                    skills_catalog,
                    fuzzy_cutoff=int(fuzzy_cutoff),
                    suggestion_limit=int(suggestion_limit),
                )
                resolved_df = resolution["resolved_skills"]
                unresolved_df = resolution["unresolved_skills"]
                suggestions = resolution["suggestions"]

                st.subheader("Resolved Skills")
                if resolved_df.empty:
                    st.warning("No skills resolved from query.")
                else:
                    st.dataframe(
                        resolved_df[["query", "label", "skill_text_normalized", "skill_id"]],
                        use_container_width=True,
                        hide_index=True,
                    )

                if not unresolved_df.empty:
                    st.subheader("Unresolved Skills")
                    st.dataframe(unresolved_df, use_container_width=True, hide_index=True)
                    suggestion_rows: list[dict[str, Any]] = []
                    for query_text, options in suggestions.items():
                        suggestion_rows.append(
                            {
                                "query": query_text,
                                "suggestions": ", ".join(options) if options else "none",
                            }
                        )
                    if suggestion_rows:
                        st.subheader("Suggestions")
                        st.dataframe(pd.DataFrame(suggestion_rows), use_container_width=True, hide_index=True)

                if not resolved_df.empty:
                    ranked_df = rank_jobs_for_skills_or(
                        edges_df=edges_catalog,
                        jobs_df=jobs_catalog,
                        resolved_skill_ids=resolved_df["skill_id"].astype(str).tolist(),
                        total_query_skills=len(query_skills),
                        limit=int(skill_limit),
                    )
                    st.subheader("Closest Jobs by Skill OR")
                    if ranked_df.empty:
                        st.info("No jobs matched the resolved skills.")
                    else:
                        st.dataframe(ranked_df, use_container_width=True, hide_index=True)

    with lookup_top_skills_tab:
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            top_job_id_query = st.text_input("Job ID", key="lookup_top_skills_job_id")
        with tcol2:
            top_title_query = st.text_input("Job title", key="lookup_top_skills_title")

        top_limit = st.number_input(
            "Top skills",
            min_value=1,
            max_value=200,
            value=20,
            step=1,
            key="lookup_top_skills_limit",
        )

        if st.button("Show top skills for job", key="lookup_top_skills_run"):
            resolved_job_id, candidates_df, reason = resolve_query_job(
                jobs_catalog,
                job_id=top_job_id_query or None,
                title_query=top_title_query or None,
                max_candidates=15,
            )
            if resolved_job_id is None:
                if not candidates_df.empty:
                    st.warning(f"Query not uniquely resolved ({reason}). Pick a job_id from candidates.")
                    st.dataframe(candidates_df, use_container_width=True, hide_index=True)
                else:
                    st.error(f"Query not resolved ({reason}).")
            else:
                top_skills_df = get_job_top_skills(
                    edges_df=edges_catalog,
                    skills_df=skills_catalog,
                    jobs_df=jobs_catalog,
                    query_job_id=resolved_job_id,
                    limit=int(top_limit),
                )
                st.success(f"Resolved job_id: {resolved_job_id}")
                if top_skills_df.empty:
                    st.info("No skills found for this job.")
                else:
                    st.dataframe(top_skills_df, use_container_width=True, hide_index=True)


settings = _render_pipeline_sidebar()

tab_pipeline, tab_visualizer, tab_lookup = st.tabs(["Pipeline", "Subgraph Visualizer", "Lookup"])

with tab_pipeline:
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="pipeline_csv_upload")
    run_clicked = st.button("Run Pipeline", type="primary", key="run_pipeline_btn")

    if run_clicked:
        if uploaded is None:
            st.error("Upload a CSV file first.")
        else:
            run_dir = Path(tempfile.mkdtemp(prefix="isb_igraph_run_"))
            input_path = run_dir / uploaded.name
            input_path.write_bytes(uploaded.getvalue())

            output_dir = run_dir / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            config = PipelineConfig(
                input_csv=input_path,
                output_dir=output_dir,
                chunksize=settings["chunksize"],
                top_k=settings["top_k"],
                similarity_threshold=settings["similarity_threshold"],
                similarity_method=settings["similarity_method"],
                projection_max_skill_degree=settings["projection_max_skill_degree"],
                compute_betweenness_enabled=settings["compute_betweenness_enabled"],
                compute_betweenness_max_vertices=settings["compute_betweenness_max_vertices"],
                compute_betweenness_max_edges=settings["compute_betweenness_max_edges"],
                compute_closeness_enabled=settings["compute_closeness_enabled"],
                compute_closeness_max_vertices=settings["compute_closeness_max_vertices"],
                compute_closeness_max_edges=settings["compute_closeness_max_edges"],
                subset_mode=settings["subset_mode"],
                subset_target_rows=settings["subset_target_rows"],
                subset_target_size_mb=settings["subset_target_size_mb"],
                subset_seed=settings["subset_seed"],
            )

            progress = st.progress(0)
            status = st.empty()
            logs = st.empty()
            progress_lines: list[str] = []

            def update_progress(message: str, fraction: float) -> None:
                progress.progress(max(0.0, min(1.0, fraction)))
                status.write(message)
                progress_lines.append(f"[{time.strftime('%H:%M:%S')}] {message}")
                logs.code("\n".join(progress_lines[-10:]))

            try:
                result = run_pipeline(config, progress_fn=update_progress)
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                st.stop()

            downloads: dict[str, tuple[str, bytes]] = {}
            for _, file_path in result.output_files.items():
                if file_path.exists():
                    downloads[file_path.name] = ("application/octet-stream", file_path.read_bytes())

            latest_nodes_path = result.output_files.get("nodes")
            latest_edges_path = result.output_files.get("edges")
            latest_similarity_path = result.output_files.get("job_similarity_topk")

            latest_nodes_df = pd.read_csv(latest_nodes_path) if latest_nodes_path and latest_nodes_path.exists() else None
            latest_edges_df = pd.read_csv(latest_edges_path) if latest_edges_path and latest_edges_path.exists() else None
            latest_similarity_df = (
                pd.read_csv(latest_similarity_path)
                if latest_similarity_path and latest_similarity_path.exists()
                else None
            )

            st.session_state["latest_result"] = {
                "graph_summary": result.graph_summary,
                "validation_report": result.validation_report,
                "qa_report": result.qa_report,
                "profile_records": result.profile_records,
                "output_dir": str(result.output_dir),
                "output_files": {k: str(v) for k, v in result.output_files.items()},
                "downloads": downloads,
                "compute_profile": settings["compute_profile"],
            }
            st.session_state["latest_nodes_df"] = latest_nodes_df
            st.session_state["latest_edges_df"] = latest_edges_df
            st.session_state["latest_similarity_df"] = latest_similarity_df

    latest_result = st.session_state.get("latest_result")
    if latest_result:
        st.success("Pipeline completed successfully")
        st.subheader("Graph Summary")
        st.json(latest_result["graph_summary"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Validation")
            st.json(latest_result["validation_report"])
        with col2:
            st.subheader("QA Checks")
            st.json(latest_result["qa_report"])

        st.subheader("Downloads")
        for file_name, (mime, data) in latest_result["downloads"].items():
            st.download_button(
                label=f"Download {file_name}",
                data=data,
                file_name=file_name,
                mime=mime,
                key=f"download_{file_name}",
            )

        st.subheader("Profiling")
        st.table(latest_result["profile_records"])

        st.subheader("Run Metadata")
        st.code(
            json.dumps(
                {
                    "output_dir": latest_result["output_dir"],
                    "files": list(latest_result["output_files"].values()),
                    "compute_profile": latest_result.get("compute_profile", "balanced"),
                },
                indent=2,
            )
        )
    else:
        st.info("Upload a CSV and run pipeline to generate outputs.")

with tab_visualizer:
    st.subheader("Interactive Subgraph Visualizer")
    st.caption(
        "Visualize graph slices, components, or global overview. Full graph rendering is constrained for stability."
    )

    source_mode = st.radio(
        "Graph source",
        ["Latest pipeline run", "Upload nodes.csv + edges.csv"],
        horizontal=True,
    )

    nodes_df: pd.DataFrame | None = None
    edges_df: pd.DataFrame | None = None

    if source_mode == "Latest pipeline run":
        nodes_df = st.session_state.get("latest_nodes_df")
        edges_df = st.session_state.get("latest_edges_df")
        if nodes_df is None or edges_df is None:
            st.info("Run the pipeline first, or switch to upload mode.")
    else:
        upload_nodes = st.file_uploader("Upload nodes.csv", type=["csv"], key="viz_nodes_upload")
        upload_edges = st.file_uploader("Upload edges.csv", type=["csv"], key="viz_edges_upload")
        if upload_nodes and upload_edges:
            nodes_df = _load_csv_bytes(upload_nodes.getvalue())
            edges_df = _load_csv_bytes(upload_edges.getvalue())
        else:
            st.info("Upload both nodes.csv and edges.csv to visualize.")

    if nodes_df is not None and edges_df is not None:
        try:
            export_graph = build_graph_from_exports(nodes_df=nodes_df, edges_df=edges_df)
        except Exception as exc:
            st.error(f"Failed to build graph from exports: {exc}")
        else:
            graph = export_graph.graph
            st.write(
                f"Graph loaded: **{graph.vcount():,}** nodes, **{graph.ecount():,}** edges"
            )
            if graph.vcount() > 200_000 or graph.ecount() > 1_000_000:
                st.warning(
                    "Very large graph detected. Prefer global sampled view or lower render caps."
                )

            max_nodes = st.slider("Max nodes in rendered subgraph", min_value=20, max_value=5000, value=300, step=10)
            max_edges = st.slider("Max edges in rendered subgraph", min_value=50, max_value=15000, value=1500, step=50)

            viz_mode = st.radio(
                "Visualization mode",
                ["Ego graph", "Top connected component", "Global overview"],
                horizontal=True,
            )

            if viz_mode == "Ego graph":
                col1, col2 = st.columns(2)
                with col1:
                    job_id_query = st.text_input("Job ID filter (optional)")
                with col2:
                    title_query = st.text_input("Job title filter (optional)")

                candidate_df = find_job_matches(
                    nodes_df,
                    job_id_query=job_id_query or None,
                    title_query=title_query or None,
                    limit=200,
                )
                if candidate_df.empty:
                    st.warning("No matching job nodes found with current filters.")
                else:
                    options = [f"{row.node_id} | {row.label}" for row in candidate_df.itertuples(index=False)]
                    selected = st.selectbox("Select center job", options)
                    selected_job_id = selected.split(" | ", 1)[0]
                    hops = st.slider("Neighborhood hops", min_value=1, max_value=4, value=2, step=1)

                    center_idx = export_graph.node_index.get(selected_job_id)
                    if center_idx is None:
                        st.error("Selected job not found in graph index.")
                    else:
                        sub_nodes = bfs_ego_nodes(
                            graph,
                            center_idx,
                            max_hops=int(hops),
                            max_nodes=int(max_nodes),
                        )
                        plot_data = subgraph_to_plot_data(
                            graph,
                            sub_nodes,
                            max_edges=int(max_edges),
                        )

                        cols = st.columns(3)
                        cols[0].metric("Rendered Nodes", f"{len(plot_data['nodes']):,}")
                        cols[1].metric("Rendered Edges", f"{len(plot_data['edges']):,}")
                        cols[2].metric("Clipped Edges", f"{int(plot_data['clipped_edges']):,}")

                        _render_subgraph_chart(plot_data)

                        st.subheader("Candidate Job Nodes")
                        st.dataframe(
                            candidate_df[["node_id", "label"]].rename(
                                columns={"node_id": "job_id", "label": "job_title"}
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

            elif viz_mode == "Top connected component":
                component_list = top_components(graph, limit=50)
                if not component_list:
                    st.warning("No components available in graph.")
                else:
                    labels = [f"Component {idx} (size={size:,})" for idx, size in component_list]
                    selected_label = st.selectbox("Select component", labels)
                    comp_idx = int(selected_label.split(" ")[1])

                    sub_nodes = component_nodes(
                        graph,
                        component_index=comp_idx,
                        max_nodes=int(max_nodes),
                    )
                    plot_data = subgraph_to_plot_data(
                        graph,
                        sub_nodes,
                        max_edges=int(max_edges),
                    )

                    cols = st.columns(3)
                    cols[0].metric("Rendered Nodes", f"{len(plot_data['nodes']):,}")
                    cols[1].metric("Rendered Edges", f"{len(plot_data['edges']):,}")
                    cols[2].metric("Clipped Edges", f"{int(plot_data['clipped_edges']):,}")

                    _render_subgraph_chart(plot_data)

            else:
                _render_global_overview(
                    graph,
                    nodes_df=nodes_df,
                    edges_df=edges_df,
                    default_max_nodes=min(1200, int(max_nodes)),
                    default_max_edges=min(4000, int(max_edges)),
                )

with tab_lookup:
    _render_lookup_tab()
