# ISB iGraph MVP + Async Deployment

End-to-end pipeline to convert large jobs CSV/XLSX into igraph structures, compute job closeness, and serve deployment-ready async execution.

## What is implemented

- Chunked ingestion and robust skill parsing
- Job-Skill bipartite graph using `python-igraph`
- Job-Job projection with deterministic similarity
- Lookup utilities:
  - Job -> Closest Jobs
  - Skill(s) -> Closest Jobs (OR)
  - Job -> Top Skills
- Streamlit UI tabs:
  - `Run` (async API jobs)
  - `Metrics`
  - `Lookup`
  - `Visualize`
- Async stack for large compute:
  - FastAPI jobs API (`services/api/main.py`)
  - SQLite job store/queue (`services/api/store.py`)
  - Worker (`services/worker/main.py`)

## Similarity is deterministic (not random)

- Unweighted similarity: `shared_skill_count`
- Weighted similarity: `sum(min(edge_weight_a, edge_weight_b))` over shared skills

Default ranking in lookup is unweighted shared-skill count first.

## Project structure

```text
app.py
requirements.txt
docker-compose.yml
Dockerfile
DEPLOYMENT.md
services/
  api/
    main.py
    store.py
  worker/
    main.py
isb_igraph/
  pipeline.py
  graph.py
  projection.py
  lookup.py
  skill_lookup.py
  runtime.py
  jobs_client.py
tests/
  test_jobs_api.py
  test_worker_async.py
  ...
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (local, async)

Terminal 1:

```bash
uvicorn services.api.main:app --host 0.0.0.0 --port 8000
```

Terminal 2:

```bash
python -m services.worker.main --poll-interval 2
```

Terminal 3:

```bash
export ISB_IGRAPH_API_BASE_URL=http://localhost:8000
streamlit run app.py
```

## Run with Docker Compose

```bash
docker compose up --build
```

## CLI pipeline (direct)

```bash
python -m isb_igraph \
  --input sample_data/jobs_sample.csv \
  --output-dir output
```

## Lookup CLIs

Closest jobs:

```bash
python -m isb_igraph.lookup \
  --nodes-csv output/nodes.csv \
  --similarity-csv output/job_similarity_topk.csv \
  --job-id job_001 \
  --method unweighted \
  --limit 20
```

Skill(s) -> jobs OR:

```bash
python -m isb_igraph.skill_lookup \
  --nodes-csv output/nodes.csv \
  --edges-csv output/edges.csv \
  --skills "problem solving, analytical thinking" \
  --limit 20
```

Job -> top skills:

```bash
python -m isb_igraph.skill_lookup \
  --nodes-csv output/nodes.csv \
  --edges-csv output/edges.csv \
  --job-id job_001 \
  --limit 20
```

## API quick examples

Queue by server path (best for very large files):

```bash
curl -X POST http://localhost:8000/v1/jobs/from-path \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "/Users/srijan26/ISB_Work/isb-igraph /Merged_All_Sheets_post6_skill_salary_merged_skills_non_empty_2_from_xlsx.csv",
    "options": {"compute_profile": "quick", "similarity_method": "both", "top_k": 20}
  }'
```

Get status:

```bash
curl http://localhost:8000/v1/jobs/<job_id>
```

## Outputs

Mandatory outputs:
- `nodes.csv`
- `edges.csv`
- `job_similarity_topk.csv`
- `graph_summary.json`
- `parse_errors.csv`

Additional:
- `job_node_metrics.csv`
- `skill_node_metrics.csv`
- `projected_job_metrics.csv`
- `validation_report.json`
- `qa_report.json`
- `profiling_summary.csv`

## Scaling notes beyond MVP

- Keep Streamlit as orchestration/inspection UI only.
- Scale workers horizontally (separate queue backend later if needed).
- Move artifacts to object storage and replace local paths with signed URLs.
- Keep deterministic subset + full validation gates for each new data drop.

## Run tests

```bash
python -m pytest -q
```

## Projection guard knobs

For very large datasets, use:

- `--projection-max-skill-degree`
- `--projection-max-pairs-per-skill`
- `--projection-max-total-pairs`

These prevent pair explosion in Job-Job projection while keeping deterministic behavior.
