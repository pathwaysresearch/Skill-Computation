# Deployment Guide (Large Files + Large Computation)

## Why 200 MB appears in Streamlit

Streamlit defaults to `200 MB` upload and message size limits.

In this project, we override that in:
- `.streamlit/config.toml`

```toml
[server]
maxUploadSize = 2048
maxMessageSize = 2048
```

This is enough for many cases on self-hosted infrastructure, but for truly large files and heavy computation, UI-only architecture is not enough.

## Recommended production architecture

Use Streamlit as the UI layer only. Run heavy graph computation in background workers.

### 1. UI (Streamlit)
- Upload metadata + create run request
- Show job status/progress
- Download outputs when done

### 2. Object storage (S3/GCS/Azure Blob)
- Store input files (`.csv`/`.xlsx`)
- Store result artifacts (`nodes.csv`, `edges.csv`, `job_similarity_topk.csv`, reports)

### 3. API service (FastAPI)
- `POST /jobs` -> create run
- `GET /jobs/{id}` -> status/progress
- `GET /jobs/{id}/artifacts` -> signed download URLs

### 4. Queue + Workers
- Redis + Celery/RQ (or managed batch jobs)
- Worker runs pipeline CLI with chosen compute profile/limits
- Worker writes progress/status and final artifact paths

### 5. Metadata store
- Postgres (or Redis for MVP) for job state

## Minimal cloud deployment options

### Option A: Smallest deployable (good MVP online)
- 1 VM for Streamlit + API
- 1 VM for worker
- Redis + object storage

### Option B: Better scaling
- Streamlit in container service
- API in container service
- Worker autoscaling group
- Managed Redis + object storage + Postgres

## Operational settings for large runs

- Keep `subset mode` available for fast validation.
- Use computation profile by dataset size:
  - `fast`: disables betweenness, strict guards
  - `balanced`: current default
  - `deep`: larger limits
- Keep centrality guards enabled to avoid runaway compute.
- Add per-job timeout and max-concurrency.

## Testing matrix (recommended)

Run these before going live:

1. Data size tiers
- 5k rows
- 20k rows
- 60k rows
- full dataset

2. Compute profiles
- fast
- balanced
- deep

3. Visualization modes
- Ego graph
- Top component
- Global overview (sampled)
- Diagnostics button in UI

## Practical notes

- "Visualize all" for very large graphs is handled via **global sampled rendering** plus distribution charts.
- Full-force rendering of every node/edge in browser is not reliable at large scale.
- For very large Excel files, prefer preprocessing to CSV/Parquet in worker.

## Suggested next implementation step

Create a tiny `jobs_api.py` (FastAPI) + `worker.py` (queue consumer) in this repo and wire Streamlit to submit jobs instead of running heavy compute in-process.
