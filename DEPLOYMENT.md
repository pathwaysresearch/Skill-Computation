# Deployment Guide (Async, Large Files, Large Compute)

## Architecture (implemented)

This repo now supports an async deployment model:

- `Streamlit` UI for submit/monitor/download/lookup/visualize
- `FastAPI` jobs service (`services/api/main.py`)
- `Worker` process (`services/worker/main.py`) polling queued jobs
- `SQLite` metadata queue (`runtime/jobs.db`)
- Shared filesystem artifacts under `runtime/`

## Why this solves the 200 MB UI limit

Streamlit upload constraints are no longer the compute boundary.

You can:
- submit by **server path** (`POST /v1/jobs/from-path`) for very large files, or
- use **chunked upload API** (`/v1/uploads/*`) and then queue a job.

Heavy igraph compute runs in worker, not in Streamlit request lifecycle.

## API endpoints

- `POST /v1/uploads/init`
- `PUT /v1/uploads/{upload_id}/part/{part_number}`
- `POST /v1/uploads/{upload_id}/complete`
- `POST /v1/jobs`
- `POST /v1/jobs/from-path`
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}/events`
- `GET /v1/jobs/{job_id}/artifacts`
- `POST /v1/jobs/{job_id}/cancel`

## Local run

Terminal 1 (API):

```bash
uvicorn services.api.main:app --host 0.0.0.0 --port 8000
```

Terminal 2 (Worker):

```bash
python -m services.worker.main --poll-interval 2
```

Terminal 3 (UI):

```bash
export ISB_IGRAPH_API_BASE_URL=http://localhost:8000
streamlit run app.py
```

## Docker Compose run

```bash
docker compose up --build
```

Services:
- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

## Operational notes

- Use `quick` profile for very large first-pass validation.
- Use subset mode for reproducible stress checks before full run.
- Keep centrality guards enabled to avoid runaway compute on huge projected graphs.
- Worker supports CSV and XLSX (XLSX is converted to CSV in worker).
