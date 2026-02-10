from __future__ import annotations

from pathlib import Path

from isb_igraph.runtime import artifacts_root, jobs_db_path
from services.api.store import JobStore
from services.worker.main import run_worker


def test_worker_processes_queued_job(tmp_path: Path, monkeypatch) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("ISB_IGRAPH_RUNTIME_ROOT", str(runtime))
    monkeypatch.setenv("ISB_IGRAPH_UPLOADS_ROOT", str(runtime / "uploads"))
    monkeypatch.setenv("ISB_IGRAPH_ARTIFACTS_ROOT", str(runtime / "artifacts"))
    monkeypatch.setenv("ISB_IGRAPH_JOBS_DB", str(runtime / "jobs.db"))
    monkeypatch.setenv("ISB_IGRAPH_SKIP_RUNTIME_CHECK", "1")

    input_csv = tmp_path / "input.csv"
    input_csv.write_text(
        "job_id,job_title,company_name,posted_at,skills\n"
        "j1,Mechanic,Acme,2025-01-01,\"[{\"\"skill\"\": \"\"Vehicle Maintenance\"\", \"\"bucket\"\": \"\"Proficient\"\", \"\"mapping_similarity\"\": 0.8}]\"\n"
        "j2,Technician,Acme,2025-01-02,\"[{\"\"skill\"\": \"\"Vehicle Maintenance\"\", \"\"bucket\"\": \"\"Working Knowledge\"\", \"\"mapping_similarity\"\": 0.6}]\"\n",
        encoding="utf-8",
    )

    store = JobStore(jobs_db_path())
    job_id = "job-test-1"
    output_dir = artifacts_root() / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    store.create_job(
        job_id=job_id,
        input_path=input_csv,
        output_dir=output_dir,
        options={
            "compute_profile": "quick",
            "similarity_method": "unweighted",
            "top_k": 5,
            "compute_betweenness_enabled": False,
            "compute_closeness_enabled": False,
        },
    )

    run_worker(poll_interval_seconds=0.1, once=True)

    final = store.get_job(job_id)
    assert final.status == "completed"
    assert float(final.progress) == 1.0

    artifacts = store.list_artifacts(job_id)
    names = {a["artifact_name"] for a in artifacts}
    assert "nodes" in names
    assert "edges" in names
    assert "job_similarity_topk" in names
    assert "graph_summary" in names
