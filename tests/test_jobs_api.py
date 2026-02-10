from __future__ import annotations

import importlib
from pathlib import Path

from fastapi.testclient import TestClient


SAMPLE_CSV = "job_id,job_title,skills\n1,Mechanic,[]\n"


def _build_client(tmp_path: Path, monkeypatch) -> tuple[TestClient, object]:
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("ISB_IGRAPH_RUNTIME_ROOT", str(runtime))
    monkeypatch.setenv("ISB_IGRAPH_UPLOADS_ROOT", str(runtime / "uploads"))
    monkeypatch.setenv("ISB_IGRAPH_ARTIFACTS_ROOT", str(runtime / "artifacts"))
    monkeypatch.setenv("ISB_IGRAPH_JOBS_DB", str(runtime / "jobs.db"))
    monkeypatch.setenv("ISB_IGRAPH_SKIP_RUNTIME_CHECK", "1")

    import services.api.main as api_main

    api_main = importlib.reload(api_main)
    client = TestClient(api_main.app)
    return client, api_main


def test_upload_and_create_job_flow(tmp_path: Path, monkeypatch) -> None:
    client, api_main = _build_client(tmp_path, monkeypatch)

    init = client.post(
        "/v1/uploads/init",
        json={"file_name": "input.csv", "total_parts": 2},
    )
    assert init.status_code == 200
    upload_id = init.json()["upload_id"]

    data = SAMPLE_CSV.encode("utf-8")
    split = len(data) // 2
    p1 = client.put(f"/v1/uploads/{upload_id}/part/1", content=data[:split])
    p2 = client.put(f"/v1/uploads/{upload_id}/part/2", content=data[split:])
    assert p1.status_code == 200
    assert p2.status_code == 200

    complete = client.post(f"/v1/uploads/{upload_id}/complete", json={})
    assert complete.status_code == 200
    assembled_path = Path(complete.json()["assembled_path"])
    assert assembled_path.exists()

    create_job = client.post(
        "/v1/jobs",
        json={"upload_id": upload_id, "options": {"compute_profile": "quick"}},
    )
    assert create_job.status_code == 200
    job_payload = create_job.json()
    assert job_payload["status"] == "queued"
    assert job_payload["job_id"]

    job_id = job_payload["job_id"]
    job_status = client.get(f"/v1/jobs/{job_id}")
    assert job_status.status_code == 200
    assert job_status.json()["status"] == "queued"

    events = client.get(f"/v1/jobs/{job_id}/events")
    assert events.status_code == 200
    assert len(events.json()["events"]) >= 1

    cancel = client.post(f"/v1/jobs/{job_id}/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["status"] == "cancelled"

    # Ensure singleton store is set (sanity check reload worked).
    assert api_main.store is not None


def test_create_job_from_path_not_found(tmp_path: Path, monkeypatch) -> None:
    client, _ = _build_client(tmp_path, monkeypatch)

    resp = client.post(
        "/v1/jobs/from-path",
        json={"input_path": str(tmp_path / "missing.csv"), "options": {}},
    )
    assert resp.status_code == 400


def test_upload_checksum_mismatch(tmp_path: Path, monkeypatch) -> None:
    client, _ = _build_client(tmp_path, monkeypatch)

    init = client.post(
        "/v1/uploads/init",
        json={"file_name": "input.csv", "total_parts": 1, "checksum_sha256": "deadbeef"},
    )
    assert init.status_code == 200
    upload_id = init.json()["upload_id"]

    p1 = client.put(f"/v1/uploads/{upload_id}/part/1", content=SAMPLE_CSV.encode("utf-8"))
    assert p1.status_code == 200

    complete = client.post(f"/v1/uploads/{upload_id}/complete", json={})
    assert complete.status_code == 400
    detail = complete.json()["detail"]
    assert detail["error"] == "checksum_mismatch"
