from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any

import requests


def default_api_base_url() -> str:
    return os.getenv("ISB_IGRAPH_API_BASE_URL", "http://localhost:8000").rstrip("/")


def _request_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.request(method=method, url=url, timeout=kwargs.pop("timeout", 120), **kwargs)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object response")
    return payload


def _sha256_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def submit_job_from_path(
    *,
    input_path: Path,
    options: dict[str, Any],
    base_url: str | None = None,
) -> dict[str, Any]:
    base = (base_url or default_api_base_url()).rstrip("/")
    return _request_json(
        "POST",
        f"{base}/v1/jobs/from-path",
        json={"input_path": str(input_path), "options": options},
    )


def upload_and_submit_job(
    *,
    file_path: Path,
    options: dict[str, Any],
    base_url: str | None = None,
    part_size_mb: int = 8,
) -> dict[str, Any]:
    base = (base_url or default_api_base_url()).rstrip("/")
    file_size = int(file_path.stat().st_size)
    part_size = max(1, int(part_size_mb)) * 1024 * 1024
    total_parts = max(1, int(math.ceil(file_size / float(part_size))))

    checksum = _sha256_file(file_path)

    init_payload = _request_json(
        "POST",
        f"{base}/v1/uploads/init",
        json={
            "file_name": file_path.name,
            "total_parts": total_parts,
            "checksum_sha256": checksum,
        },
    )
    upload_id = str(init_payload["upload_id"])

    with file_path.open("rb") as f:
        for part_number in range(1, total_parts + 1):
            part_bytes = f.read(part_size)
            if not part_bytes:
                break
            response = requests.put(
                f"{base}/v1/uploads/{upload_id}/part/{part_number}",
                data=part_bytes,
                headers={"Content-Type": "application/octet-stream"},
                timeout=240,
            )
            response.raise_for_status()

    _request_json("POST", f"{base}/v1/uploads/{upload_id}/complete", json={})

    return _request_json(
        "POST",
        f"{base}/v1/jobs",
        json={"upload_id": upload_id, "options": options},
    )


def get_job(job_id: str, *, base_url: str | None = None) -> dict[str, Any]:
    base = (base_url or default_api_base_url()).rstrip("/")
    return _request_json("GET", f"{base}/v1/jobs/{job_id}")


def get_job_events(job_id: str, *, base_url: str | None = None, limit: int = 200) -> dict[str, Any]:
    base = (base_url or default_api_base_url()).rstrip("/")
    return _request_json("GET", f"{base}/v1/jobs/{job_id}/events", params={"limit": int(limit)})


def get_job_artifacts(job_id: str, *, base_url: str | None = None) -> dict[str, Any]:
    base = (base_url or default_api_base_url()).rstrip("/")
    return _request_json("GET", f"{base}/v1/jobs/{job_id}/artifacts")


def cancel_job(job_id: str, *, base_url: str | None = None) -> dict[str, Any]:
    base = (base_url or default_api_base_url()).rstrip("/")
    return _request_json("POST", f"{base}/v1/jobs/{job_id}/cancel")
