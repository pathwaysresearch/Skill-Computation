from __future__ import annotations

import hashlib
import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from isb_igraph.environment import assert_runtime_compatibility
from isb_igraph.runtime import artifacts_root, ensure_runtime_dirs, jobs_db_path, uploads_root

from .store import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JobStore,
)


app = FastAPI(title="ISB iGraph Jobs API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


store: JobStore | None = None


class UploadInitRequest(BaseModel):
    file_name: str = Field(min_length=1)
    total_parts: int = Field(ge=1)
    checksum_sha256: str | None = None


class UploadInitResponse(BaseModel):
    upload_id: str
    file_name: str
    total_parts: int
    status: str


class UploadCompleteRequest(BaseModel):
    file_name: str | None = None


class CreateJobRequest(BaseModel):
    upload_id: str
    options: dict[str, Any] = Field(default_factory=dict)


class CreateJobFromPathRequest(BaseModel):
    input_path: str
    options: dict[str, Any] = Field(default_factory=dict)


class CreateJobFromGCSRequest(BaseModel):
    gcs_uri: str
    options: dict[str, Any] = Field(default_factory=dict)


class CancelJobResponse(BaseModel):
    job_id: str
    status: str


def _get_store() -> JobStore:
    global store
    if store is None:
        ensure_runtime_dirs()
        store = JobStore(jobs_db_path())
    return store


@app.on_event("startup")
def startup_event() -> None:
    assert_runtime_compatibility()
    _get_store()


def _upload_parts_dir(upload_id: str) -> Path:
    return uploads_root() / upload_id / "parts"


def _upload_assembled_dir(upload_id: str) -> Path:
    return uploads_root() / upload_id / "assembled"


def _json_loads_safe(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _validate_supported_input_name(name: str) -> None:
    suffix = Path(name).suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise HTTPException(status_code=400, detail="Only CSV/XLSX files are supported")


def _validate_gcs_uri(gcs_uri: str) -> str:
    uri = gcs_uri.strip()
    if not uri.startswith("gs://"):
        raise HTTPException(status_code=400, detail="gcs_uri must start with gs://")
    if len(uri) <= len("gs://"):
        raise HTTPException(status_code=400, detail="gcs_uri is empty")
    remainder = uri[len("gs://") :]
    bucket, sep, blob = remainder.partition("/")
    if not bucket or not sep or not blob:
        raise HTTPException(status_code=400, detail="gcs_uri must be in format gs://bucket/path/file.csv")
    _validate_supported_input_name(blob)
    return uri


def _job_payload(job_id: str) -> dict[str, Any]:
    record = _get_store().get_job(job_id)
    artifacts = _get_store().list_artifacts(job_id)
    return {
        "job_id": record.job_id,
        "upload_id": record.upload_id,
        "input_path": record.input_path,
        "output_dir": record.output_dir,
        "status": record.status,
        "stage": record.stage,
        "progress": float(record.progress),
        "error_message": record.error_message,
        "options": _json_loads_safe(record.options_json) or {},
        "run_metadata": _json_loads_safe(record.run_metadata_json),
        "created_at": record.created_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "artifacts": artifacts,
    }


@app.get("/health")
def health() -> dict[str, str]:
    _get_store()
    return {"status": "ok"}


@app.post("/v1/uploads/init", response_model=UploadInitResponse)
def init_upload(request: UploadInitRequest) -> UploadInitResponse:
    upload = _get_store().create_upload(
        file_name=request.file_name,
        total_parts=request.total_parts,
        checksum_sha256=request.checksum_sha256,
    )

    _upload_parts_dir(upload.upload_id).mkdir(parents=True, exist_ok=True)
    _upload_assembled_dir(upload.upload_id).mkdir(parents=True, exist_ok=True)

    return UploadInitResponse(
        upload_id=upload.upload_id,
        file_name=upload.file_name,
        total_parts=upload.total_parts,
        status=upload.status,
    )


@app.put("/v1/uploads/{upload_id}/part/{part_number}")
async def upload_part(upload_id: str, part_number: int, request: Request) -> dict[str, Any]:
    try:
        upload = _get_store().get_upload(upload_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if part_number < 1 or part_number > int(upload.total_parts):
        raise HTTPException(status_code=400, detail="part_number out of range")

    content = await request.body()
    if not content:
        raise HTTPException(status_code=400, detail="empty upload part")

    part_dir = _upload_parts_dir(upload_id)
    part_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_dir / f"{part_number:08d}.part"

    previous_size = part_path.stat().st_size if part_path.exists() else 0
    tmp_path = part_dir / f"{part_number:08d}.part.tmp"
    tmp_path.write_bytes(content)
    tmp_path.replace(part_path)

    delta = int(len(content) - previous_size)
    if delta != 0:
        _get_store().add_uploaded_bytes(upload_id, delta)

    return {
        "upload_id": upload_id,
        "part_number": part_number,
        "bytes": len(content),
        "status": "ok",
    }


@app.post("/v1/uploads/{upload_id}/complete")
def complete_upload(upload_id: str, payload: UploadCompleteRequest | None = None) -> dict[str, Any]:
    try:
        upload = _get_store().get_upload(upload_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    parts_dir = _upload_parts_dir(upload_id)
    if not parts_dir.exists():
        raise HTTPException(status_code=400, detail="No uploaded parts found")

    expected_parts = [parts_dir / f"{i:08d}.part" for i in range(1, int(upload.total_parts) + 1)]
    missing = [str(path.name) for path in expected_parts if not path.exists()]
    if missing:
        raise HTTPException(status_code=400, detail={"missing_parts": missing[:50], "count": len(missing)})

    output_name = (payload.file_name if payload and payload.file_name else upload.file_name).strip()
    if not output_name:
        output_name = upload.file_name

    assembled_dir = _upload_assembled_dir(upload_id)
    assembled_dir.mkdir(parents=True, exist_ok=True)
    assembled_path = assembled_dir / output_name

    hasher = hashlib.sha256()
    total_bytes = 0

    with assembled_path.open("wb") as out:
        for part_path in expected_parts:
            with part_path.open("rb") as part_file:
                while True:
                    chunk = part_file.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    hasher.update(chunk)
                    total_bytes += len(chunk)

    checksum = hasher.hexdigest()
    if upload.checksum_sha256 and checksum.lower() != upload.checksum_sha256.lower():
        assembled_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail={
                "error": "checksum_mismatch",
                "expected": upload.checksum_sha256,
                "actual": checksum,
            },
        )

    _get_store().mark_upload_completed(upload_id, assembled_path)

    return {
        "upload_id": upload_id,
        "assembled_path": str(assembled_path),
        "bytes": total_bytes,
        "checksum_sha256": checksum,
        "status": "completed",
    }


@app.post("/v1/jobs")
def create_job(payload: CreateJobRequest) -> dict[str, Any]:
    try:
        upload = _get_store().get_upload(payload.upload_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not upload.assembled_path:
        raise HTTPException(status_code=400, detail="Upload not completed")

    input_path = Path(upload.assembled_path)
    if not input_path.exists():
        raise HTTPException(status_code=400, detail="Uploaded file path not found on server")

    job_id = str(uuid.uuid4())
    output_dir = artifacts_root() / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _get_store().create_job(
        job_id=job_id,
        upload_id=upload.upload_id,
        input_path=input_path,
        output_dir=output_dir,
        options=payload.options,
    )

    return _job_payload(job_id)


@app.post("/v1/jobs/from-path")
def create_job_from_path(payload: CreateJobFromPathRequest) -> dict[str, Any]:
    input_path = Path(payload.input_path)
    if not input_path.exists() or not input_path.is_file():
        raise HTTPException(status_code=400, detail=f"input_path not found: {input_path}")

    _validate_supported_input_name(input_path.name)

    job_id = str(uuid.uuid4())
    output_dir = artifacts_root() / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _get_store().create_job(
        job_id=job_id,
        input_path=input_path,
        output_dir=output_dir,
        options=payload.options,
    )

    return _job_payload(job_id)


@app.post("/v1/jobs/from-gcs")
def create_job_from_gcs(payload: CreateJobFromGCSRequest) -> dict[str, Any]:
    gcs_uri = _validate_gcs_uri(payload.gcs_uri)

    job_id = str(uuid.uuid4())
    output_dir = artifacts_root() / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _get_store().create_job(
        job_id=job_id,
        input_path=gcs_uri,
        output_dir=output_dir,
        options=payload.options,
    )
    _get_store().add_event(job_id, level="info", message=f"Registered GCS input: {gcs_uri}", progress=0.0)

    return _job_payload(job_id)


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    try:
        return _job_payload(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/jobs/{job_id}/events")
def get_job_events(job_id: str, limit: int = Query(default=200, ge=1, le=2000)) -> dict[str, Any]:
    try:
        _ = _get_store().get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"job_id": job_id, "events": _get_store().list_events(job_id, limit=limit)}


@app.get("/v1/jobs/{job_id}/artifacts")
def get_job_artifacts(job_id: str) -> dict[str, Any]:
    try:
        _ = _get_store().get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"job_id": job_id, "artifacts": _get_store().list_artifacts(job_id)}


@app.get("/v1/jobs/{job_id}/artifacts/{artifact_name}")
def download_artifact(job_id: str, artifact_name: str) -> FileResponse:
    artifacts = _get_store().list_artifacts(job_id)
    match = next((a for a in artifacts if a["artifact_name"] == artifact_name), None)
    if match is None:
        raise HTTPException(status_code=404, detail="artifact not found")

    file_path = Path(str(match["file_path"]))
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="artifact file missing")

    mime_type = match.get("mime_type") or mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    return FileResponse(path=file_path, media_type=mime_type, filename=file_path.name)


@app.post("/v1/jobs/{job_id}/cancel", response_model=CancelJobResponse)
def cancel_job(job_id: str) -> CancelJobResponse:
    try:
        status = _get_store().get_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if status in {JOB_STATUS_COMPLETED, JOB_STATUS_CANCELLED}:
        return CancelJobResponse(job_id=job_id, status=status)

    _get_store().cancel_job(job_id)
    _get_store().add_event(job_id, level="warning", message="Job cancellation requested", progress=None)

    return CancelJobResponse(job_id=job_id, status=JOB_STATUS_CANCELLED)
