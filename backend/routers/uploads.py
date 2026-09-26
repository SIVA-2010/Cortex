from __future__ import annotations

import hashlib
import re
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select

from backend.config import get_settings
from backend.dependencies import CurrentUserDependency, DatabaseDependency
from backend.models import Mission, UploadedDocument
from backend.schemas import UploadResponse
from backend.services.complaint_processor import ComplaintProcessor
from backend.services.vector_store import get_vector_store_service
from backend.services.workflow_profile import detect_workflow_profile


router = APIRouter(prefix="/api/v1/uploads", tags=["Uploads and Knowledge"])
settings = get_settings()


def _safe_name(name: str) -> str:
    base = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return cleaned or "upload"


def _validate_mission_access(
    database: DatabaseDependency,
    current_user: CurrentUserDependency,
    mission_id: UUID | None,
) -> Mission | None:
    if mission_id is None:
        return None
    mission = database.get(Mission, mission_id)
    if mission is None:
        raise HTTPException(status_code=404, detail="Mission not found.")
    if current_user.role not in {"administrator", "reviewer"}:
        if mission.created_by_id != current_user.id:
            raise HTTPException(status_code=403, detail="Mission access denied.")
    return mission


async def _save_upload(
    upload: UploadFile,
    *,
    destination: Path,
    allowed_extensions: set[str],
) -> tuple[Path, int, str]:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{uuid4().hex}_{_safe_name(upload.filename or 'upload')}"
    digest = hashlib.sha256()
    total = 0

    try:
        with path.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {settings.max_upload_size_mb} MB.",
                    )
                digest.update(chunk)
                handle.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    if total == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    return path, total, digest.hexdigest()


@router.post(
    "/complaints",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_complaints(
    database: DatabaseDependency,
    current_user: CurrentUserDependency,
    file: UploadFile = File(...),
    mission_id: UUID | None = Form(default=None),
) -> UploadResponse:
    mission = _validate_mission_access(database, current_user, mission_id)
    if mission is not None and detect_workflow_profile(mission) != "complaint":
        raise HTTPException(
            status_code=422,
            detail=(
                "Dataset Mismatch — this mission is Logistics Intelligence. "
                "A complaint CSV cannot be linked to the logistics workflow."
            ),
        )

    folder = settings.upload_path / (str(mission_id) if mission_id else "unassigned")
    path, size, sha256 = await _save_upload(
        file,
        destination=folder,
        allowed_extensions={".csv"},
    )
    try:
        validation = ComplaintProcessor.validate_dataset_suitability(
            path,
            objective=(f"{mission.title} {mission.objective}" if mission is not None else ""),
        )
    except ValueError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    record = UploadedDocument(
        mission_id=mission_id,
        uploaded_by_id=current_user.id,
        file_name=Path(file.filename or path.name).name,
        stored_path=str(path),
        content_type=file.content_type or "text/csv",
        document_type="complaint_csv",
        size_bytes=size,
        sha256=sha256,
        status="ready",
        metadata_json={"synthetic_data_expected": True, "dataset_validation": validation},
    )
    database.add(record)
    database.commit()
    database.refresh(record)
    return UploadResponse(
        document_id=record.id,
        mission_id=record.mission_id,
        file_name=record.file_name,
        document_type=record.document_type,
        size_bytes=record.size_bytes,
        status=record.status,
        message="Complaint CSV passed schema and semantic validation and is ready for execution.",
    )


@router.post(
    "/knowledge",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_knowledge(
    database: DatabaseDependency,
    current_user: CurrentUserDependency,
    file: UploadFile = File(...),
    mission_id: UUID | None = Form(default=None),
) -> UploadResponse:
    _validate_mission_access(database, current_user, mission_id)
    folder = settings.upload_path / "knowledge"
    path, size, sha256 = await _save_upload(
        file,
        destination=folder,
        allowed_extensions={".md", ".txt"},
    )
    try:
        ingestion = get_vector_store_service().ingest_file(
            path, document_type="uploaded_enterprise_knowledge"
        )
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Knowledge ingestion failed: {exc}") from exc

    record = UploadedDocument(
        mission_id=mission_id,
        uploaded_by_id=current_user.id,
        file_name=Path(file.filename or path.name).name,
        stored_path=str(path),
        content_type=file.content_type or "text/plain",
        document_type="knowledge",
        size_bytes=size,
        sha256=sha256,
        status="indexed",
        metadata_json=ingestion.model_dump(mode="json"),
    )
    database.add(record)
    database.commit()
    database.refresh(record)
    return UploadResponse(
        document_id=record.id,
        mission_id=record.mission_id,
        file_name=record.file_name,
        document_type=record.document_type,
        size_bytes=record.size_bytes,
        status=record.status,
        message=f"Knowledge indexed into {settings.chroma_collection}.",
    )


@router.get("", response_model=list[UploadResponse])
def list_uploads(
    database: DatabaseDependency,
    current_user: CurrentUserDependency,
    mission_id: UUID | None = None,
) -> list[UploadResponse]:
    query = select(UploadedDocument).order_by(UploadedDocument.created_at.desc())
    if mission_id is not None:
        _validate_mission_access(database, current_user, mission_id)
        query = query.where(UploadedDocument.mission_id == mission_id)
    elif current_user.role not in {"administrator", "reviewer"}:
        query = query.where(UploadedDocument.uploaded_by_id == current_user.id)
    records = list(database.scalars(query.limit(100)))
    return [
        UploadResponse(
            document_id=record.id,
            mission_id=record.mission_id,
            file_name=record.file_name,
            document_type=record.document_type,
            size_bytes=record.size_bytes,
            status=record.status,
            message="",
        )
        for record in records
    ]
