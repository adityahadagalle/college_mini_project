import os
import uuid

from app.api.v1.auth import get_current_user
from app.core.database import get_db
from app.models.database import BatchJob, User
from app.schemas.sentiment import BatchJobResponse
from app.services.batch_service import process_batch_job_async
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()

UPLOAD_DIR = "./batch_uploads"
RESULTS_DIR = "./batch_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


@router.post("/upload", response_model=BatchJobResponse)  # type: ignore[misc]
async def upload_batch_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BatchJobResponse:
    filename = file.filename or ""
    if not (filename.endswith(".csv") or filename.endswith(".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .csv and .txt files are supported.",
        )

    job_id = str(uuid.uuid4())
    input_file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{filename}")

    content = await file.read()
    with open(input_file_path, "wb") as f:
        f.write(content)

    batch_job = BatchJob(
        id=job_id,
        user_id=current_user.id,
        filename=filename,
        total_rows=0,
        processed_rows=0,
        status="PENDING",
    )
    db.add(batch_job)
    await db.commit()
    await db.refresh(batch_job)

    background_tasks.add_task(process_batch_job_async, job_id, input_file_path, current_user.id)

    return BatchJobResponse(
        job_id=batch_job.id,
        filename=batch_job.filename,
        total_rows=batch_job.total_rows,
        processed_rows=batch_job.processed_rows,
        status=batch_job.status,
        result_file_path=batch_job.result_file_path,
        created_at=str(batch_job.created_at) if batch_job.created_at else None,
        completed_at=str(batch_job.completed_at) if batch_job.completed_at else None,
    )


@router.get("/status/{job_id}", response_model=BatchJobResponse)  # type: ignore[misc]
async def get_batch_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BatchJobResponse:
    batch_job = await db.get(BatchJob, job_id)
    if not batch_job or batch_job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch job not found.",
        )

    return BatchJobResponse(
        job_id=batch_job.id,
        filename=batch_job.filename,
        total_rows=batch_job.total_rows,
        processed_rows=batch_job.processed_rows,
        status=batch_job.status,
        result_file_path=batch_job.result_file_path,
        created_at=str(batch_job.created_at) if batch_job.created_at else None,
        completed_at=str(batch_job.completed_at) if batch_job.completed_at else None,
    )


@router.get("/download/{job_id}")  # type: ignore[misc]
async def download_batch_results(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    batch_job = await db.get(BatchJob, job_id)
    if not batch_job or batch_job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch job not found.",
        )

    if batch_job.status != "COMPLETED" or not batch_job.result_file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch job is not completed yet.",
        )

    if not os.path.exists(batch_job.result_file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result file not found on disk.",
        )

    return FileResponse(
        path=batch_job.result_file_path,
        filename=f"sentiment_results_{job_id}.csv",
        media_type="text/csv",
    )
