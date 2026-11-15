import time

import psutil
from app.schemas.sentiment import SystemHealthResponse
from app.services.tf_inference import tf_engine
from fastapi import APIRouter

router = APIRouter()
START_TIME = time.time()


@router.get("", response_model=SystemHealthResponse)  # type: ignore[misc]
async def get_health_status() -> SystemHealthResponse:
    mem = psutil.virtual_memory()
    uptime = time.time() - START_TIME

    return SystemHealthResponse(
        status="OK",
        cpu_percent=psutil.cpu_percent(interval=None),
        memory_mb=round(mem.used / (1024 * 1024), 2),
        memory_percent=mem.percent,
        uptime_seconds=round(uptime, 2),
        model_loaded=tf_engine.is_warm,
    )


@router.get("/system", response_model=SystemHealthResponse)  # type: ignore[misc]
async def get_system_health() -> SystemHealthResponse:
    status: SystemHealthResponse = await get_health_status()
    return status
