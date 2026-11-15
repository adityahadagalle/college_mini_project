from typing import Any, Dict

from app.api.v1.auth import get_current_user
from app.core.database import get_db
from app.models.database import User, UserSettings
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.get("")  # type: ignore[misc]
async def get_settings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    result = await db.execute(select(UserSettings).where(UserSettings.user_id == current_user.id))
    s = result.scalars().first()
    if not s:
        s = UserSettings(user_id=current_user.id)
        db.add(s)
        await db.commit()
        await db.refresh(s)

    return {
        "confidence_threshold": s.confidence_threshold,
        "default_export_format": s.default_export_format,
        "theme": s.theme,
    }


@router.put("")  # type: ignore[misc]
async def update_settings(
    payload: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    result = await db.execute(select(UserSettings).where(UserSettings.user_id == current_user.id))
    s = result.scalars().first()
    if not s:
        s = UserSettings(user_id=current_user.id)
        db.add(s)

    if "confidence_threshold" in payload:
        s.confidence_threshold = float(payload["confidence_threshold"])
    if "default_export_format" in payload:
        s.default_export_format = str(payload["default_export_format"])
    if "theme" in payload:
        s.theme = str(payload["theme"])

    await db.commit()
    await db.refresh(s)
    return {
        "confidence_threshold": s.confidence_threshold,
        "default_export_format": s.default_export_format,
        "theme": s.theme,
    }
