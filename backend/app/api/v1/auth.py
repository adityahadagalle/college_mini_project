# backend/app/api/v1/auth.py
"""Authentication API routes.

All business logic is delegated to :class:`AuthenticationService`. The routes are thin
wrappers that translate FastAPI request/response models to service method calls.
"""

from typing import Any, Dict

from app.auth.exceptions import AuthenticationError, PasswordPolicyError, TokenRevokedError
from app.auth.repositories import AuditRepository, SessionRepository, UserRepository
from app.auth.service import AuthenticationService
from app.core.database import get_db
from app.core.security import decode_access_token
from app.models.database import Role
from app.schemas import auth as auth_schemas
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# Dependency that creates an AuthenticationService instance per request
async def get_auth_service(db: AsyncSession = Depends(get_db)) -> AuthenticationService:
    user_repo = UserRepository(db)
    session_repo = SessionRepository(db)
    audit_repo = AuditRepository(db)
    return AuthenticationService(
        user_repo=user_repo,
        session_repo=session_repo,
        audit_repo=audit_repo,
    )


# Helper to retrieve the current user from the access token
async def get_current_user(
    token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)
) -> auth_schemas.UserProfile:
    user_id = decode_access_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_repo = UserRepository(db)
    user = await user_repo.get(user_id)
    return auth_schemas.UserProfile.from_orm(user)  # type: ignore[no-any-return]


@router.post(  # type: ignore[misc]
    "/register",
    response_model=auth_schemas.UserProfile,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    req: auth_schemas.RegisterRequest,
    service: AuthenticationService = Depends(get_auth_service),
) -> auth_schemas.UserProfile:
    role_enum = Role(req.role) if req.role else Role.USER
    try:
        user = await service.register_user(
            email=req.email,
            password=req.password,
            full_name=req.full_name or "",
            role=role_enum,
        )
        return auth_schemas.UserProfile.from_orm(user)  # type: ignore[no-any-return]
    except (AuthenticationError, PasswordPolicyError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/login", response_model=auth_schemas.TokenResponse)  # type: ignore[misc]
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    service: AuthenticationService = Depends(get_auth_service),
) -> auth_schemas.TokenResponse:
    try:
        result = await service.login(
            email=form.username,
            password=form.password,
            ip_address=None,
            user_agent=None,
        )
        return auth_schemas.TokenResponse(**result)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/refresh", response_model=auth_schemas.TokenResponse)  # type: ignore[misc]
async def refresh(
    req: auth_schemas.RefreshTokenRequest,
    service: AuthenticationService = Depends(get_auth_service),
) -> auth_schemas.TokenResponse:
    try:
        result = await service.refresh_access_token(
            refresh_token=req.refresh_token,
            ip_address=None,
            user_agent=None,
        )
        return auth_schemas.TokenResponse(**result)
    except TokenRevokedError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))


@router.post("/logout")  # type: ignore[misc]
async def logout(
    req: auth_schemas.LogoutRequest,
    service: AuthenticationService = Depends(get_auth_service),
) -> Dict[str, Any]:
    try:
        await service.logout(refresh_token=req.refresh_token)
        return {"detail": "Successfully logged out"}
    except TokenRevokedError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))


@router.post("/logout-all")  # type: ignore[misc]
async def logout_all(
    current_user: auth_schemas.UserProfile = Depends(get_current_user),
    service: AuthenticationService = Depends(get_auth_service),
) -> Dict[str, Any]:
    await service.logout_all_devices(user_id=current_user.id)
    return {"detail": "All sessions revoked"}


@router.post("/change-password")  # type: ignore[misc]
async def change_password(
    req: auth_schemas.PasswordChangeRequest,
    current_user: auth_schemas.UserProfile = Depends(get_current_user),
    service: AuthenticationService = Depends(get_auth_service),
) -> Dict[str, Any]:
    try:
        await service.change_password(
            user_id=current_user.id,
            current_password=req.current_password,
            new_password=req.new_password,
        )
        return {"detail": "Password changed"}
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))
    except PasswordPolicyError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/me", response_model=auth_schemas.UserProfile)  # type: ignore[misc]
async def me(
    current_user: auth_schemas.UserProfile = Depends(get_current_user),
) -> auth_schemas.UserProfile:
    return current_user
