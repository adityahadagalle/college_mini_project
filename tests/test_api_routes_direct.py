"""Direct async unit tests for API routes to achieve full line & branch coverage."""

import io
from datetime import datetime, timezone

import pytest
from app.api.v1 import analytics, auth, batch, history
from app.api.v1 import settings as user_settings
from app.core.database import Base
from app.models.auth import RefreshToken, Session
from app.models.database import BatchJob, Prediction, User, UserSettings
from app.schemas import auth as auth_schemas
from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionLocal = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)

_TEST_USER_ID = "direct-route-user-111"


@pytest.fixture(autouse=True)
async def setup_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with TestSessionLocal() as db:
        user = User(
            id=_TEST_USER_ID,
            email="direct@test.com",
            hashed_password="hash",
            full_name="Direct User",
            is_active=True,
        )
        db.add(user)
        await db.commit()
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# ---------------------------------------------------------------------------
# Direct Auth Router Tests
# ---------------------------------------------------------------------------


class TestAuthRouterDirect:
    async def test_auth_router_methods(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            user = await db.get(User, _TEST_USER_ID)
            assert user is not None

            auth_svc = await auth.get_auth_service(db)

            # register
            reg_req = auth_schemas.RegisterRequest(
                email="new_direct@test.com",
                password="SecurePassword123!",
                full_name="New Direct User",
            )
            reg_res = await auth.register(reg_req, auth_svc)
            assert reg_res.email == "new_direct@test.com"

            # register duplicate raises 400
            with pytest.raises(HTTPException) as exc1:
                await auth.register(reg_req, auth_svc)
            assert exc1.value.status_code == 400

            # me
            user_prof = auth_schemas.UserProfile(
                id=user.id,
                email=user.email,
                full_name=user.full_name or "",
                role="user",
                created_at=datetime.now(timezone.utc),
            )
            me_res = await auth.me(user_prof)
            assert me_res.id == _TEST_USER_ID

            # logout_all
            log_all_res = await auth.logout_all(user_prof, auth_svc)
            assert log_all_res["detail"] == "All sessions revoked"

            # change_password
            cp_req = auth_schemas.PasswordChangeRequest(
                current_password="WrongPassword123!",
                new_password="NewSecurePassword123!",
            )
            with pytest.raises(HTTPException) as exc2:
                await auth.change_password(cp_req, user_prof, auth_svc)
            assert exc2.value.status_code == 401


# ---------------------------------------------------------------------------
# Direct Settings Router Tests
# ---------------------------------------------------------------------------


class TestSettingsRouterDirect:
    async def test_settings_get_and_update(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            user = await db.get(User, _TEST_USER_ID)
            assert user is not None

            # Get settings (auto-creates)
            s_get = await user_settings.get_settings(user, db)
            assert s_get["theme"] == "dark"

            # Update settings
            s_put = await user_settings.update_settings(
                {"confidence_threshold": 0.85, "default_export_format": "csv", "theme": "light"},
                user,
                db,
            )
            assert s_put["theme"] == "light"
            assert s_put["confidence_threshold"] == 0.85


# ---------------------------------------------------------------------------
# Direct Batch Router Tests
# ---------------------------------------------------------------------------


class TestBatchRouterDirect:
    async def test_batch_upload_status_download(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            user = await db.get(User, _TEST_USER_ID)
            assert user is not None

            # Invalid file extension raises 400
            invalid_file = UploadFile(io.BytesIO(b"data"), filename="test.pdf")
            from fastapi import BackgroundTasks

            bg = BackgroundTasks()
            with pytest.raises(HTTPException) as exc1:
                await batch.upload_batch_file(invalid_file, bg, user, db)
            assert exc1.value.status_code == 400

            # Valid CSV upload
            valid_file = UploadFile(io.BytesIO(b"text\nGreat!"), filename="test.csv")
            b_resp = await batch.upload_batch_file(valid_file, bg, user, db)
            assert b_resp.job_id is not None

            # Status check
            status_resp = await batch.get_batch_status(b_resp.job_id, user, db)
            assert status_resp.status == "PENDING"

            # Download non-completed job raises 400
            with pytest.raises(HTTPException) as exc2:
                await batch.download_batch_results(b_resp.job_id, user, db)
            assert exc2.value.status_code == 400

            # Status nonexistent raises 404
            with pytest.raises(HTTPException) as exc3:
                await batch.get_batch_status("nonexistent-id", user, db)
            assert exc3.value.status_code == 404


# ---------------------------------------------------------------------------
# Direct History Router Tests
# ---------------------------------------------------------------------------


class TestHistoryRouterDirect:
    async def test_history_get_and_export(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            user = await db.get(User, _TEST_USER_ID)
            assert user is not None

            p = Prediction(
                id="pred-direct-1",
                user_id=user.id,
                text_content="Direct test prediction text",
                sentiment_label="POSITIVE",
                confidence_score=0.92,
                positive_prob=0.92,
                neutral_prob=0.05,
                negative_prob=0.03,
                latency_ms=8.5,
            )
            db.add(p)
            await db.commit()

            # Get history
            h_list = await history.get_history(
                search="Direct",
                sentiment="POSITIVE",
                sentiment_filter=None,
                limit=10,
                offset=0,
                current_user=user,
                db=db,
            )
            assert len(h_list) == 1

            # Export history CSV
            csv_resp = await history.export_history_csv(user, db)
            assert csv_resp.media_type == "text/csv"
