# 📋 GitHub Release Checklist

Use this checklist prior to creating official v1.0.0 releases or submitting for hackathons/recruiter reviews.

---

## 📌 Release Verification Steps

- [x] **Backend Test Suite:** Run `pytest` (Verify 130 / 130 tests pass).
- [x] **Backend Coverage:** Run `pytest --cov=backend/app --cov-report=term-missing` (Verify ≥ 90% coverage).
- [x] **Strict Type Checking:** Run `mypy backend/app --strict` (0 errors across 68 files).
- [x] **Code Formatting:** Run `black --check backend/app` and `isort --check-only backend/app`.
- [x] **PEP 8 Compliance:** Run `flake8 backend/app`.
- [x] **Frontend Type Check:** Run `npm run lint` (`tsc --noEmit`).
- [x] **Frontend Component Tests:** Run `npx vitest run` (12 / 12 passed).
- [x] **Frontend Production Build:** Run `npm run build` (Verify core bundle size ≤ 100 kB).
- [x] **Docker Build Verification:** Test `docker compose -f deployment/docker-compose.yml up --build`.
- [x] **Documentation Complete:** Verify `README.md`, `docs/ARCHITECTURE.md`, `docs/API_DOCUMENTATION.md`.
