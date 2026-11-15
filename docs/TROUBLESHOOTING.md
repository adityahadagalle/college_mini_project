# 🔍 SentimentAI Enterprise — Troubleshooting Guide

Common issues and resolution steps for development and deployment.

---

## ❓ Frequently Encountered Issues

### 1. `CORS Error when calling API from Frontend`
- **Cause:** Backend `ALLOWED_ORIGINS` does not match the frontend URL.
- **Fix:** Update `ALLOWED_ORIGINS` in `backend/app/core/config.py` or `.env` file to include `http://localhost:5173`.

### 2. `TensorFlow Model File Not Found`
- **Cause:** The Keras model path `models/` is missing or unbuilt.
- **Fix:** Ensure default model artifacts exist in `models/` directory or trigger model initialization script in `backend/app/training/`.

### 3. `Port 8000 Already in Use`
- **Cause:** Another process is occupying port 8000.
- **Fix:** Kill existing process or run Uvicorn on another port:
  ```bash
  uvicorn app.main:app --port 8001
  ```

### 4. `Database Locked Error (SQLite)`
- **Cause:** Concurrent write locks in SQLite during multi-threaded batch operations.
- **Fix:** Switch `DATABASE_URL` to PostgreSQL in production environments.
