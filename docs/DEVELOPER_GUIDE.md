# 💻 SentimentAI Enterprise — Developer & Setup Guide

Welcome to the developer setup guide for SentimentAI Enterprise. Follow these steps to set up your local development environment.

---

## ⚙️ Prerequisites

- Python 3.10+
- Node.js 18+
- Git

---

## 🛠️ Local Environment Setup

### 1. Repository Setup
```bash
git clone https://github.com/your-username/sentiment-analysis-system.git
cd sentiment-analysis-system
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
API server running at: `http://localhost:8000`

### 3. Frontend Setup
```bash
cd ../frontend
npm install
npm run dev
```
Frontend SPA running at: `http://localhost:5173`

---

## 🧪 Running Automated Quality Gates

Run all quality checks prior to submitting pull requests:

```bash
# Frontend Validation
cd frontend
npm run lint          # TypeScript type checking
npm run build         # Vite production build
npx vitest run        # Vitest component test suite

# Backend Validation
cd backend
black --check app/    # Code format check
isort --check-only app/ # Import order check
flake8 app/           # Flake8 PEP 8 check
mypy app/ --strict    # Strict static type check
pytest                # Integration test suite
pytest --cov=app --cov-report=term-missing # Coverage check
```
