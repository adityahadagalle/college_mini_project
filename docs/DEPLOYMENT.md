# 🚀 SentimentAI Enterprise — Production Deployment Guide

This guide provides step-by-step instructions for deploying SentimentAI Enterprise across cloud providers and Docker environments.

---

## 🐳 Option 1: Docker Compose Deployment

The simplest production deployment method uses Docker Compose and NGINX.

```bash
# Clone the repository
git clone https://github.com/your-username/sentiment-analysis-system.git
cd sentiment-analysis-system

# Build and start services in background mode
docker compose -f deployment/docker-compose.yml up --build -d

# Verify health status
docker compose -f deployment/docker-compose.yml ps
```
Access the application at `http://localhost`.

---

## ☁️ Option 2: Railway / Render (PaaS)

### Railway Deployment (1-Click)
1. Fork repository to GitHub.
2. Log into Railway.app and create a new project from GitHub.
3. Add Environment Variables:
   - `ENVIRONMENT=production`
   - `SECRET_KEY=PROD_SECRET_KEY_32_BYTES_MINIMUM`
   - `DATABASE_URL=sqlite:///./sentiment.db`
4. Set Dockerfile path to `backend/Dockerfile`.

---

## 🌐 Option 3: AWS App Runner / ECS

1. Push Backend Container to AWS ECR:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
   docker build -t sentiment-backend ./backend
   docker tag sentiment-backend:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/sentiment-backend:latest
   docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/sentiment-backend:latest
   ```
2. Provision App Runner service pointing to ECR image.
3. Configure HTTPS domain mapping in AWS Route 53.

---

## 🔐 Environment Variables Reference

| Variable Name | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `ENVIRONMENT` | Yes | `development` | Deployment environment (`production` / `development`) |
| `SECRET_KEY` | Yes | (Random 32B) | JWT signature secret key |
| `DATABASE_URL` | Yes | `sqlite:///./sentiment.db` | SQLAlchemy database connection string |
| `ALLOWED_ORIGINS` | No | `*` | Allowed CORS origins (comma-separated URLs) |
