#!/bin/bash
# ── ParkCast SF — Deploy to Cloud Run ────────────────────────
# Run this from your project root folder
# Requirements: gcloud CLI authenticated, Docker installed

set -e  # exit on any error

# ── Config — update these ─────────────────────────────────────
PROJECT_ID="parkcast"
REGION="us-central1"
SERVICE_NAME="parkcast-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
MLFLOW_URI="http://34.133.160.231:5000"

echo "=========================================="
echo "ParkCast SF — Cloud Run Deployment"
echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Service:  ${SERVICE_NAME}"
echo "=========================================="

# Step 1 — Enable required APIs
echo ""
echo "Step 1 — Enabling Cloud Run and Container Registry APIs..."
gcloud services enable run.googleapis.com --project=${PROJECT_ID}
gcloud services enable containerregistry.googleapis.com --project=${PROJECT_ID}

# Step 2 — Authenticate Docker with GCR
echo ""
echo "Step 2 — Authenticating Docker with Google Container Registry..."
gcloud auth configure-docker

# Step 3 — Build multi-arch image for Cloud Run (linux/amd64)
echo ""
echo "Step 3 — Building Docker image for linux/amd64..."
docker buildx build \
  --platform linux/amd64 \
  -t ${IMAGE_NAME}:latest \
  -f Dockerfile.cloudrun \
  --push \
  .

echo "  Image pushed to: ${IMAGE_NAME}:latest"

# Step 4 — Deploy to Cloud Run
echo ""
echo "Step 4 — Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image=${IMAGE_NAME}:latest \
  --platform=managed \
  --region=${REGION} \
  --allow-unauthenticated \
  --port=8080 \
  --memory=1Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=10 \
  --set-env-vars="MLFLOW_TRACKING_URI=${MLFLOW_URI}" \
  --project=${PROJECT_ID}

# Step 5 — Get the deployed URL
echo ""
echo "Step 5 — Getting deployed service URL..."
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
  --platform=managed \
  --region=${REGION} \
  --format='value(status.url)' \
  --project=${PROJECT_ID})

echo ""
echo "=========================================="
echo "DEPLOYMENT COMPLETE!"
echo "=========================================="
echo "Service URL:  ${SERVICE_URL}"
echo "Health check: ${SERVICE_URL}/health"
echo "API docs:     ${SERVICE_URL}/docs"
echo ""
echo "Test it:"
echo "curl ${SERVICE_URL}/health"
echo ""
echo "Update your Next.js next.config.mjs:"
echo "destination: '${SERVICE_URL}/:path*'"
echo "=========================================="
