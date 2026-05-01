#!/bin/bash
set -e
PROJECT_ID="parkcast"
REGION="us-central1"
SERVICE_NAME="parkcast-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
MLFLOW_URI="http://34.133.160.231:5000"
GCS_BUCKET="parkcast-bucket"
GCS_PREFIX="Data/"

gcloud services enable run.googleapis.com --project=${PROJECT_ID}
gcloud services enable containerregistry.googleapis.com --project=${PROJECT_ID}
gcloud auth configure-docker

docker buildx build --platform linux/amd64 -t ${IMAGE_NAME}:latest -f Dockerfile.cloudrun --push .

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
  --set-env-vars="MLFLOW_TRACKING_URI=${MLFLOW_URI},GCS_BUCKET=${GCS_BUCKET},GCS_PREFIX=${GCS_PREFIX}" \
  --project=${PROJECT_ID}

gcloud run services describe ${SERVICE_NAME} --platform=managed --region=${REGION} --format='value(status.url)' --project=${PROJECT_ID}
