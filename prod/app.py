import mlflow
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

# Defining path operation for root endpoint


@app.get('/')
def main():
    return {'message': 'This is a model for classifying Reddit comments'}


class request_body(BaseModel):
    reddit_comment: str


@app.lifespan('startup')
def load_artifacts():
    mlflow.set_tracking_uri("https://parkcast:5000/")
    global model_pipeline
    model_pipeline = mlflow.load_model("lasso-demo")


# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(data: request_body):
    X = [data.reddit_comment]
    predictions = model_pipeline.predict_proba(X)
    return {'Predictions': predictions}


@app.get('/health')
def health():
    return {'status': 'ok'}
