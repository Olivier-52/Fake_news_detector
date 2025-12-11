import os
import uvicorn
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import mlflow
import os
from dotenv import load_dotenv

description = """

# Climate Fake News Detector(https://github.com/Olivier-52/Fake_news_detector.git)

This API allows you to use a Machine Learning model to detect fake news related to climate change.

## Machine-Learning 

Where you can:
* `/predict` : prediction for a single value

Check out documentation for more information on each endpoint. 
"""

tags_metadata = [
    {
        "name": "Predictions",
        "description": "Endpoints that uses our Machine Learning model",
    },
]

load_dotenv()
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_APP_URI"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
MODEL_URI = "models:/climate-fake-news-detector-model-XGBoost-v1@production"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_uri = MODEL_URI 
model = mlflow.sklearn.load_model(model_uri)

app = FastAPI(
    title="API for Climate Fake News Detector",
    description=description,
    version="1.0",
    contact={
        "name": "Olivier",
        "url": "https://github.com/Olivier-52/Fake_news_detector",
    },
    openapi_tags=tags_metadata,)

@app.get("/")
def index():
    """Return a message to the user.

    This endpoint does not take any parameters and returns a message
    to the user. It is used to test the API.

    Returns:
        str: A message to the user.
    """
    return "Hello world! Go to /docs to try the API."


class PredictionFeatures(BaseModel):
    text: str

@app.post("/predict", tags=["Predictions"])
def predict(features: PredictionFeatures):
    """Predict Climate Fake News.

    This endpoint takes a text as input and returns the predicted class : fake, real, or biased.

    Args:
        features (PredictionFeatures): A PredictionFeatures object
            containing the text to analyze.

    Returns:
        dict: A dictionary containing the predicted class.
    """
    df = pd.DataFrame({
        "text": [features.text],
    })
    
    prediction = model.predict(df)[0]
    return {"prediction": float(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
