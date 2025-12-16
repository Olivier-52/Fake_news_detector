import os
import mlflow
import pickle
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import asyncio
from contextlib import asynccontextmanager

load_dotenv()

# Variables MLflow : URI de tracking, nom du modèle et stage
MLFLOW_TRACKING_APP_URI = os.getenv("MLFLOW_TRACKING_APP_URI")
MODEL_NAME = os.getenv("MODEL_NAME")
STAGE = os.getenv("STAGE", "production")

# Variables AWS pour accéder au bucket S3 qui contient les artifacts de MLflow
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

# Variables globales pour stocker le modèle et le vectorizer
model = None
vectorizer = None

def load_model():
    """
    Charge le modèle depuis MLflow.

    Charge le modèle depuis MLflow en utilisant l'URI de tracking
    MLFLOW_TRACKING_APP_URI, le nom du modèle MODEL_NAME, et le stage STAGE.
     
    Si le modèle n'existe pas, leve une exception HTTPException avec un code
    d'état 500 et un message d'erreur détaillé.

    Returns:
        None
    """
    global model
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_APP_URI)
        model_uri = f"models:/{MODEL_NAME}@{STAGE}"
        model = mlflow.sklearn.load_model(model_uri)
        print("Modèle chargé avec succès depuis MLflow.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle depuis MLflow : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Impossible de charger le modèle depuis MLflow : {e}"
        )

def load_vectorizer():
    """
    Charge le vectorizer depuis MLflow.

    Charge le vectorizer depuis MLflow en utilisant l'URI de tracking
    MLFLOW_TRACKING_APP_URI, le nom du modèle MODEL_NAME, et le stage STAGE.
     
    Si le vectorizer n'existe pas, leve une exception HTTPException avec un code
    d'état 500 et un message d'erreur détaillé.

    Returns:
        Le vectorizer chargé depuis MLflow.
    """
    try:
        client = mlflow.MlflowClient(MLFLOW_TRACKING_APP_URI)

        model_info = client.get_model_version_by_alias(MODEL_NAME, STAGE)
        run_id = model_info.run_id

        local_path = mlflow.artifacts.download_artifacts(
            artifact_path="vectorizer.pkl",
            run_id=run_id
        )

        with open(local_path, "rb") as f:
            vectorizer = pickle.load(f)

        return vectorizer
    except Exception as e:
        print(f"Erreur lors du chargement du vectorizer : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Impossible de charger le vectorizer : {e}"
        )

async def load_model_and_vectorizer():
    """
    Charge le modèle et le vectorizer depuis MLflow en parallèle.

    Charge le modèle et le vectorizer en utilisant l'URI de tracking
    MLFLOW_TRACKING_APP_URI, le nom du modèle MODEL_NAME, et le stage STAGE.
     
    Si le modèle ou le vectorizer n'existe pas, leve une exception HTTPException
    avec un code d'état 500 et un message d'erreur détaillé.

    Returns:
        None
    """
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, load_model)
        global vectorizer
        vectorizer = await loop.run_in_executor(None, load_vectorizer)
        print("Modèle et vectorizer chargés avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Impossible de charger le modèle ou le vectorizer : {e}"
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Charge le modèle et le vectorizer au démarrage de l'application de maniere asynchrone.
    """
    await load_model_and_vectorizer()
    yield
    

app = FastAPI(
    title="Climate Fake News Detector API",
    description="API pour détecter les fake news sur le climat",
    version="1.0.0",
    lifespan=lifespan
)

class TextInput(BaseModel):
    text: str

@app.get("/")
async def read_root():
    """
    Renvoie un message de bienvenue sur l'API ainsi que le lien vers la documentation.
    """
    return {
        "message": "Bienvenue sur l'API Climate Fake News Detector !",
        "documentation": "Consultez la documentation de l'API à l'adresse /docs."    
    }

@app.post("/predict")
async def predict(input_data: TextInput):
    """
    Fait une prédiction sur un texte donné en utilisant le modèle et le vectorizer chargés.

    Args:
        input_data (TextInput): Objet contenant le texte à prédire.

    Returns:
        dict: Dictionnaire contenant la prédiction (0 les articles avec un biais, 1 pour les articles faux, et 2 pour les articles fiable).

    Raises:
        HTTPException: Si le modèle ou le vectorizer n'est pas chargé ou si une erreur survient lors de la prédiction.
    """
    global model, vectorizer
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Le modèle ou le vectorizer n'est pas chargé."
        )

    try:
        X_vectorized = vectorizer.transform([input_data.text]).toarray()
        prediction = model.predict(X_vectorized)
        return {"prediction": int(prediction[0])}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de la prédiction : {e}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)