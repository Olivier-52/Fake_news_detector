import os
import mlflow
import pickle
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import asyncio
from contextlib import asynccontextmanager

# Charge les variables d'environnement
load_dotenv()

# Configuration des variables d'environnement
MLFLOW_TRACKING_APP_URI = os.getenv("MLFLOW_TRACKING_APP_URI")
MODEL_NAME = os.getenv("MODEL_NAME")
STAGE = os.getenv("STAGE")

# Configure les identifiants AWS pour accéder au bucket S3
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

# Variables globales pour stocker le modèle et le vectorizer
model = None
vectorizer = None

# Fonction pour charger le modèle depuis MLflow
def load_model():
    global model
    try:
        # Configure l'URI de tracking MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_APP_URI)

        # Charge le modèle depuis MLflow
        model_uri = f"models:/{MODEL_NAME}@{STAGE}"
        model = mlflow.sklearn.load_model(model_uri)
        print("Modèle chargé avec succès depuis MLflow.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle depuis MLflow : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Impossible de charger le modèle depuis MLflow : {e}"
        )

# Fonction pour charger le vectorizer depuis MLflow
def load_vectorizer():
    try:
        # Initialise le client MLflow
        client = mlflow.MlflowClient(MLFLOW_TRACKING_APP_URI)

        # Récupère les informations sur le modèle
        model_info = client.get_model_version_by_alias(MODEL_NAME, STAGE)
        run_id = model_info.run_id

        # Télécharge le fichier vectorizer.pkl depuis MLflow
        local_path = mlflow.artifacts.download_artifacts(
            artifact_path="vectorizer.pkl",
            run_id=run_id
        )

        # Charge le vectorizer depuis le fichier
        with open(local_path, "rb") as f:
            vectorizer = pickle.load(f)

        return vectorizer
    except Exception as e:
        print(f"Erreur lors du chargement du vectorizer : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Impossible de charger le vectorizer : {e}"
        )

# Fonction asynchrone pour charger le modèle et le vectorizer
async def load_model_and_vectorizer():
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

# Charge le modèle et le vectorizer au démarrage
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code à exécuter au démarrage
    await load_model_and_vectorizer()
    yield

# Initialise FastAPI
app = FastAPI(
    title="Climate Fake News Detector API",
    description="API pour détecter les fake news sur le climat",
    version="1.0.0",
    lifespan=lifespan
)

# Modèle pour les données d'entrée
class TextInput(BaseModel):
    text: str

@app.get("/")
async def read_root():
    return {
        "message": "Bienvenue sur l'API Climate Fake News Detector !",
        "documentation": "Consultez la documentation de l'API à l'adresse /docs."    
    }

@app.post("/predict")
async def predict(input_data: TextInput):
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