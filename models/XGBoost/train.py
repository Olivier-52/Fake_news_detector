import pandas as pd
import time
import mlflow
import os
import pickle
from mlflow.models.signature import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from dotenv import load_dotenv

load_dotenv()
# Tracking URI (HF Space)
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_APP_URI"])
MLFLOW_TRACKING_APP_URI= os.environ["MLFLOW_TRACKING_APP_URI"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

URL_TRAIN_DATA = "hf://datasets/readerbench/fakenews-climate-fr/fake-fr.csv"
EXPERIMENT_NAME = "Climate_Fake_News_Detector_Project"
TARGET_COLUMN = "Label"

if __name__ == "__main__":

    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id)

    print("training model...")

    start_time = time.time()

    mlflow.sklearn.autolog(log_models=False)
    
    df = pd.read_csv(URL_TRAIN_DATA)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    vectorizer = TfidfVectorizer(
    max_features=25000,
    ngram_range=(1, 2)
    )

    X_vectorized = vectorizer.fit_transform(X["Text"]).toarray()

    # Model hyperparameter dictionary
    hyperparameters = {'colsample_bytree': 0.6, 
                       'learning_rate': 0.12, 
                       'max_depth': 5, 
                       'n_estimators': 200, 
                       'random_state': 42, 
                       'subsample': 0.4}

    model = XGBClassifier(**hyperparameters)

    with mlflow.start_run(run_id=run.info.run_id) as run:
        model.fit(X_vectorized, y)
        predictions = model.predict(X_vectorized)

        # Log model seperately to have more flexibility on setup
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="climate-fake-news-detector-model",
            registered_model_name="climate-fake-news-detector-model-XGBoost-v1",
            signature=infer_signature(X, predictions),
        )

        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        
        mlflow.log_artifact("vectorizer.pkl")

        os.remove("vectorizer.pkl")

    print("...Done!")
    print(f"---Total training time: {time.time()-start_time:.2f}s")