import pandas as pd
import time
import mlflow
import os
import pickle
import nltk
from nltk.corpus import wordnet
import random
from mlflow.models.signature import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
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
    
    ### Augmentation des données

    df = pd.read_csv(URL_TRAIN_DATA)
    # Téléchargement WordNet si nécessaire
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    # Fonctions d'augmentation
    def synonym_replace(sentence, n=1):
        words = sentence.split()
        new_words = words.copy()
        for _ in range(n):
            word_to_replace = random.choice(words)
            synonyms = wordnet.synsets(word_to_replace)
            if synonyms:
                new_word = synonyms[0].lemmas()[0].name()
                if new_word != word_to_replace:
                    new_words[words.index(word_to_replace)] = new_word
        return ' '.join(new_words)

    def random_swap(sentence, n=1):
        words = sentence.split()
        for _ in range(n):
            if len(words) < 2:
                break
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)


    # Fonction combinée pour augmenter un texte
    def augment_text(sentence, num_augmented=3):
        augmented = []
        for _ in range(num_augmented):
            technique = random.choice(['synonym', 'swap', 'delete'])
            if technique == 'synonym':
                augmented.append(synonym_replace(sentence))
            elif technique == 'swap':
                augmented.append(random_swap(sentence))
        return augmented

    augmented_data = []

    for _, row in df.iterrows():
        new_texts = augment_text(row['Text'], num_augmented=3)  # 3 variantes par texte
        for text in new_texts:
            augmented_data.append({'Text': text, 'Label': row['Label']})

    augmented_df = pd.DataFrame(augmented_data)

    df = pd.concat([df, augmented_df], ignore_index=True)

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 1)
    )

    X_vectorized = vectorizer.fit_transform(X["Text"]).toarray()

    # Model hyperparameter dictionary
    hyperparameters = {
        'C': 3,
        'loss': 'squared_hinge',
                       }

    model = LinearSVC(**hyperparameters)

    with mlflow.start_run(run_id=run.info.run_id) as run:
        model.fit(X_vectorized, y)
        predictions = model.predict(X_vectorized)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="climate-fake-news-detector-model",
            registered_model_name="climate-fake-news-detector-model-SVM-DA-v1",
            signature=infer_signature(X, predictions),
        )

        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        
        mlflow.log_artifact("vectorizer.pkl")

        os.remove("vectorizer.pkl")

    print("...Done!")
    print(f"---Total training time: {time.time()-start_time:.2f}s")