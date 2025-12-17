# Détection des fausses informations sur le réchauffement climatique

**Outils pour classifier automatiquement les articles sur le climat en "vrai", "biaisé" ou "faux".**

---

## 📝 À propos

Les fausses informations et les contenus manipulateurs sur le climat se propagent rapidement, nuisant à la lutte contre le réchauffement climatique. Ce projet vise à automatiser la classification des articles en trois catégories : **vrai**, **biaisé** ou **faux**.

**Objectifs :**
- Améliorer la modération des contenus en ligne.
- Assister les journalistes dans la vérification des informations.
- Augmenter la qualité de l’information disponible pour le public.
- Réduire la diffusion des fausses informations.

**Avantages :**
- Réduction du temps de vérification manuelle.
- Automatisation des tâches répétitives.
- Protection des lecteurs contre la désinformation.

---

## ✨ Fonctionnalités

### Modèles de Machine Learning
Le projet inclut cinq modèles entraînables via le script `train.py` :
- CamemBERT
- Régression logistique
- Naive Bayes
- Support Vector Machine (SVM)
- XGBoost

### API FastAPI
- **Déploiement simplifié** via Docker.
- **Sélection dynamique du modèle** grâce aux variables d’environnement :
  - `MODEL_NAME` : Nom du modèle à utiliser.
  - `STAGE` : Alias pour sélectionner la version du modèle (via MLFlow).

### Interface utilisateur (GUI) avec Streamlit
- **Déploiement facile** via Docker.
- **Intégration avec l’API FastAPI** pour les prédictions.

---

## 🔧 Prérequis
- **Python 3.10 ou supérieur**
- **Docker**
- **Environnement [MLFlow](https://mlflow.org/docs/latest/genai/getting-started/connect-environment/) (version 2.21.3)**
- **Librairies Python** : Installées automatiquement via `requirements.txt` (spécifique à chaque modèle).
- **Accès à un [bucket S3 AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html)** (pour les artefacts MLFlow).

---

## 🛠 Installation

### 1. Cloner le projet
```bash
git clone https://github.com/Olivier-52/Fake_news_detector.git
cd Fake_news_detector
```

### 2. Entraîner les modèles souhaités

Pour chaque modèle, se placer dans le répertoire correspondant :
```Bash
cd models/\$NOM_DU_MODELE
pip install -r requirements.txt
```
Créer un fichier .env avec les variables suivantes :

```
MLFLOW_TRACKING_APP_URI=Endpoint_de_votre_serveur_MLFlow
AWS_ACCESS_KEY_ID=Votre_ID_clé_AWS
AWS_SECRET_ACCESS_KEY=Votre_clé_secrète_AWS
```
Puis lancer l’entraînement :
```Bash
python train.py
```
### 3. Déployer l’API FastAPI (Backend)

Construire l’image Docker depuis le répertoire app_fastapi.
Variables d’environnement requises :

L'application comporte les variables d'environnement suivantes :
```
MLFLOW_TRACKING_APP_URI=Endpoint_du_serveur_MLFlow
MODEL_NAME=Nom_du_modèle
STAGE=Alias_du_modèle
AWS_ACCESS_KEY_ID=Votre_ID_clé_AWS
AWS_SECRET_ACCESS_KEY=Votre_clé_secrète_AWS
```

### 4. Déployer l'application Streamlit (Frontend)

Construire l’image Docker depuis le répertoire app_streamlit.
Variable d’environnement requise 

```
API_URL=URL_du_endpoint_/predict
```

---

## 📂 Utilisation

Accéder à l’interface Streamlit via un navigateur.
Saisir le texte de l’article dans la zone prévue.
Cliquer sur "Vérifier la nouvelle" pour obtenir la prédiction.

Résultats possibles :
- Probablement vrai
- Probablement faux
- Biaisé (si l’article contient un biais identifiable)

Capture d’écran :

Page d’accueil :

![Page d’accueil](/images/FakeNews_app_homepage.png)

Résultat de prédiction :

![Résultat de prédiction](/images/FakeNews_app_utilisation.png)

---

## 📜 Licence
Ce projet est sous licence MIT.