# Air Paradis : analyse de sentiment de tweets

Projet 7 du parcours Ingénieur IA d'OpenClassrooms.

## Objectif

Air Paradis souhaite anticiper les bad buzz sur les réseaux sociaux. Ce projet livre un modèle de classification de sentiment de tweets (positif / négatif), entraîné sur le dataset Sentiment140 (1,6 million de tweets) et exposé via une API.

Trois approches de modélisation sont comparées :

1. Modèle classique : TF-IDF + Régression Logistique
2. Modèle sur mesure avancé : LSTM (PyTorch) + embeddings GloVe Twitter 200d
3. Modèle BERT : fine-tuning de BERTweet

Le tracking des expérimentations et le stockage centralisé des modèles sont gérés avec MLflow (backend SQLite local + Model Registry).

## Découpage des dossiers

```
.
├── notebooks/                  Notebooks de modélisation
│   ├── 01_exploration.ipynb           Exploration du dataset
│   ├── 02_model_classique.ipynb       Approche 1 (TF-IDF + LR)
│   ├── 03_download_embeddings.ipynb   Téléchargement GloVe / FastText
│   ├── 04_model_avance_pytorch.ipynb  Approche 2 (LSTM PyTorch + embeddings)
│   └── 05_model_bertweet.ipynb        Approche 3 (BERTweet fine-tuné)
│
├── src/                        Modules partagés entre notebooks
│   └── preprocessing.py                Nettoyage texte (NLTK)
│
├── api/                        API FastAPI de prédiction
│   ├── main.py                         Endpoints /predict, /feedback, /health
│   ├── model_loader.py                 Chargement et inférence du modèle
│   └── requirements.txt                Dépendances de l'API
│
├── streamlit_app/              Interface locale de test de l'API
│   ├── app.py
│   └── requirements.txt
│
├── tests/                      Tests unitaires (pytest)
│   ├── test_api.py
│   └── test_preprocessing.py
│
├── .github/workflows/          Pipeline CI/CD
│   └── deploy.yml                      Tests puis déploiement Azure App Service
│
├── requirements.txt            Dépendances de l'environnement d'entraînement
└── .gitignore
```

## Lancer en local

Entraînement (notebooks) :

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet
jupyter notebook
```

API :

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload
# http://localhost:8000
```

Streamlit :

```bash
pip install -r streamlit_app/requirements.txt
API_URL=http://localhost:8000 streamlit run streamlit_app/app.py
```

Tests :

```bash
python -m pytest tests/ -v
```

## CI/CD

À chaque push sur `main`, le workflow `.github/workflows/deploy.yml` :

1. Job `test` : lance pytest
2. Job `deploy` : si les tests passent, déploie l'API sur Azure App Service

Secrets GitHub requis : `AZURE_WEBAPP_NAME`, `AZURE_WEBAPP_PUBLISH_PROFILE`.

## MLflow

Lancer l'UI :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# http://localhost:5000
```

La base `mlflow.db` et le dossier `mlruns/` sont exclus du dépôt (volume) et fournis avec les livrables du projet.
