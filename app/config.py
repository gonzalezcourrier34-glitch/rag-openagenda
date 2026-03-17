"""
Module de configuration de l'application.

Ce module centralise le chargement des variables d'environnement
utilisées par l'API RAG OpenAgenda.

Les variables sont récupérées depuis un fichier `.env` via la
bibliothèque python-dotenv.

Variables utilisées
-------------------
OPENAGENDA_API_KEY
    Clé API permettant d'interroger l'API OpenAgenda.

ZONE_CHOISIE
    Zone géographique par défaut utilisée pour récupérer les événements.

TYPE_ZONE
    Niveau géographique utilisé pour filtrer les événements
    (ex : city, region, etc.).
"""

import os
from dotenv import load_dotenv

# Chargement des variables définies dans le fichier .env
load_dotenv()

# Clé API OpenAgenda utilisée pour interroger l'API officielle
OPENAGENDA_API_KEY = os.getenv("OPENAGENDA_API_KEY", "")

# URL de la base de données
DATABASE_URL = os.getenv("DATABASE_URL", "")

# URL mlflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# URL api pour l'appel 
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Zone géographique par défaut utilisée pour les recherches
DEFAULT_ZONE = os.getenv("ZONE_CHOISIE", "Montpellier")

# Portée géographique du filtre (ex : city, region, country)
DEFAULT_SCOPE = os.getenv("TYPE_ZONE", "city")

# Sécurité API
API_KEY = os.getenv("API_KEY", "")