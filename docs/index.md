# Assistant RAG OpenAgenda

## Objectif

Je développe un assistant intelligent permettant de recommander des événements culturels.

Le système repose sur une architecture RAG combinant :
- récupération de documents
- index vectoriel FAISS
- modèle de langage
- API FastAPI
- interface Streamlit

## Architecture

- API FastAPI
- Dashboard Streamlit
- MLflow
- PostgreSQL

## Fonctionnement

1. récupération des événements
2. transformation des données
3. indexation vectorielle
4. recherche de contexte
5. génération de réponse