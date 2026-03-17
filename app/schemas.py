"""
Schémas Pydantic utilisés par l'API RAG.

Ce module définit les structures de données échangées
entre le client et l'API.

Les modèles permettent :
- la validation automatique des requêtes
- la structuration des réponses
- la génération automatique de la documentation Swagger
  via FastAPI.

Chaque modèle correspond à une requête ou une réponse
utilisée par les endpoints de l'API.
"""

from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    """
    Représente un document récupéré par le moteur RAG.

    Dans ce projet, un document correspond à un événement
    culturel provenant d'OpenAgenda. Ces documents sont
    retournés dans la réponse afin d'indiquer les sources
    utilisées pour générer la réponse.
    """

    title: str = ""
    location_name: str = ""
    city: str = ""
    region: str = ""
    first_date: str = ""
    last_date: str = ""
    event_type: str = ""
    url: str = ""
    score: float | None = None


class AskRequest(BaseModel):
    """
    Modèle de requête envoyé à l'endpoint `/ask`.
    """

    question: str = Field(
        ...,
        min_length=1,
        description="Question posée au chatbot."
    )


class AskResponse(BaseModel):
    """
    Réponse retournée par l'endpoint `/ask`.
    """

    question: str
    answer: str
    n_docs: int
    documents: list[RetrievedDocument] = Field(default_factory=list)


class RebuildRequest(BaseModel):
    """
    Requête envoyée à l'endpoint `/rebuild`.

    Elle permet de préciser la zone géographique utilisée
    pour reconstruire la base documentaire du système RAG.
    """

    zone: str | None = Field(
        default=None,
        description="Zone géographique recherchée.",
        examples=["Montpellier"]
    )

    scope: str | None = Field(
        default=None,
        description="Type de zone utilisé pour la recherche (city, region, etc.).",
        examples=["city"]
    )


class RebuildResponse(BaseModel):
    """
    Réponse retournée par l'endpoint `/rebuild`.
    """

    status: str
    message: str
    n_docs_indexed: int


class HealthResponse(BaseModel):
    """
    Réponse retournée par l'endpoint `/health`.
    """

    status: str
    index_loaded: bool