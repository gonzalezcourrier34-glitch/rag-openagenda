"""
Schémas Pydantic utilisés par l'API RAG.

Ce module définit les structures de données échangées
entre le client et l'API.

Les modèles permettent :
- la validation automatique des requêtes
- la structuration des réponses
- la génération automatique de la documentation Swagger
  via FastAPI

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

    title: str = Field(default="", description="Titre de l'événement.")
    location_name: str = Field(default="", description="Nom du lieu de l'événement.")
    city: str = Field(default="", description="Ville de l'événement.")
    region: str = Field(default="", description="Région de l'événement.")
    first_date: str = Field(default="", description="Date de début de l'événement.")
    last_date: str = Field(default="", description="Date de fin de l'événement.")
    event_type: str = Field(default="", description="Type d'événement.")
    url: str = Field(default="", description="URL source de l'événement.")
    score: float | None = Field(
        default=None,
        description="Score éventuel associé au document lors du retrieval."
    )


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

    question: str = Field(description="Question envoyée par l'utilisateur.")
    answer: str = Field(description="Réponse générée par le système RAG.")
    n_docs: int = Field(description="Nombre de documents utilisés pour produire la réponse.")
    documents: list[RetrievedDocument] = Field(
        default_factory=list,
        description="Documents sources retournés par le moteur RAG."
    )


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

    status: str = Field(description="Statut de l'opération de reconstruction.")
    message: str = Field(description="Message décrivant le résultat de l'opération.")
    n_docs_indexed: int = Field(description="Nombre de documents indexés.")


class HealthResponse(BaseModel):
    """
    Réponse retournée par l'endpoint `/health`.
    """

    status: str = Field(description="Statut général du service.")
    index_loaded: bool = Field(description="Indique si l'index vectoriel est chargé en mémoire.")