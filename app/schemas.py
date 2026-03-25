"""
Schémas Pydantic utilisés par l'API RAG.

Ce module définit les structures de données échangées
entre le client et l'API.

Les modèles permettent :
- la validation des requêtes
- la structuration des réponses
- la documentation automatique de l'API
- une cohérence claire entre les services internes et les endpoints
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class RetrievedDocument(BaseModel):
    """
    Représente un document source retourné par le système RAG.

    Dans ce projet, un document correspond à un événement culturel
    provenant d'OpenAgenda.

    Le schéma expose les métadonnées utiles à :
    - l'affichage côté client
    - l'interprétation de la réponse générée
    - l'inspection du retrieval lors du debug
    """

    title: str = Field(
        default="",
        description="Titre de l'événement.",
    )
    location_name: str = Field(
        default="",
        description="Nom du lieu où se déroule l'événement.",
    )
    city: str = Field(
        default="",
        description="Ville de l'événement.",
    )
    region: str = Field(
        default="",
        description="Région associée à l'événement.",
    )
    first_date: str = Field(
        default="",
        description="Date de début de l'événement.",
    )
    last_date: str = Field(
        default="",
        description="Date de fin de l'événement.",
    )
    event_type: str = Field(
        default="",
        description="Type d'événement.",
    )
    url: str = Field(
        default="",
        description="Lien vers l'événement source.",
    )
    price_info: str = Field(
        default="",
        description="Information de tarification simplifiée de l'événement.",
    )
    is_free: bool | None = Field(
        default=None,
        description="Indique si l'événement est identifié comme gratuit.",
    )
    keywords_title: list[str] = Field(
        default_factory=list,
        description="Mots-clés extraits du titre de l'événement.",
    )
    score: float | None = Field(
        default=None,
        description=(
            "Score éventuel du document si le système décide "
            "de l'exposer dans le futur."
        ),
    )


class AskRequest(BaseModel):
    """
    Requête envoyée à l'endpoint `/ask`.

    Ce schéma contient uniquement la question utilisateur à transmettre
    au pipeline RAG.
    """

    question: str = Field(
        ...,
        min_length=1,
        description="Question utilisateur.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """
        Vérifie que la question n'est pas vide après nettoyage.

        Parameters
        ----------
        value : str
            Question brute reçue dans la requête.

        Returns
        -------
        str
            Question nettoyée.

        Raises
        ------
        ValueError
            Si la question est vide ou ne contient que des espaces.
        """
        value = value.strip()
        if not value:
            raise ValueError("La question ne peut pas être vide.")
        return value


class AskResponse(BaseModel):
    """
    Réponse retournée par l'endpoint `/ask`.

    Cette réponse contient :
    - la question utilisateur
    - la réponse générée par le modèle
    - le nombre de documents retenus
    - la liste structurée des documents utilisés
    """

    question: str = Field(
        description="Question utilisateur.",
    )
    answer: str = Field(
        description="Réponse générée par le système RAG.",
    )
    n_docs: int = Field(
        description="Nombre de documents retenus pour construire la réponse.",
    )
    documents: list[RetrievedDocument] = Field(
        default_factory=list,
        description="Documents utilisés pour construire la réponse.",
    )


class DebugResponse(BaseModel):
    """
    Réponse retournée par l'endpoint `/ask/debug`.

    Ce schéma permet d'inspecter plus finement le comportement
    du pipeline RAG.

    Il expose :
    - la réponse générée
    - les documents récupérés
    - les contextes textuels transmis au modèle
    """

    question: str = Field(
        description="Question utilisateur.",
    )
    answer: str = Field(
        description="Réponse générée par le système RAG.",
    )
    n_docs: int = Field(
        description="Nombre de documents récupérés et retenus.",
    )
    documents: list[RetrievedDocument] = Field(
        default_factory=list,
        description="Documents récupérés avec leurs métadonnées.",
    )
    retrieved_contexts: list[str] = Field(
        default_factory=list,
        description="Liste des contextes textuels utilisés pour la génération.",
    )


class RebuildRequest(BaseModel):
    """
    Requête envoyée à l'endpoint `/rebuild`.

    Elle permet de relancer le chargement des documents et la
    reconstruction de l'index vectoriel à partir d'une zone donnée.
    """

    zone: str | None = Field(
        default=None,
        description="Zone géographique à charger.",
        examples=["Montpellier"],
    )
    scope: str | None = Field(
        default=None,
        description="Portée géographique utilisée pour filtrer les événements.",
        examples=["city"],
    )


class RebuildResponse(BaseModel):
    """
    Réponse retournée par l'endpoint `/rebuild`.

    Elle résume le résultat de l'opération de reconstruction
    de l'index vectoriel.
    """

    status: str = Field(
        description="Statut global de l'opération.",
    )
    message: str = Field(
        description="Message décrivant le résultat de l'opération.",
    )
    n_docs_indexed: int = Field(
        description="Nombre de documents indexés dans FAISS.",
    )


class HealthResponse(BaseModel):
    """
    Réponse retournée par l'endpoint `/health`.

    Ce schéma permet de vérifier rapidement l'état du service
    ainsi que la disponibilité de l'index vectoriel.
    """

    status: str = Field(
        description="Statut global du service.",
    )
    index_loaded: bool = Field(
        description="Indique si l'index FAISS est chargé en mémoire.",
    )