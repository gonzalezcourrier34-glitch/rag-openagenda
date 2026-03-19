"""
Application principale FastAPI pour l'API RAG OpenAgenda.

Ce module expose les endpoints REST de l'application. Il permet :
- de vérifier l'état de l'API
- de poser une question au système RAG
- de reconstruire la base documentaire utilisée par le moteur RAG

Le système repose sur un pipeline Retrieval-Augmented Generation (RAG)
qui récupère des événements culturels via OpenAgenda, les indexe
dans FAISS, puis génère une réponse à partir des documents retrouvés.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from app.config import DEFAULT_SCOPE, DEFAULT_ZONE
from app.document_service import load_documents
from app.rag_service import RAGService
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    RebuildRequest,
    RebuildResponse,
)
from app.security import require_api_key


app = FastAPI(
    title="OpenAgenda RAG API",
    description="API REST locale pour interroger un chatbot RAG d'événements culturels",
    version="1.0.0",
)

rag_service = RAGService()


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """
    Redirige la racine de l'API vers la documentation interactive.

    Returns
    -------
    RedirectResponse
        Redirection vers `/docs`.
    """
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, summary="État de l'API")
def health() -> HealthResponse:
    """
    Vérifie l'état général de l'API.

    Returns
    -------
    HealthResponse
        État global de l'API et disponibilité de l'index vectoriel.
    """
    return HealthResponse(
        status="ok",
        index_loaded=rag_service.is_index_loaded(),
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Posez une question au chatbot RAG",
    dependencies=[Depends(require_api_key)],
)
def ask(payload: AskRequest) -> AskResponse:
    """
    Interroge le système RAG avec une question utilisateur.

    Parameters
    ----------
    payload : AskRequest
        Corps de requête contenant la question utilisateur.

    Returns
    -------
    AskResponse
        Réponse générée par le système RAG, avec les documents utilisés.

    Raises
    ------
    HTTPException
        400 : question invalide ou erreur de données
        503 : index vectoriel indisponible
        500 : erreur interne du serveur
    """
    try:
        return rag_service.ask(payload.question)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception:
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")


@app.post(
    "/ask/debug",
    summary="Endpoint de debug pour inspection du pipeline RAG",
    dependencies=[Depends(require_api_key)],
)
def ask_debug(payload: AskRequest) -> dict:
    """
    Exécute le pipeline RAG en mode debug avec accès complet aux contextes.

    Cet endpoint est destiné à l'analyse, au debugging et à l'évaluation
    du système RAG. Contrairement à `/ask`, il expose explicitement
    les documents récupérés ainsi que leur contenu brut.

    Parameters
    ----------
    payload : AskRequest
        Corps de requête contenant la question utilisateur.

    Returns
    -------
    dict
        Dictionnaire contenant :
        - question : question utilisateur
        - answer : réponse générée par le modèle
        - contexts : liste des contenus textuels des documents récupérés
        - metadata : liste des métadonnées associées aux documents

    Raises
    ------
    HTTPException
        400 : question invalide
        503 : index vectoriel indisponible
        500 : erreur interne du serveur
    """
    try:
        question = payload.question.strip()
        current_date = datetime.today().strftime("%Y-%m-%d")

        if not question:
            raise ValueError("La question ne peut pas être vide.")

        docs = rag_service.retrieve(question)
        answer = rag_service.generate(
            question=question,
            docs=docs,
            current_date=current_date,
        )

        return {
            "question": question,
            "answer": answer,
            "contexts": [doc.page_content for doc in docs],
            "metadata": [doc.metadata for doc in docs],
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Erreur interne du serveur.",
        )


@app.post(
    "/rebuild",
    response_model=RebuildResponse,
    summary="Reconstruire l'index documentaire",
    dependencies=[Depends(require_api_key)],
)
def rebuild(payload: RebuildRequest) -> RebuildResponse:
    """
    Recharge les documents et reconstruit l'index vectoriel du RAG.

    Parameters
    ----------
    payload : RebuildRequest
        Corps de requête contenant la zone et le scope.

    Returns
    -------
    RebuildResponse
        Statut de l'opération et nombre de documents indexés.

    Raises
    ------
    HTTPException
        400 : aucun document trouvé ou paramètres invalides
        500 : erreur interne du serveur
    """
    try:
        zone = payload.zone or DEFAULT_ZONE
        scope = payload.scope or DEFAULT_SCOPE

        documents = load_documents(zone=zone, scope=scope, source="api")

        if not documents:
            raise ValueError("Aucun document n'a été chargé.")

        rag_service.set_documents(documents)
        n_docs = rag_service.build_index()

        return RebuildResponse(
            status="success",
            message="La base d'information a été reconstruite avec succès.",
            n_docs_indexed=n_docs,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception:
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")