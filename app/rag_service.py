"""
Service principal du pipeline RAG OpenAgenda.

Ce module orchestre le pipeline complet de recherche et de génération
de réponses à partir d'événements OpenAgenda indexés dans FAISS.

## Responsabilités principales

Cette classe pilote les grandes étapes du système :

- chargement ou reconstruction de l'index vectoriel global
- chargement du corpus documentaire source
- préfiltrage structuré des documents via `FilterService`
- fallback de préfiltrage lorsque les contraintes sont trop strictes
- recherche vectorielle sur le sous-corpus retenu
- injection du score vectoriel dans les métadonnées
- ranking métier via `RetrievalService`
- contrôle final de cohérence documentaire
- formatage du contexte pour le LLM
- génération de la réponse finale
- gestion d'une mémoire conversationnelle courte via `MemoryService`
- reformulation contextuelle des questions selon l'historique récent
- construction des objets API
- écriture des traces JSONL

## Philosophie du pipeline

Le service ne contient pas la logique métier fine du filtrage lexical
ou du scoring métier. Ces responsabilités restent déléguées à :

- `FilterService`
- `RetrievalService`

`RAGService` agit comme chef d'orchestre du pipeline. Il décide :

- quand lancer les différentes briques
- quand déclencher un fallback
- quels documents transmettre au ranking
- quels documents conserver pour la réponse finale

## Notes

Deux représentations textuelles coexistent volontairement :

- `search_text` :
  texte riche destiné à l'embedding et à la recherche vectorielle
- `format_doc()` :
  fiche courte et structurée destinée au LLM

Cette séparation permet :

- un retrieval plus riche sémantiquement
- un contexte plus compact et plus propre pour la génération

## Mémoire conversationnelle

La mémoire courte sert uniquement à :

- maintenir la cohérence conversationnelle entre plusieurs tours
- reformuler les questions elliptiques ou dépendantes du contexte
- améliorer la qualité du retrieval en produisant une question autonome

Les faits utilisés dans la réponse finale doivent toujours provenir
des documents récupérés dans le contexte, jamais de l'historique seul.
"""
from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import (
    DEFAULT_SCOPE,
    DEFAULT_ZONE,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    LLM_MODEL,
)
from app.document_service import load_documents
from app.filter_service import FilterService
from app.memory_service import MemoryService
from app.retrieval_service import RetrievalService
from app.schemas import AskResponse, RetrievedDocument
from app.trace_service import TraceService


class RAGService:
    """
    Service principal du pipeline RAG.

    Cette classe orchestre l'ensemble du flux de traitement :

    1. disponibilité des documents et de l'index
    2. préfiltrage structuré
    3. fallback éventuel si le filtrage est trop dur
    4. reformulation contextuelle via mémoire courte
    5. recherche vectorielle locale
    6. ranking métier
    7. contrôle final des documents
    8. génération de la réponse
    9. mise à jour de la mémoire conversationnelle
    10. traçage complet

    Le service ne porte pas la logique métier détaillée du filtrage
    ou du scoring. Il pilote uniquement le flux global.

    Les modèles d'embeddings et de génération sont configurables via
    des variables d'environnement afin de permettre un changement
    de provider sans modifier le code métier.
    """
    # Cette classe ne fait pas "tout toute seule".
    # Elle orchestre les briques spécialisées :
    # documents, filtre, retrieval, mémoire, génération, traces.

    EMPTY_ANSWER = "Je n'ai trouvé aucun événement correspondant dans les données disponibles."
    # Réponse standard lorsque le pipeline ne trouve aucun document exploitable.

    def __init__(
        self,
        documents: list[Document] | None = None,
        index_dir: str | Path | None = None,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL,
        top_k_retrieval: int = 10,
        top_k_final: int = 3,
        zone: str = DEFAULT_ZONE,
        scope: str = DEFAULT_SCOPE,
        trace_enabled: bool = True,
        trace_file: str | Path = "artifacts/rag_trace.jsonl",
    ) -> None:
        """
        Initialise le service RAG et ses dépendances principales.

        Parameters
        ----------
        documents : list[Document] | None
            Corpus documentaire déjà chargé en mémoire, si disponible.
        index_dir : str | Path | None
            Répertoire contenant l'index FAISS global.
        embedding_model : str
            Nom du modèle d'embeddings à utiliser.
            Peut provenir des variables d'environnement.
        llm_model : str
            Nom du modèle de génération / reformulation à utiliser.
            Peut provenir des variables d'environnement.
        top_k_retrieval : int
            Nombre maximal de documents candidats retenus après retrieval vectoriel.
        top_k_final : int
            Nombre maximal de documents conservés pour la réponse finale.
        zone : str
            Zone par défaut utilisée dans le préfiltrage.
        scope : str
            Niveau de portée géographique par défaut.
        trace_enabled : bool
            Active ou non l'écriture des traces JSONL.
        trace_file : str | Path
            Fichier de sortie des traces.
        """
        # Paramètres généraux du pipeline :
        # corpus, index, embeddings, LLM, zone, scope, tailles de top-k.
        self.documents = documents or []
        self.index_dir = Path(index_dir) if index_dir else FAISS_INDEX_DIR
        self.top_k_retrieval = top_k_retrieval
        self.top_k_final = top_k_final
        self.zone = zone
        self.scope = scope

        # On mémorise explicitement les noms de modèles afin de pouvoir
        # les tracer et les afficher en debug.
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model

        # Construction dynamique des composants selon le nom de modèle fourni.
        # Cela permet de changer de provider via l'environnement sans toucher
        # au code métier de la classe.
        self.embeddings = self._build_embeddings(embedding_model)
        self.llm = self._build_llm(llm_model)

        # Vectorstore FAISS global chargé en mémoire.
        self.vectorstore: FAISS | None = None

        # Briques spécialisées du pipeline.
        self.filter_service = FilterService()
        self.retrieval_service = RetrievalService()
        self.memory_service = MemoryService()
        self.trace_service = TraceService(
            trace_file=trace_file,
            enabled=trace_enabled,
        )

        # Prompt principal de génération.
        # Il force le modèle à rester strictement fidèle au contexte documentaire.
        self.prompt = ChatPromptTemplate.from_template(
            """
Tu es un assistant spécialisé dans les événements culturels.

Tu réponds uniquement à partir des documents fournis dans le contexte.
Tu ne dois jamais inventer d'événement.

L'historique récent sert uniquement à comprendre la continuité
de la conversation.
Les faits utilisés dans la réponse doivent toujours provenir
du contexte documentaire.

Consignes de réponse :
- Si aucun document pertinent n'est disponible, réponds exactement :
  "Je n'ai trouvé aucun événement correspondant dans les données disponibles."

- Les dates doivent toujours être sous la forme AAAA/MM/JJ.
- Les intervalles de dates doivent toujours être sous la forme AAAA/MM/JJ au AAAA/MM/JJ.
- Tu dois lister tous les événements présents dans le contexte final.
- Tu ne dois en oublier aucun.
- Chaque événement du contexte doit apparaître une seule fois dans la réponse.
- Le nombre de lignes de la liste doit correspondre au nombre d'événements présents dans le contexte.

- Si des documents pertinents sont disponibles, réponds toujours et uniquement sous la forme :
Liste d'événements pour {question} :
- Titre de l'événement (date : ..., lieu : ...)
- Titre de l'événement (date : ..., lieu : ...)

Tu ne dois pas ajouter d'introduction, de conclusion, ni de commentaire.
Tu dois rester strictement fidèle au contexte.
Tu ne dois jamais mentionner d'événement absent du contexte.

Historique récent :
{history}

Question :
{question}

Contexte :
{context}
""".strip()
        )

        # Prompt de reformulation.
        # Il sert à transformer une question dépendante de l'historique
        # en question autonome avant le retrieval.
        self.rewrite_prompt = ChatPromptTemplate.from_template(
            """
Tu reçois :
- un historique récent de conversation
- une nouvelle question utilisateur

Ta mission :
- reformuler la nouvelle question pour qu'elle soit autonome
- conserver strictement l'intention de l'utilisateur
- ne rien inventer
- si la question est déjà autonome, retourne-la telle quelle ou presque

Historique :
{history}

Nouvelle question :
{question}

Question reformulée :
""".strip()
        )

        # Chaîne principale de génération de réponse.
        self.chain = self.prompt | self.llm | StrOutputParser()

        # Chaîne secondaire de reformulation de question.
        self.rewrite_chain = self.rewrite_prompt | self.llm | StrOutputParser()

    # ------------------------------------------------------------------
    # Construction dynamique des modèles
    # ------------------------------------------------------------------

    def _build_embeddings(self, model_name: str) -> Embeddings:
        """
        Construit l'objet d'embeddings adapté au modèle demandé.

        Parameters
        ----------
        model_name : str
            Nom du modèle d'embeddings.

        Returns
        -------
        Embeddings
            Instance d'embeddings compatible LangChain.

        Raises
        ------
        ValueError
            Si le modèle demandé n'est pas reconnu ou non supporté.
        """
        # On normalise le nom pour simplifier la détection du provider.
        normalized = model_name.strip().lower()

        # Cas Mistral.
        if normalized.startswith("mistral"):
            return MistralAIEmbeddings(model=model_name)

        # Cas OpenAI embeddings.
        # On accepte à la fois les noms explicites type "text-embedding-3-small"
        # et une éventuelle convention interne commençant par "openai".
        if "text-embedding" in normalized or normalized.startswith("openai"):
            return OpenAIEmbeddings(model=model_name)

        raise ValueError(
            f"Embedding model non supporté : {model_name}. "
            "Valeurs attendues : ex. 'mistral-embed' ou 'text-embedding-3-small'."
        )

    def _build_llm(self, model_name: str) -> BaseChatModel:
        """
        Construit le LLM adapté au modèle demandé.

        Parameters
        ----------
        model_name : str
            Nom du modèle conversationnel / génératif.

        Returns
        -------
        BaseChatModel
            Instance de modèle de chat compatible LangChain.

        Raises
        ------
        ValueError
            Si le modèle demandé n'est pas reconnu ou non supporté.
        """
        # Même logique que pour les embeddings :
        # on choisit la bonne classe selon le provider détecté.
        normalized = model_name.strip().lower()

        # Cas Mistral.
        if normalized.startswith("mistral"):
            return ChatMistralAI(
                model=model_name,
                temperature=0.2,
            )

        # Cas OpenAI.
        if normalized.startswith("gpt") or normalized.startswith("openai"):
            return ChatOpenAI(
                model=model_name,
                temperature=0.2,
            )

        raise ValueError(
            f"LLM model non supporté : {model_name}. "
            "Valeurs attendues : ex. 'mistral-small-latest' ou 'gpt-4o-mini'."
        )

    # ------------------------------------------------------------------
    # Gestion de l'index et des documents
    # ------------------------------------------------------------------

    def is_index_loaded(self) -> bool:
        """
        Indique si le vectorstore FAISS global est déjà chargé en mémoire.
        """
        return self.vectorstore is not None

    def set_documents(self, documents: list[Document]) -> None:
        """
        Injecte explicitement une nouvelle liste de documents dans le service.
        """
        self.documents = documents

    def _ensure_documents_loaded(self) -> None:
        """
        Charge les documents si le corpus mémoire est vide.
        """
        # Évite de recharger inutilement si le corpus est déjà présent.
        if self.documents:
            return

        self.documents = load_documents(
            zone=self.zone,
            scope=self.scope,
        )

    def ensure_index_ready(self) -> None:
        """
        S'assure que l'index FAISS et le corpus documentaire sont prêts.

        Stratégie :
        - si le vectorstore est déjà chargé, on complète juste les documents
        - sinon, on tente de charger l'index depuis le disque
        - sinon, on construit un nouvel index à partir des documents
        """
        # Fonction d'initialisation défensive du pipeline.
        if self.vectorstore is not None:
            self._ensure_documents_loaded()
            return

        if self.index_dir.exists() and any(self.index_dir.iterdir()):
            self.load_index()
            self._ensure_documents_loaded()
            return

        if not self.documents:
            self._ensure_documents_loaded()

        if not self.documents:
            raise ValueError(
                "Aucun index FAISS disponible et aucun document fourni pour le construire."
            )

        self.build_index()

    def _build_docs_for_vector_index(self, docs: list[Document]) -> list[Document]:
        """
        Prépare les documents à injecter dans l'index vectoriel.

        Le `search_text` est privilégié car il est plus riche pour l'embedding
        que le `page_content` destiné au LLM.
        """
        # On construit des documents dédiés à l'indexation vectorielle,
        # distincts du format compact utilisé pour le LLM.
        docs_for_index: list[Document] = []

        for doc in docs:
            md = dict(doc.metadata or {})
            search_text = md.get("search_text", "") or doc.page_content

            docs_for_index.append(
                Document(
                    page_content=search_text,
                    metadata=md,
                )
            )

        return docs_for_index

    def _execute_with_retry(
        self,
        func: Callable[[], Any],
        max_retries: int = 5,
    ) -> Any:
        """
        Exécute une fonction avec retry exponentiel sur erreur 429.
        """
        # Utile pour les appels liés aux embeddings
        # lorsqu'une limitation de débit survient.
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as exc:
                message = str(exc)
                if "429" in message and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                raise

    def build_index(self) -> int:
        """
        Construit l'index FAISS global à partir du corpus documentaire.

        Returns
        -------
        int
            Nombre de documents indexés.
        """
        # Construction par batch pour limiter la pression
        # sur les embeddings et mieux gérer les retries.
        if not self.documents:
            raise ValueError(
                "Impossible de construire l'index : aucun document n'est disponible."
            )

        docs_for_index = self._build_docs_for_vector_index(self.documents)

        batch_size = 32
        pause_seconds = 0.3
        vectorstore: FAISS | None = None

        for i in range(0, len(docs_for_index), batch_size):
            batch = docs_for_index[i : i + batch_size]

            if vectorstore is None:
                vectorstore = self._execute_with_retry(
                    lambda: FAISS.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                    )
                )
            else:
                self._execute_with_retry(
                    lambda: vectorstore.add_documents(batch)
                )

            time.sleep(pause_seconds)

        self.vectorstore = vectorstore

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.index_dir))

        return len(self.documents)

    def load_index(self) -> None:
        """
        Charge l'index FAISS depuis le disque.
        """
        # Recharge un index existant au démarrage
        # sans devoir recalculer les embeddings.
        if not self.index_dir.exists() or not any(self.index_dir.iterdir()):
            raise FileNotFoundError(
                f"Index FAISS introuvable dans le dossier : {self.index_dir}"
            )

        self.vectorstore = FAISS.load_local(
            folder_path=str(self.index_dir),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def rebuild_index(self, documents: list[Document]) -> int:
        """
        Remplace le corpus courant puis reconstruit complètement l'index.
        """
        # Utile après un refresh documentaire complet
        # ou après un changement de modèle d'embeddings.
        self.set_documents(documents)
        return self.build_index()

    def _build_local_vectorstore(self, docs: list[Document]) -> FAISS:
        """
        Construit un FAISS local temporaire sur un sous-corpus filtré.
        """
        # C'est un choix important de ton pipeline :
        # au lieu d'interroger l'index global puis filtrer,
        # tu recrées localement un petit index sur le sous-corpus retenu.
        docs_for_index = self._build_docs_for_vector_index(docs)

        return FAISS.from_documents(
            documents=docs_for_index,
            embedding=self.embeddings,
        )

    # ------------------------------------------------------------------
    # Mémoire conversationnelle
    # ------------------------------------------------------------------

    def _get_recent_history_text(
        self,
        session_id: str,
        max_messages: int | None = None,
    ) -> str:
        """
        Retourne l'historique récent formaté en texte pour le prompt.
        """
        return self.memory_service.format_history_for_prompt(
            session_id=session_id,
            max_messages=max_messages,
        ).strip()

    def _get_recent_history_messages(
        self,
        session_id: str,
        max_messages: int | None = None,
    ) -> list[dict[str, str]]:
        """
        Retourne l'historique récent sous forme structurée.
        """
        return self.memory_service.get_recent_messages(
            session_id=session_id,
            max_messages=max_messages,
        )

    def _rewrite_question_with_history(self, question: str, session_id: str) -> str:
        """
        Reformule la question à partir de l'historique récent si nécessaire.
        """
        # Permet de transformer :
        # "et à Sète ?" ou "et pour les enfants ?"
        # en question autonome avant le retrieval.
        history = self._get_recent_history_text(
            session_id=session_id,
            max_messages=6,
        )

        if not history:
            return question.strip()

        rewritten = self.rewrite_chain.invoke(
            {
                "history": history,
                "question": question,
            }
        ).strip()

        return rewritten or question.strip()

    # ------------------------------------------------------------------
    # Sérialisation / debug
    # ------------------------------------------------------------------

    def _serialize_for_json(self, value: Any) -> Any:
        """
        Convertit des objets potentiellement non sérialisables en JSON.
        """
        # Sert surtout pour les traces et les réponses debug.
        if isinstance(value, (date, datetime)):
            return value.isoformat()

        if isinstance(value, dict):
            return {
                str(key): self._serialize_for_json(val)
                for key, val in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [self._serialize_for_json(item) for item in value]

        return value

    def _sanitize_filter_debug(self, debug_data: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Réduit les informations de debug filtre à un format compact et sérialisable.
        """
        # Évite de tracer des objets trop lourds ou non JSON-compatibles.
        if debug_data is None:
            return None

        return {
            "filters": self._serialize_for_json(debug_data.get("filters", {})),
            "n_input_docs": debug_data.get("n_input_docs", 0),
            "n_after_city": debug_data.get("n_after_city", 0),
            "n_after_date": debug_data.get("n_after_date", 0),
            "n_after_type": debug_data.get("n_after_type", 0),
            "n_after_music": debug_data.get("n_after_music", 0),
            "n_after_cultural": debug_data.get("n_after_cultural", 0),
            "n_after_audience": debug_data.get("n_after_audience", 0),
            "n_after_duration": debug_data.get("n_after_duration", 0),
            "n_after_price": debug_data.get("n_after_price", 0),
        }

    # ------------------------------------------------------------------
    # Helpers de trace
    # ------------------------------------------------------------------

    def _doc_to_trace_row(self, doc: Document) -> dict[str, Any]:
        """
        Transforme un document en ligne légère de trace.
        """
        md = doc.metadata or {}

        return {
            "title": md.get("title", ""),
            "location_name": md.get("location_name", ""),
            "city": md.get("city", ""),
            "region": md.get("region", ""),
            "first_date": md.get("first_date", ""),
            "last_date": md.get("last_date", ""),
            "event_type": md.get("canonical_event_type", "") or md.get("event_type", ""),
            "music_genre": md.get("music_genre", ""),
            "price_info": md.get("price_info", ""),
            "is_free": md.get("is_free"),
            "vector_score": md.get("vector_score"),
            "final_score": md.get("final_score"),
            "diversified_score": md.get("diversified_score"),
            "url": md.get("source_url", "") or md.get("url", ""),
        }

    def _trace_pipeline(
        self,
        *,
        question: str,
        effective_question: str,
        session_id: str,
        history: list[dict[str, str]],
        filter_debug: dict[str, Any],
        fallback_filter_debug: dict[str, Any] | None,
        prefiltered_docs: list[Document],
        raw_docs: list[Document],
        ranked_docs: list[Document],
        final_docs: list[Document],
        context: str,
        answer: str,
        fallback_used: bool,
    ) -> None:
        """
        Enregistre une trace complète du pipeline pour analyse ultérieure.
        """
        # Cette trace sert à comprendre précisément
        # ce qui s'est passé à chaque étape du pipeline.
        payload = {
            "question": question,
            "effective_question": effective_question,
            "session_id": session_id,
            "history": history,
            "zone": self.zone,
            "scope": self.scope,
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model_name,
            "top_k_retrieval": self.top_k_retrieval,
            "top_k_final": self.top_k_final,
            "fallback_used": fallback_used,
            "n_input_docs": len(self.documents),
            "n_prefiltered_docs": len(prefiltered_docs),
            "n_raw_docs": len(raw_docs),
            "n_ranked_docs": len(ranked_docs),
            "n_final_docs": len(final_docs),
            "filter_debug": self._sanitize_filter_debug(filter_debug),
            "fallback_filter_debug": self._sanitize_filter_debug(fallback_filter_debug),
            "prefiltered_docs": [self._doc_to_trace_row(doc) for doc in prefiltered_docs[:50]],
            "raw_docs": [self._doc_to_trace_row(doc) for doc in raw_docs[:50]],
            "ranked_docs": [self._doc_to_trace_row(doc) for doc in ranked_docs[:50]],
            "final_docs": [self._doc_to_trace_row(doc) for doc in final_docs[:50]],
            "context": context,
            "answer": answer,
        }

        self.trace_service.write_trace(self._serialize_for_json(payload))

    # ------------------------------------------------------------------
    # Retrieval interne
    # ------------------------------------------------------------------

    def _attach_vector_scores(
        self,
        results: list[tuple[Document, float]],
    ) -> list[Document]:
        """
        Injecte les scores vectoriels dans les métadonnées des documents.
        """
        # Permet au RetrievalService de réutiliser ensuite
        # le signal vectoriel dans son score hybride.
        docs: list[Document] = []

        for doc, score in results:
            if doc.metadata is None:
                doc.metadata = {}

            doc.metadata["vector_score"] = float(score)
            docs.append(doc)

        return docs

    def _is_reliable_document(self, doc: Document) -> bool:
        """
        Vérifie qu'un document semble suffisamment fiable pour la réponse finale.
        """
        # Sert de garde-fou final :
        # on évite de répondre avec des documents trop incomplets.
        md = doc.metadata or {}

        has_title = bool(md.get("title"))
        has_date = bool(md.get("first_date"))
        has_location = bool(md.get("location_name") or md.get("city"))
        has_url = bool(md.get("source_url") or md.get("url"))
        content_quality = md.get("content_quality", 0)

        try:
            content_quality = int(content_quality)
        except (TypeError, ValueError):
            content_quality = 0

        return has_title and has_date and has_location and (has_url or content_quality >= 4)

    def _post_filter_ranked_docs(self, docs: list[Document]) -> list[Document]:
        """
        Retient les documents fiables après ranking.
        """
        # Dernier tamis avant génération.
        if not docs:
            return []

        reliable_docs = [doc for doc in docs if self._is_reliable_document(doc)]

        if reliable_docs:
            return reliable_docs[: self.top_k_final]

        return docs[: self.top_k_final]

    def _run_vector_retrieval(self, question: str, candidate_docs: list[Document]) -> list[Document]:
        """
        Lance la recherche vectorielle sur le sous-corpus préfiltré.
        """
        # Point clé de ton architecture :
        # retrieval local sur le sous-corpus candidat, pas sur tout l'index global.
        if not candidate_docs:
            return []

        local_vectorstore = self._build_local_vectorstore(candidate_docs)

        raw_results = local_vectorstore.similarity_search_with_score(
            query=question,
            k=min(self.top_k_retrieval, len(candidate_docs)),
        )

        if not raw_results:
            return []

        return self._attach_vector_scores(raw_results)

    def _build_relaxed_question(self, question: str) -> str:
        """
        Construit une version relâchée de la question pour le fallback.

        Actuellement, la version relâchée est identique à la question.
        """
        # Point d'extension futur :
        # ici on pourrait simplifier la requête pour un fallback plus sophistiqué.
        return question.strip()

    def _prefilter_with_fallback(
        self,
        question: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, list[Document], bool]:
        """
        Préfiltre les documents avec une stratégie de fallback progressive.

        Retourne `fallback_used=True` uniquement si le fallback a réellement
        permis de récupérer des documents.
        """
        # Étape importante du pipeline :
        # on tente d'abord un préfiltrage strict avec ville par défaut,
        # puis un fallback plus souple si cela vide tout.
        filter_debug = self.filter_service.filter_documents_with_debug(
            question=question,
            docs=self.documents,
            default_city=self.zone,
        )

        prefiltered_docs = filter_debug.get("docs", [])
        if prefiltered_docs:
            return filter_debug, None, prefiltered_docs, False

        fallback_question = self._build_relaxed_question(question)
        fallback_filter_debug = self.filter_service.filter_documents_with_debug(
            question=fallback_question,
            docs=self.documents,
            default_city=None,
        )

        fallback_docs = fallback_filter_debug.get("docs", [])
        if fallback_docs:
            return filter_debug, fallback_filter_debug, fallback_docs, True

        return filter_debug, fallback_filter_debug, [], False

    # ------------------------------------------------------------------
    # API retrieval
    # ------------------------------------------------------------------

    def retrieve(self, question: str, session_id: str = "default") -> list[Document]:
        """
        Retourne directement les documents finaux retenus par le pipeline.
        """
        # API simple orientée retrieval seul.
        if not question or not question.strip():
            raise ValueError("La question utilisateur ne peut pas être vide.")

        result = self._run_pipeline(question, session_id=session_id)
        return result["docs"]

    # ------------------------------------------------------------------
    # Formatage du contexte
    # ------------------------------------------------------------------

    def format_doc(self, doc: Document, index: int | None = None) -> str:
        """
        Formate un document en fiche courte lisible par le LLM.
        """
        # Version compacte et structurée du document,
        # distincte du search_text utilisé pour le retrieval.
        md = doc.metadata or {}

        lines: list[str] = []

        if index is not None:
            lines.append(f"Événement {index}")

        description = (md.get("description", "") or "")[:500]

        lines.extend(
            [
                f"Titre : {md.get('title', '')}",
                f"Lieu : {md.get('location_name', '')}",
                f"Ville : {md.get('city', '')}",
                f"Région : {md.get('region', '')}",
                f"Date de début : {md.get('first_date', '')}",
                f"Date de fin : {md.get('last_date', '')}",
                f"Type : {md.get('canonical_event_type', '') or md.get('event_type', '')}",
                f"Genre musical : {md.get('music_genre', '')}",
                f"Tarification : {md.get('price_info', '')}",
                f"Description : {description}",
                f"URL : {md.get('source_url', '') or md.get('url', '')}",
            ]
        )

        return "\n".join(lines)

    def format_docs(self, docs: list[Document]) -> str:
        """
        Concatène plusieurs documents formatés en un seul contexte texte.
        """
        if not docs:
            return ""

        return "\n\n".join(
            self.format_doc(doc, idx)
            for idx, doc in enumerate(docs, start=1)
        )

    def format_docs_list(self, docs: list[Document]) -> list[str]:
        """
        Retourne la liste des fiches formatées, une par document.
        """
        if not docs:
            return []

        return [
            self.format_doc(doc, idx)
            for idx, doc in enumerate(docs, start=1)
        ]

    # ------------------------------------------------------------------
    # Conversion API
    # ------------------------------------------------------------------

    def _build_retrieved_document(self, doc: Document) -> RetrievedDocument:
        """
        Convertit un Document LangChain en schéma API RetrievedDocument.
        """
        md = doc.metadata or {}

        raw_score = md.get("final_score", md.get("vector_score"))
        try:
            score = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError):
            score = None

        return RetrievedDocument(
            title=md.get("title", ""),
            location_name=md.get("location_name", ""),
            city=md.get("city", ""),
            region=md.get("region", ""),
            first_date=md.get("first_date", ""),
            last_date=md.get("last_date", ""),
            event_type=md.get("canonical_event_type", "") or md.get("event_type", ""),
            url=md.get("source_url", "") or md.get("url", ""),
            price_info=md.get("price_info", ""),
            is_free=md.get("is_free"),
            keywords_title=md.get("keywords_title", []),
            score=score,
        )

    def build_retrieved_documents(
        self,
        docs: list[Document],
    ) -> list[RetrievedDocument]:
        """
        Convertit une liste de Documents en objets API.
        """
        return [self._build_retrieved_document(doc) for doc in docs]

    # ------------------------------------------------------------------
    # Génération
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        docs: list[Document],
        session_id: str = "default",
    ) -> str:
        """
        Génère la réponse finale du LLM à partir du contexte documentaire.
        """
        # Si aucun document exploitable n'est présent,
        # on renvoie directement la réponse vide standard.
        if not docs:
            return self.EMPTY_ANSWER

        context = self.format_docs(docs).strip()
        if not context:
            return self.EMPTY_ANSWER

        history = self._get_recent_history_text(
            session_id=session_id,
            max_messages=6,
        )

        answer = self.chain.invoke(
            {
                "question": question,
                "context": context,
                "history": history,
            }
        )
        answer = answer.strip()

        return answer or self.EMPTY_ANSWER

    # ------------------------------------------------------------------
    # Pipeline complet
    # ------------------------------------------------------------------

    def _run_pipeline(self, question: str, session_id: str = "default") -> dict[str, Any]:
        """
        Exécute le pipeline complet de bout en bout.
        """
        # Cœur orchestral du service.
        # C'est ici que toutes les briques s'enchaînent dans le bon ordre.
        if not question or not question.strip():
            raise ValueError("La question utilisateur ne peut pas être vide.")

        self.ensure_index_ready()

        # Reformulation contextuelle si nécessaire.
        effective_question = self._rewrite_question_with_history(question, session_id)

        # Préfiltrage structuré + fallback éventuel.
        filter_debug, fallback_filter_debug, prefiltered_docs, fallback_used = (
            self._prefilter_with_fallback(effective_question)
        )

        raw_docs: list[Document] = []
        ranked_docs: list[Document] = []
        final_docs: list[Document] = []

        if prefiltered_docs:
            # Recherche vectorielle locale sur le sous-corpus retenu.
            raw_docs = self._run_vector_retrieval(
                question=effective_question,
                candidate_docs=prefiltered_docs,
            )

            if raw_docs:
                # Ranking métier hybride sur les documents candidats.
                ranked_docs = self.retrieval_service.rank_documents(
                    question=effective_question,
                    raw_docs=raw_docs,
                    top_k=min(self.top_k_retrieval, len(raw_docs)),
                )

                # Contrôle final de fiabilité documentaire.
                final_docs = self._post_filter_ranked_docs(ranked_docs)

        # Construction du contexte, génération, conversion API.
        context = self.format_docs(final_docs)
        answer = self.generate_answer(question, final_docs, session_id)
        retrieved_documents = self.build_retrieved_documents(final_docs)
        retrieved_contexts = self.format_docs_list(final_docs)

        # Variante détaillée du ranking pour le debug.
        retrieval_debug = self.retrieval_service.rank_documents_with_scores(
            question=effective_question,
            raw_docs=raw_docs,
            top_k=len(raw_docs) if raw_docs else self.top_k_final,
        )

        # Mise à jour de la mémoire courte après réponse.
        self.memory_service.append_turn(
            session_id=session_id,
            user_message=question,
            assistant_message=answer,
        )

        history = self._get_recent_history_messages(
            session_id=session_id,
            max_messages=6,
        )

        # Écriture de la trace complète.
        self._trace_pipeline(
            question=question,
            effective_question=effective_question,
            session_id=session_id,
            history=history,
            filter_debug=filter_debug,
            fallback_filter_debug=fallback_filter_debug,
            prefiltered_docs=prefiltered_docs,
            raw_docs=raw_docs,
            ranked_docs=ranked_docs,
            final_docs=final_docs,
            context=context,
            answer=answer,
            fallback_used=fallback_used,
        )

        return {
            "question": question,
            "effective_question": effective_question,
            "session_id": session_id,
            "history": history,
            "answer": answer,
            "docs": final_docs,
            "retrieved_documents": retrieved_documents,
            "retrieved_contexts": retrieved_contexts,
            "filter_debug": self._sanitize_filter_debug(filter_debug),
            "fallback_filter_debug": self._sanitize_filter_debug(fallback_filter_debug),
            "fallback_used": fallback_used,
            "retrieval_debug": self._serialize_for_json(retrieval_debug),
        }

    def ask(self, question: str, session_id: str = "default") -> AskResponse:
        """
        API principale orientée production.

        Retourne la réponse finale typée sous forme d'AskResponse.
        """
        result = self._run_pipeline(question, session_id=session_id)

        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            n_docs=len(result["docs"]),
            documents=result["retrieved_documents"],
            session_id=result["session_id"],
        )

    def ask_debug(self, question: str, session_id: str = "default") -> dict[str, Any]:
        """
        Variante debug de l'API ask.

        Retourne la réponse ainsi que les états intermédiaires utiles
        à l'analyse du pipeline.
        """
        result = self._run_pipeline(question, session_id=session_id)

        retrieved_documents = result.get("retrieved_documents", [])
        retrieved_contexts = result.get("retrieved_contexts", [])
        retrieval_debug = result.get("retrieval_debug", [])
        filter_debug = result.get("filter_debug", {})
        fallback_filter_debug = result.get("fallback_filter_debug")
        fallback_used = result.get("fallback_used", False)

        response = {
            "question": result.get("question", question),
            "effective_question": result.get("effective_question", question),
            "session_id": result.get("session_id", session_id),
            "history": result.get("history", []),
            "answer": result.get("answer", ""),
            "n_docs": len(retrieved_documents),
            "documents": [
                doc.model_dump() if hasattr(doc, "model_dump") else doc.dict()
                for doc in retrieved_documents
            ],
            "retrieved_contexts": retrieved_contexts if isinstance(retrieved_contexts, list) else [],
            "fallback_used": fallback_used,
            "filter_debug": filter_debug if isinstance(filter_debug, dict) else {},
            "retrieval_debug": retrieval_debug if isinstance(retrieval_debug, list) else [],
        }

        if fallback_filter_debug is not None:
            response["fallback_filter_debug"] = (
                fallback_filter_debug if isinstance(fallback_filter_debug, dict) else {}
            )

        return self._serialize_for_json(response)