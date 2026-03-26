"""
Service principal du pipeline RAG OpenAgenda.

Ce module orchestre le pipeline complet de recherche et de génération
de réponses à partir d'événements OpenAgenda indexés dans FAISS.

Responsabilités principales
---------------------------
Cette classe pilote les grandes étapes du système :

- chargement ou reconstruction de l'index vectoriel global
- chargement du corpus documentaire source
- préfiltrage structuré des documents via FilterService
- fallback de préfiltrage lorsque les contraintes sont trop strictes
- recherche vectorielle sur le sous-corpus retenu
- injection du score vectoriel dans les métadonnées
- ranking métier via RetrievalService
- contrôle final de cohérence documentaire
- formatage du contexte pour le LLM
- génération de la réponse finale
- gestion d'une mémoire conversationnelle courte via MemoryService
- reformulation contextuelle des questions selon l'historique récent
- construction des objets API
- écriture des traces JSONL

Philosophie du pipeline
-----------------------
Le service ne contient pas la logique métier fine du filtrage lexical
ou du scoring métier. Ces responsabilités restent déléguées à :

- FilterService
- RetrievalService

RAGService agit comme chef d'orchestre du pipeline. Il décide :
- quand lancer les différentes briques
- quand déclencher un fallback
- quels documents transmettre au ranking
- quels documents conserver pour la réponse finale

Notes
-----
Deux représentations textuelles coexistent volontairement :

- ``search_text`` :
  texte riche destiné à l'embedding et à la recherche vectorielle
- ``format_doc()`` :
  fiche courte et structurée destinée au LLM

Cette séparation permet :
- un retrieval plus riche sémantiquement
- un contexte plus compact et plus propre pour la génération
"""

from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from app.config import DEFAULT_SCOPE, DEFAULT_ZONE, FAISS_INDEX_DIR
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
    9. traçage complet du pipeline

    Le service ne porte pas la logique métier détaillée du filtrage
    ou du scoring. Il pilote uniquement le flux global.
    """

    EMPTY_ANSWER = "Je n'ai trouvé aucun événement correspondant dans les données disponibles."

    def __init__(
        self,
        documents: list[Document] | None = None,
        index_dir: str | Path | None = None,
        embedding_model: str = "mistral-embed",
        llm_model: str = "mistral-small-latest",
        top_k_retrieval: int = 10,
        top_k_final: int = 3,
        zone: str = DEFAULT_ZONE,
        scope: str = DEFAULT_SCOPE,
        trace_enabled: bool = True,
        trace_file: str | Path = "artifacts/rag_trace.jsonl",
    ) -> None:
        """
        Initialise le service RAG.

        Parameters
        ----------
        documents : list[Document] | None, default=None
            Corpus documentaire éventuellement déjà chargé.
        index_dir : str | Path | None, default=None
            Dossier local de stockage de l'index FAISS.
        embedding_model : str, default="mistral-embed"
            Modèle d'embedding utilisé.
        llm_model : str, default="mistral-small-latest"
            Modèle de génération utilisé.
        top_k_retrieval : int, default=10
            Nombre maximal de documents récupérés après recherche vectorielle.
        top_k_final : int, default=3
            Nombre maximal de documents conservés pour la réponse finale.
        zone : str, default=DEFAULT_ZONE
            Zone géographique principale du corpus.
        scope : str, default=DEFAULT_SCOPE
            Scope géographique principal du corpus.
        trace_enabled : bool, default=True
            Active l'écriture des traces JSONL.
        trace_file : str | Path, default="artifacts/rag_trace.jsonl"
            Fichier cible pour les traces.
        """
        self.documents = documents or []
        self.index_dir = Path(index_dir) if index_dir else FAISS_INDEX_DIR
        self.top_k_retrieval = top_k_retrieval
        self.top_k_final = top_k_final
        self.zone = zone
        self.scope = scope

        self.embeddings = MistralAIEmbeddings(model=embedding_model)
        self.llm = ChatMistralAI(
            model=llm_model,
            temperature=0.2,
        )

        self.vectorstore: FAISS | None = None

        self.filter_service = FilterService()
        self.retrieval_service = RetrievalService()
        self.memory_service = MemoryService()
        self.trace_service = TraceService(
            trace_file=trace_file,
            enabled=trace_enabled,
        )

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

        self.chain = self.prompt | self.llm | StrOutputParser()
        self.rewrite_chain = self.rewrite_prompt | self.llm | StrOutputParser()

    # ------------------------------------------------------------------
    # Gestion de l'index et des documents
    # ------------------------------------------------------------------

    def is_index_loaded(self) -> bool:
        """
        Indique si l'index vectoriel global est chargé en mémoire.
        """
        return self.vectorstore is not None

    def set_documents(self, documents: list[Document]) -> None:
        """
        Remplace les documents actuellement stockés dans le service.
        """
        self.documents = documents

    def _ensure_documents_loaded(self) -> None:
        """
        Recharge le corpus documentaire source si nécessaire.

        Même si un index FAISS existe déjà, le corpus source reste utile,
        car le pipeline applique ensuite un préfiltrage structuré avant
        de lancer une recherche vectorielle locale.
        """
        if self.documents:
            return

        self.documents = load_documents(
            zone=self.zone,
            scope=self.scope,
        )

    def ensure_index_ready(self) -> None:
        """
        S'assure que le corpus documentaire et l'index FAISS sont disponibles.

        Comportement :
        - si l'index est déjà chargé en mémoire, rien à faire
        - si un index existe sur disque, il est chargé
        - sinon, il est reconstruit à partir des documents disponibles
        """
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
        Construit une version des documents adaptée à l'indexation vectorielle.

        L'index vectoriel travaille de préférence sur `search_text`,
        plus riche que le contenu court destiné au LLM.
        """
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

    def _execute_with_retry(self, func, max_retries: int = 5):
        """
        Exécute une fonction avec retry simple en cas d'erreur 429.
        """
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
        """
        if not self.documents:
            raise ValueError(
                "Impossible de construire l'index : aucun document n'est disponible."
            )

        docs_for_index = self._build_docs_for_vector_index(self.documents)

        batch_size = 32
        pause_seconds = 0.3
        vectorstore = None

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
        Charge un index FAISS existant depuis le disque.
        """
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
        Reconstruit complètement l'index FAISS global.
        """
        self.set_documents(documents)
        return self.build_index()

    def _build_local_vectorstore(self, docs: list[Document]) -> FAISS:
        """
        Construit un index FAISS local temporaire sur un sous-ensemble
        de documents préfiltrés.
        """
        docs_for_index = self._build_docs_for_vector_index(docs)

        return FAISS.from_documents(
            documents=docs_for_index,
            embedding=self.embeddings,
        )

    # ------------------------------------------------------------------
    # Mémoire conversationnelle
    # ------------------------------------------------------------------

    def _rewrite_question_with_history(self, question: str, session_id: str) -> str:
        """
        Reformule la question avec l'historique court de la session
        afin d'améliorer la cohérence conversationnelle.
        """
        history = self.memory_service.format_history_for_prompt(session_id).strip()

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
        Convertit récursivement une structure Python en structure JSON-safe.

        Utile pour :
        - les traces JSONL
        - les réponses debug
        - la sérialisation des objets date / datetime
        """
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
        Nettoie une structure de debug issue du FilterService pour la rendre
        stable et sérialisable.
        """
        if debug_data is None:
            return None

        return {
            "filters": self._serialize_for_json(debug_data.get("filters", {})),
            "n_input_docs": debug_data.get("n_input_docs", 0),
            "n_after_city": debug_data.get("n_after_city", 0),
            "n_after_type": debug_data.get("n_after_type", 0),
            "n_after_music": debug_data.get("n_after_music", 0),
            "n_after_cultural": debug_data.get("n_after_cultural", 0),
            "n_after_audience": debug_data.get("n_after_audience", 0),
            "n_after_duration": debug_data.get("n_after_duration", 0),
            "n_after_date": debug_data.get("n_after_date", 0),
            "n_after_price": debug_data.get("n_after_price", 0),
        }

    # ------------------------------------------------------------------
    # Helpers de trace
    # ------------------------------------------------------------------

    def _doc_to_trace_row(self, doc: Document) -> dict[str, Any]:
        """
        Convertit un document en structure légère pour le traçage JSONL.
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
        Écrit une trace JSONL complète du pipeline.

        La trace permet de rejouer mentalement le pipeline :
        corpus d'entrée → préfiltrage → retrieval → ranking → réponse.
        """
        payload = {
            "question": question,
            "effective_question": effective_question,
            "session_id": session_id,
            "history": history,
            "zone": self.zone,
            "scope": self.scope,
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
        Injecte le score vectoriel FAISS dans les métadonnées des documents.
        """
        docs: list[Document] = []

        for doc, score in results:
            if doc.metadata is None:
                doc.metadata = {}

            doc.metadata["vector_score"] = float(score)
            docs.append(doc)

        return docs

    def _is_reliable_document(self, doc: Document) -> bool:
        """
        Vérifie qu'un document possède un minimum d'information utile.

        Ce garde-fou évite d'envoyer au LLM des documents trop pauvres,
        même s'ils ont survécu au ranking.
        """
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
        Applique un dernier contrôle léger après le ranking métier.

        On privilégie les documents réellement exploitables par le LLM.
        """
        if not docs:
            return []

        reliable_docs = [doc for doc in docs if self._is_reliable_document(doc)]

        if reliable_docs:
            return reliable_docs[: self.top_k_final]

        return docs[: self.top_k_final]

    def _run_vector_retrieval(self, question: str, candidate_docs: list[Document]) -> list[Document]:
        """
        Lance la recherche vectorielle sur un sous-corpus préfiltré.

        Returns
        -------
        list[Document]
            Documents enrichis avec leur score vectoriel.
        """
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
        Construit une version plus souple de la question pour un fallback.
        """
        return question.strip()

    def _prefilter_with_fallback(
        self,
        question: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, list[Document], bool]:
        """
        Préfiltre les documents avec une stratégie de fallback progressive.

        Stratégie :
        1. filtrage normal avec ville par défaut
        2. si aucun document, second essai sans ville par défaut
        3. si aucun document encore, on garde le résultat vide

        Returns
        -------
        tuple
            (
                filter_debug_principal,
                filter_debug_fallback_ou_none,
                docs_prefiltres,
                fallback_used
            )
        """
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

        return filter_debug, fallback_filter_debug, [], True

    # ------------------------------------------------------------------
    # API retrieval
    # ------------------------------------------------------------------

    def retrieve(self, question: str, session_id: str = "default") -> list[Document]:
        """
        Récupère les documents finaux les plus pertinents pour une question.

        Ce helper expose uniquement les documents retenus après :
        préfiltrage, recherche vectorielle, ranking métier et contrôle final.
        """
        if not question or not question.strip():
            raise ValueError("La question utilisateur ne peut pas être vide.")

        result = self._run_pipeline(question, session_id=session_id)
        return result["docs"]

    # ------------------------------------------------------------------
    # Formatage du contexte
    # ------------------------------------------------------------------

    def format_doc(self, doc: Document, index: int | None = None) -> str:
        """
        Formate un document unique sous forme de fiche courte pour le LLM.

        Le contexte transmis au modèle doit rester :
        - court
        - factuel
        - lisible
        - centré sur les informations réellement utiles à la réponse
        """
        md = doc.metadata or {}

        lines: list[str] = []

        if index is not None:
            lines.append(f"Événement {index}")

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
                f"Description : {md.get('description', '')}",
                f"URL : {md.get('source_url', '') or md.get('url', '')}",
            ]
        )

        return "\n".join(lines)

    def format_docs(self, docs: list[Document]) -> str:
        """
        Formate les documents retenus en un bloc texte unique pour le LLM.
        """
        if not docs:
            return ""

        return "\n\n".join(
            self.format_doc(doc, idx)
            for idx, doc in enumerate(docs, start=1)
        )

    def format_docs_list(self, docs: list[Document]) -> list[str]:
        """
        Formate séparément chaque document.

        Utile pour le debug, l'évaluation et les inspections fines.
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
        Convertit un document LangChain en objet structuré pour l'API.
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
        Convertit les documents LangChain en objets structurés pour l'API.
        """
        return [self._build_retrieved_document(doc) for doc in docs]

    # ------------------------------------------------------------------
    # Génération
    # ------------------------------------------------------------------

    def generate_answer(self, question: str, docs: list[Document], session_id: str) -> str:
        """
        Génère une réponse finale à partir de la question, des documents
        et de l'historique court de session.

        Si aucun document n'est disponible, le LLM n'est pas appelé.
        """
        if not docs:
            return self.EMPTY_ANSWER

        context = self.format_docs(docs).strip()
        if not context:
            return self.EMPTY_ANSWER

        history = self.memory_service.format_history_for_prompt(session_id).strip()

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
        Exécute le pipeline RAG complet et retourne toutes les données utiles.

        Étapes
        ------
        1. validation de la question
        2. disponibilité du corpus et de l'index
        3. reformulation contextuelle éventuelle via mémoire courte
        4. préfiltrage structuré
        5. fallback éventuel du préfiltrage
        6. recherche vectorielle locale
        7. ranking métier
        8. contrôle final documentaire
        9. génération de réponse
        10. mise à jour de la mémoire conversationnelle
        11. traçage complet
        """
        if not question or not question.strip():
            raise ValueError("La question utilisateur ne peut pas être vide.")

        self.ensure_index_ready()

        effective_question = self._rewrite_question_with_history(question, session_id)

        filter_debug, fallback_filter_debug, prefiltered_docs, fallback_used = (
            self._prefilter_with_fallback(effective_question)
        )

        raw_docs: list[Document] = []
        ranked_docs: list[Document] = []
        final_docs: list[Document] = []

        if prefiltered_docs:
            raw_docs = self._run_vector_retrieval(
                question=effective_question,
                candidate_docs=prefiltered_docs,
            )

            if raw_docs:
                ranked_docs = self.retrieval_service.rank_documents(
                    question=effective_question,
                    raw_docs=raw_docs,
                    top_k=min(self.top_k_retrieval, len(raw_docs)),
                )

                final_docs = self._post_filter_ranked_docs(ranked_docs)

        context = self.format_docs(final_docs)
        answer = self.generate_answer(question, final_docs, session_id)
        retrieved_documents = self.build_retrieved_documents(final_docs)
        retrieved_contexts = self.format_docs_list(final_docs)

        retrieval_debug = self.retrieval_service.rank_documents_with_scores(
            question=effective_question,
            raw_docs=raw_docs,
            top_k=len(raw_docs) if raw_docs else self.top_k_final,
        )

        self.memory_service.append_message(session_id, "user", question)
        self.memory_service.append_message(session_id, "assistant", answer)

        history = self.memory_service.get_history(session_id)

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
        Exécute le pipeline RAG complet pour une question utilisateur.
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
        Exécute le pipeline complet avec des informations de debug détaillées.

        Ce mode permet notamment d'inspecter :
        - la question reformulée pour le retrieval
        - l'historique conversationnel court
        - les documents finaux
        - les contextes envoyés au LLM
        - les étapes du filtrage
        - le fallback éventuel
        - les scores métier du ranking
        """
        result = self._run_pipeline(question, session_id=session_id)

        response = {
            "question": result["question"],
            "effective_question": result["effective_question"],
            "session_id": result["session_id"],
            "history": result["history"],
            "answer": result["answer"],
            "n_docs": len(result["docs"]),
            "documents": [
                doc.model_dump() if hasattr(doc, "model_dump") else doc.dict()
                for doc in result["retrieved_documents"]
            ],
            "retrieved_contexts": result["retrieved_contexts"],
            "fallback_used": result["fallback_used"],
            "filter_debug": result["filter_debug"],
            "retrieval_debug": result["retrieval_debug"],
        }

        if result["fallback_filter_debug"] is not None:
            response["fallback_filter_debug"] = result["fallback_filter_debug"]

        return self._serialize_for_json(response)