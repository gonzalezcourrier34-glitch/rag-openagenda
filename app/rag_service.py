from __future__ import annotations

from datetime import datetime
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from app.document_service import (
    doc_matches_filters,
    extract_filters_from_question,
    score_document,
)
from app.memory_service import MemoryService
from app.schemas import AskResponse, RetrievedDocument


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index" / "faiss_index_openagenda"


class RAGService:
    """
    Service principal du pipeline RAG pour la recommandation d'événements culturels.

    Ce service orchestre :
    - la recherche documentaire (FAISS)
    - le filtrage métier basé sur les métadonnées
    - le reranking des documents
    - la génération de réponse via un LLM
    - la gestion d'une mémoire conversationnelle

    Pipeline :
    ----------
    1. gestion mémoire (choix utilisateur / question déjà posée)
    2. retrieval vectoriel
    3. extraction de filtres métier
    4. filtrage strict des documents
    5. reranking
    6. génération de réponse
    7. sauvegarde en mémoire
    """

    FALLBACK_NO_RESULT_MESSAGE = (
        "Je n'ai trouvé aucun événement correspondant dans les données disponibles."
    )

    def __init__(
        self,
        documents: list[Document] | None = None,
        index_dir: str | Path = INDEX_DIR,
        embedding_model: str = "mistral-embed",
        llm_model: str = "mistral-small-latest",
        temperature: float = 0.2,
        default_k: int = 3,
        initial_fetch_k: int = 10,
    ) -> None:

        self.documents = documents or []
        self.index_dir = Path(index_dir)

        self.default_k = default_k
        self.initial_fetch_k = initial_fetch_k

        # Embeddings pour FAISS
        self.embeddings = MistralAIEmbeddings(model=embedding_model)

        # LLM
        self.llm = ChatMistralAI(
            model=llm_model,
            temperature=temperature,
        )

        # Mémoire conversationnelle
        self.memory_service = MemoryService()

        # Vectorstore (chargé à la demande)
        self.vectorstore: FAISS | None = None

        # Prompt strict (compatible évaluation RAGAS)
        self.prompt = ChatPromptTemplate.from_template(
            """
Tu es un assistant spécialisé dans la recommandation d'événements culturels.

Tu dois répondre uniquement à partir du CONTEXTE DOCUMENTAIRE fourni.

RÈGLES :
- Ne pas inventer d'information
- Ne pas utiliser de connaissances externes

FORMAT :
Liste d'événements :

- <titre> (date : <date>, lieu : <lieu>)

CAS DE REFUS :
"Je n'ai trouvé aucun événement correspondant dans les données disponibles."

Question :
{question}

Contexte :
{context}
"""
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    # UTILITAIRES
    def _get_current_date(self) -> str:
        """Retourne la date actuelle (YYYY-MM-DD)."""
        return datetime.today().strftime("%Y-%m-%d")

    def _normalize_k(self, k: int | None) -> int:
        """Garantit un k >= 1."""
        return max(1, self.default_k if k is None else k)

    def _ensure_vectorstore_available(self) -> FAISS:
        """Charge ou construit l'index FAISS si nécessaire."""
        self.ensure_index_ready()

        if self.vectorstore is None:
            raise RuntimeError("Vectorstore indisponible.")

        return self.vectorstore

    # INDEX
    def build_index(self, documents: list[Document] | None = None) -> int:
        """Construit l'index FAISS."""
        if documents is not None:
            self.documents = documents

        if not self.documents:
            raise ValueError("Aucun document pour construire l'index.")

        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
        self.vectorstore.save_local(str(self.index_dir))

        return len(self.documents)

    def load_index(self) -> None:
        """Charge un index existant."""
        if not self.index_dir.exists():
            raise FileNotFoundError("Index introuvable.")

        self.vectorstore = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def ensure_index_ready(self) -> None:
        """Assure que l'index est disponible."""
        if self.vectorstore is None:
            if self.index_dir.exists():
                self.load_index()
            elif self.documents:
                self.build_index()
            else:
                raise FileNotFoundError("Aucun index ni document.")

    # RETRIEVE
    def retrieve(self, question: str, k: int | None = None) -> list[Document]:
        """
        Récupère les documents pertinents avec filtrage métier strict.

        Pipeline :
        ----------
        1. extraction des filtres métier
        2. retrieval vectoriel large
        3. filtrage strict
        4. reranking
        """

        if not question.strip():
            raise ValueError("Question vide.")

        final_k = self._normalize_k(k)
        initial_k = max(self.initial_fetch_k, final_k * 3)

        vectorstore = self._ensure_vectorstore_available()

        # 🔥 Extraction filtres (CORRECTION MAJEURE)
        filters = extract_filters_from_question(
            question=question,
            documents=self.documents,
        )

        # 🔍 Retrieval initial
        raw_docs = vectorstore.similarity_search(
            question.strip(),
            k=initial_k,
        )

        if not raw_docs:
            return []

        # 🔥 Filtrage strict
        filtered_docs = [
            doc for doc in raw_docs
            if doc_matches_filters(doc, filters)
        ]

        # ❌ PAS de fallback dangereux
        if not filtered_docs:
            return []

        # 🔥 Reranking métier
        ranked_docs = sorted(
            filtered_docs,
            key=lambda doc: score_document(question, doc, filters),
            reverse=True,
        )

        return ranked_docs[:final_k]

    # FORMAT CONTEXTE
    def format_docs(self, docs: list[Document]) -> str:
        """Transforme les documents en contexte texte."""
        if not docs:
            return "Aucun événement trouvé."

        blocks = []

        for i, doc in enumerate(docs, 1):
            md = doc.metadata or {}

            block = "\n".join([
                f"Événement {i}",
                f"Titre : {md.get('title', '')}",
                f"Lieu : {md.get('location_name', '')}",
                f"Ville : {md.get('city', '')}",
                f"Date début : {md.get('first_date', '')}",
                f"Date fin : {md.get('last_date', '')}",
                f"Contenu : {doc.page_content}",
            ])

            blocks.append(block)

        return "\n\n".join(blocks)

    # GENERATION
    def generate(self, question: str, docs: list[Document]) -> str:
        """Génère la réponse finale."""
        if not docs:
            return self.FALLBACK_NO_RESULT_MESSAGE

        context = self.format_docs(docs)

        return self.chain.invoke({
            "question": question.strip(),
            "context": context,
        })

    # API RESPONSE
    def to_retrieved_document(self, doc: Document) -> RetrievedDocument:
        """Convertit un Document en DTO API."""
        md = doc.metadata or {}

        return RetrievedDocument(
            title=md.get("title", ""),
            location_name=md.get("location_name", ""),
            city=md.get("city", ""),
            region=md.get("region", ""),
            first_date=md.get("first_date", ""),
            last_date=md.get("last_date", ""),
            event_type=md.get("event_type", ""),
            url=md.get("url", ""),
            score=None,
        )

    # ASK PIPELINE
    def ask(self, question: str, k: int | None = None) -> AskResponse:
        """
        Exécute le pipeline complet RAG.

        Étapes :
        --------
        1. gestion mémoire
        2. retrieval
        3. génération
        4. sauvegarde
        """

        question = question.strip()

        if not question:
            raise ValueError("Question vide.")

        # Mémoire - choix utilisateur
        choice = self.memory_service.build_choice_answer(question)
        if choice:
            docs = [RetrievedDocument(**d) for d in choice.get("documents", [])]
            return AskResponse(question=question, answer=choice["answer"], n_docs=len(docs), documents=docs)

        # Mémoire - question identique
        memory = self.memory_service.find_exact_question(question)
        if memory:
            docs = [RetrievedDocument(**d) for d in memory.get("documents", [])]
            return AskResponse(question=question, answer=memory["answer"], n_docs=len(docs), documents=docs)

        # Pipeline normal
        docs = self.retrieve(question, k)
        answer = self.generate(question, docs)

        retrieved_docs = [self.to_retrieved_document(d) for d in docs]

        response = AskResponse(
            question=question,
            answer=answer,
            n_docs=len(retrieved_docs),
            documents=retrieved_docs,
        )

        # Sauvegarde mémoire
        self.memory_service.add_entry(
            question=question,
            answer=answer,
            documents=[doc.model_dump() for doc in retrieved_docs],
        )

        return response

    # UTILS
    def is_index_loaded(self) -> bool:
        """Indique si l'index est chargé."""
        return self.vectorstore is not None