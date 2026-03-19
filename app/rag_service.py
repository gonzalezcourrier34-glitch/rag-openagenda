"""
Service principal du pipeline RAG.

Ce module implémente la logique centrale du système de
Retrieval-Augmented Generation appliqué à la recommandation
d'événements culturels.

Le service repose sur trois briques complémentaires :

- un index vectoriel FAISS pour retrouver les documents les plus pertinents
- une mémoire locale pour réutiliser certains échanges passés
- un modèle de langage pour générer une réponse finale contextualisée

Le déroulement général du pipeline suit les étapes suivantes :

1. vérification d'un éventuel choix utilisateur basé sur la mémoire
2. recherche d'une question déjà posée à l'identique
3. récupération initiale de documents candidats dans l'index vectoriel
4. extraction de filtres métier depuis la question
5. filtrage et reranking des documents candidats
6. construction d'un contexte combinant mémoire et documents
7. génération de la réponse finale
8. sauvegarde de l'échange dans la mémoire locale

L'objectif est de conserver le contexte documentaire comme source
principale d'information, tout en améliorant la pertinence du retrieval
grâce à une logique métier simple fondée sur les métadonnées
des événements.
"""
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


# Racine du projet
BASE_DIR = Path(__file__).resolve().parents[1]

# Répertoires de données
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index" / "faiss_index_openagenda"


class RAGService:
    """
    Service central du pipeline RAG.

    Cette classe orchestre les différentes étapes du système :
    indexation, recherche documentaire, génération de réponse
    et gestion d'une mémoire conversationnelle locale.

    Elle s'appuie sur :
    - un stockage vectoriel FAISS pour la recherche sémantique
    - un service de mémoire locale pour réutiliser certains échanges
    - un modèle de langage pour produire la réponse finale
    - un ensemble de fonctions documentaires pour filtrer et reranker
      les événements en fonction de la question

    Parameters
    ----------
    documents : list[Document] | None, default=None
        Liste initiale de documents à indexer.
    index_dir : str | Path, default=INDEX_DIR
        Répertoire contenant l'index vectoriel FAISS.
    embedding_model : str, default="mistral-embed"
        Modèle utilisé pour générer les embeddings.
    llm_model : str, default="mistral-small-latest"
        Modèle de langage utilisé pour la génération de réponses.
    temperature : float, default=0.2
        Température utilisée par le modèle de langage.
    default_k : int, default=3
        Nombre de documents retournés par défaut après reranking.
    initial_fetch_k : int, default=10
        Nombre initial de documents candidats récupérés avant filtrage
        et reranking.
    """

    FALLBACK_NO_RESULT_MESSAGE = (
        "Je ne trouve pas d'événement correspondant, je suis un assistant "
        "qui ne peut vous conseiller que des événements culturels."
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

        # Modèle d'embedding utilisé pour vectoriser les documents.
        self.embeddings = MistralAIEmbeddings(model=embedding_model)

        # Modèle de langage utilisé pour générer la réponse finale.
        self.llm = ChatMistralAI(
            model=llm_model,
            temperature=temperature,
        )

        # Service de mémoire conversationnelle locale.
        self.memory_service = MemoryService()

        # Stockage vectoriel FAISS chargé en mémoire à la demande.
        self.vectorstore: FAISS | None = None

        # Prompt principal transmis au modèle de langage.
        self.prompt = ChatPromptTemplate.from_template(
            """
Tu es un assistant spécialisé dans la recommandation d'événements culturels.

Date actuelle : {current_date}

Tu dois répondre uniquement à partir du CONTEXTE DOCUMENTAIRE fourni.

Tu peux utiliser la MÉMOIRE uniquement comme aide complémentaire si elle est pertinente,
mais tu ne dois jamais inventer d'événement qui ne serait pas présent dans le contexte documentaire.

---

RÈGLES DE RAISONNEMENT :

Avant de répondre, analyse chaque événement du contexte et vérifie sa cohérence avec la question.

Prends en compte les contraintes suivantes si elles sont explicitement demandées :
- ville
- lieu précis
- date ou période
- type d'événement
- mots-clés

---

VALIDATION DES ÉVÉNEMENTS :

- Un événement est valide s'il respecte clairement les contraintes exprimées dans la question.
- Si une contrainte importante n'est pas présente dans le contexte,
  indique que cette information n'est pas confirmée.
- Ne rejette pas automatiquement un événement pertinent à cause d'une information manquante non critique.

- Pour une question large, tu peux proposer plusieurs événements pertinents.

---

CAS DE REFUS :

Tu dois répondre EXACTEMENT :

"Je ne trouve pas d'événement correspondant, je suis un assistant qui ne peut vous conseiller que des événements culturels."

UNIQUEMENT si :
- aucun événement du contexte ne correspond à la demande
- ou la demande est hors périmètre
- ou les contraintes sont trop strictes et aucun événement ne les respecte

---

INTERDICTIONS :

- N'invente aucun événement
- N'utilise pas de connaissances externes
- Ne suppose jamais une information absente

---

FORMAT DE RÉPONSE OBLIGATOIRE :

Pour chaque événement retenu :

Titre : <titre>

Lieu : <ville / lieu>
Date : <date>
Description : <résumé court basé sur le contexte>

Pourquoi cet événement pourrait vous intéresser :
<explication personnalisée basée sur la question utilisateur>

---

Question :
{question}

Mémoire :
{memory_context}

Contexte documentaire :
{context}
"""
        )

        # Chaîne complète : prompt -> modèle -> sortie texte.
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _get_current_date(self) -> str:
        """
        Retourne la date actuelle au format YYYY-MM-DD.

        Cette date est injectée dans le prompt afin de permettre
        l'interprétation correcte des expressions temporelles
        relatives comme "aujourd'hui", "demain" ou "ce week-end".

        Returns
        -------
        str
            Date actuelle au format YYYY-MM-DD.
        """
        return datetime.today().strftime("%Y-%m-%d")

    def _normalize_k(self, k: int | None) -> int:
        """
        Normalise le nombre de documents à retourner.

        Parameters
        ----------
        k : int | None
            Nombre demandé par l'appelant.

        Returns
        -------
        int
            Valeur finale strictement positive.
        """
        final_k = self.default_k if k is None else k
        return max(1, final_k)

    def _ensure_vectorstore_available(self) -> FAISS:
        """
        Garantit que le vectorstore est disponible en mémoire.

        Returns
        -------
        FAISS
            Index vectoriel prêt à être utilisé.

        Raises
        ------
        RuntimeError
            Si le vectorstore reste indisponible après initialisation.
        """
        self.ensure_index_ready()

        if self.vectorstore is None:
            raise RuntimeError(
                "Le vectorstore n'est pas disponible après initialisation."
            )

        return self.vectorstore

    def _build_ask_response(
        self,
        question: str,
        answer: str,
        documents: list[RetrievedDocument],
    ) -> AskResponse:
        """
        Construit un objet de réponse API standardisé.

        Parameters
        ----------
        question : str
            Question utilisateur.
        answer : str
            Réponse finale.
        documents : list[RetrievedDocument]
            Documents retournés.

        Returns
        -------
        AskResponse
            Réponse complète sérialisable.
        """
        return AskResponse(
            question=question,
            answer=answer,
            n_docs=len(documents),
            documents=documents,
        )

    def _save_interaction_in_memory(
        self,
        question: str,
        answer: str,
        documents: list[RetrievedDocument],
    ) -> None:
        """
        Sauvegarde une interaction dans la mémoire locale.

        Parameters
        ----------
        question : str
            Question utilisateur.
        answer : str
            Réponse produite.
        documents : list[RetrievedDocument]
            Documents associés à la réponse.
        """
        self.memory_service.add_entry(
            question=question,
            answer=answer,
            documents=[doc.model_dump() for doc in documents],
        )

    def set_documents(self, documents: list[Document]) -> None:
        """
        Définit la liste de documents utilisée par le service.

        Cette méthode permet de remplacer les documents actuellement
        stockés dans le service, par exemple après un rechargement
        ou une reconstruction de l'index.

        Parameters
        ----------
        documents : list[Document]
            Documents LangChain à utiliser.
        """
        self.documents = documents

    def build_index(self, documents: list[Document] | None = None) -> int:
        """
        Construit l'index vectoriel FAISS à partir des documents disponibles.

        Si une liste de documents est fournie, elle remplace d'abord
        les documents actuellement stockés dans le service.

        Cette opération génère les embeddings puis sauvegarde l'index
        sur disque dans le répertoire configuré.

        Parameters
        ----------
        documents : list[Document] | None, default=None
            Documents à indexer. Si None, utilise les documents déjà chargés.

        Returns
        -------
        int
            Nombre de documents indexés.

        Raises
        ------
        ValueError
            Si aucun document n'est disponible pour construire l'index.
        """
        if documents is not None:
            self.documents = documents

        if not self.documents:
            raise ValueError("Aucun document disponible pour construire l'index.")

        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
        self.vectorstore.save_local(str(self.index_dir))

        return len(self.documents)

    def load_index(self) -> None:
        """
        Charge un index FAISS existant depuis le disque.

        Cette méthode permet de réutiliser un index déjà construit
        sans recalculer les embeddings des documents.

        Raises
        ------
        FileNotFoundError
            Si le répertoire de l'index n'existe pas.
        """
        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"Index introuvable dans '{self.index_dir}'."
            )

        self.vectorstore = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def ensure_index_ready(self) -> None:
        """
        Vérifie que l'index vectoriel est prêt à être utilisé.

        Si l'index n'est pas encore chargé en mémoire, il est
        automatiquement chargé depuis le disque ou reconstruit
        à partir des documents déjà disponibles.

        Raises
        ------
        FileNotFoundError
            Si aucun index n'est disponible et qu'aucun document
            n'est présent pour le reconstruire.
        """
        if self.vectorstore is None:
            if self.index_dir.exists():
                self.load_index()
            elif self.documents:
                self.build_index()
            else:
                raise FileNotFoundError(
                    f"Index introuvable dans '{self.index_dir}' et aucun document disponible pour le reconstruire."
                )

    def retrieve(self, question: str, k: int | None = None) -> list[Document]:
        """
        Récupère les documents les plus pertinents pour une question.

        La méthode suit une logique en plusieurs étapes :

        1. récupération initiale de documents candidats via la recherche sémantique
        2. extraction de filtres métier depuis la question
        3. filtrage des documents incompatibles avec la requête
        4. reranking des documents restants
        5. retour des `k` meilleurs documents

        Si le filtrage s'avère trop strict et élimine tous les candidats,
        le service retombe sur les résultats bruts issus du moteur vectoriel.

        Parameters
        ----------
        question : str
            Question posée par l'utilisateur.
        k : int | None, default=None
            Nombre de documents à retourner. Si None, utilise `default_k`.

        Returns
        -------
        list[Document]
            Liste des documents jugés les plus pertinents.

        Raises
        ------
        ValueError
            Si la question est vide.
        """
        if not question or not question.strip():
            raise ValueError("La question ne peut pas être vide.")

        final_k = self._normalize_k(k)
        initial_k = max(self.initial_fetch_k, final_k * 3)

        vectorstore = self._ensure_vectorstore_available()

        # Première récupération large de documents candidats.
        raw_docs = vectorstore.similarity_search(
            question.strip(),
            k=initial_k,
        )

        if not raw_docs:
            return []

        # Extraction de filtres métier à partir de la question
        # et des métadonnées connues dans les documents candidats.
        filters = extract_filters_from_question(
            question=question,
            documents=raw_docs if raw_docs else self.documents,
        )

        # Filtrage documentaire fondé sur les métadonnées.
        filtered_docs = [
            doc for doc in raw_docs
            if doc_matches_filters(doc, filters)
        ]

        # Si le filtrage est trop strict, on conserve les documents bruts.
        candidate_docs = filtered_docs if filtered_docs else raw_docs

        # Reranking métier des documents conservés.
        ranked_docs = sorted(
            candidate_docs,
            key=lambda doc: score_document(question, doc, filters),
            reverse=True,
        )

        return ranked_docs[:final_k]

    def format_docs(self, docs: list[Document]) -> str:
        """
        Formate les documents récupérés en un contexte textuel structuré.

        Chaque document est transformé en bloc lisible contenant
        ses principales métadonnées ainsi que son contenu textuel.

        Parameters
        ----------
        docs : list[Document]
            Documents à formater.

        Returns
        -------
        str
            Contexte documentaire prêt à être injecté dans le prompt.
        """
        if not docs:
            return "Aucun événement trouvé."

        blocks = []

        for i, doc in enumerate(docs, start=1):
            md = doc.metadata or {}

            content = str(doc.page_content).strip()
            block = "\n".join(
                [
                    f"Événement {i}",
                    f"Titre : {md.get('title', '')}",
                    f"Lieu : {md.get('location_name', '')}",
                    f"Ville : {md.get('city', '')}",
                    f"Région : {md.get('region', '')}",
                    f"Date de début : {md.get('first_date', '')}",
                    f"Date de fin : {md.get('last_date', '')}",
                    f"Type : {md.get('event_type', '')}",
                    f"URL : {md.get('url', '')}",
                    f"Contenu : {content}",
                ]
            )
            blocks.append(block.strip())

        return "\n\n".join(blocks)

    def build_full_context(self, question: str, docs: list[Document]) -> dict[str, str]:
        """
        Construit le contexte complet transmis au modèle de langage.

        Ce contexte combine :
        - le contexte documentaire issu des documents récupérés
        - un contexte mémoire issu des échanges passés

        Si aucun souvenir pertinent n'est disponible, un message
        par défaut est utilisé.

        Parameters
        ----------
        question : str
            Question utilisateur.
        docs : list[Document]
            Documents retrouvés par le moteur de recherche.

        Returns
        -------
        dict[str, str]
            Dictionnaire contenant :
            - `context` : contexte documentaire
            - `memory_context` : contexte mémoire
        """
        context = self.format_docs(docs)
        memory_context = self.memory_service.build_memory_context(question=question)

        if not memory_context:
            memory_context = "Aucun souvenir pertinent trouvé."

        return {
            "context": context,
            "memory_context": memory_context,
        }

    def generate(
        self,
        question: str,
        docs: list[Document],
        current_date: str,
    ) -> str:
        """
        Génère une réponse à partir de la question et des documents récupérés.

        La réponse finale s'appuie sur les documents retrouvés dans
        l'index ainsi que, si besoin, sur un rappel mémoire léger.

        Parameters
        ----------
        question : str
            Question utilisateur.
        docs : list[Document]
            Documents récupérés dans l'index.
        current_date : str
            Date actuelle au format YYYY-MM-DD, utilisée pour interpréter
            les contraintes temporelles relatives dans la question.

        Returns
        -------
        str
            Réponse générée par le modèle de langage.
        """
        if not docs:
            return self.FALLBACK_NO_RESULT_MESSAGE

        full_context = self.build_full_context(question=question, docs=docs)

        return self.chain.invoke(
            {
                "question": question.strip(),
                "context": full_context["context"],
                "memory_context": full_context["memory_context"],
                "current_date": current_date,
            }
        )

    def to_retrieved_document(self, doc: Document) -> RetrievedDocument:
        """
        Convertit un document LangChain en objet de réponse API.

        Cette conversion permet de ne conserver que les champs utiles
        pour la sérialisation et l'affichage côté API ou interface.

        Parameters
        ----------
        doc : Document
            Document LangChain à convertir.

        Returns
        -------
        RetrievedDocument
            Représentation sérialisable du document.
        """
        metadata = doc.metadata or {}

        return RetrievedDocument(
            title=metadata.get("title", ""),
            location_name=metadata.get("location_name", ""),
            city=metadata.get("city", ""),
            region=metadata.get("region", ""),
            first_date=metadata.get("first_date", ""),
            last_date=metadata.get("last_date", ""),
            event_type=metadata.get("event_type", ""),
            url=metadata.get("url", ""),
            score=None,
        )

    def ask(self, question: str, k: int | None = None) -> AskResponse:
        """
        Exécute le pipeline complet de question-réponse.

        Le traitement suit les étapes suivantes :
        1. vérification d'une éventuelle référence à un choix précédent
        2. recherche d'une question identique déjà présente en mémoire
        3. retrieval documentaire avec filtrage et reranking
        4. génération de la réponse
        5. sauvegarde du nouvel échange en mémoire

        Parameters
        ----------
        question : str
            Question utilisateur.
        k : int | None, default=None
            Nombre de documents à récupérer.

        Returns
        -------
        AskResponse
            Réponse complète contenant :
            - la question
            - la réponse générée
            - le nombre de documents utilisés
            - les documents retournés

        Raises
        ------
        ValueError
            Si la question est vide.
        """
        question = question.strip()

        if not question:
            raise ValueError("La question ne peut pas être vide.")

        current_date = self._get_current_date()

        # Cas 1 : l'utilisateur fait référence à un choix précédent.
        choice_result = self.memory_service.build_choice_answer(question)
        if choice_result is not None:
            choice_docs = [
                RetrievedDocument(**doc_data)
                for doc_data in choice_result.get("documents", [])
            ]

            return self._build_ask_response(
                question=question,
                answer=choice_result.get("answer", ""),
                documents=choice_docs,
            )

        # Cas 2 : la même question existe déjà dans la mémoire.
        exact_memory = self.memory_service.find_exact_question(question)
        if exact_memory:
            memory_docs = [
                RetrievedDocument(**doc_data)
                for doc_data in exact_memory.get("documents", [])
            ]

            return self._build_ask_response(
                question=question,
                answer=exact_memory.get("answer", ""),
                documents=memory_docs,
            )

        # Cas 3 : exécution normale du pipeline RAG.
        docs = self.retrieve(question=question, k=k)
        answer = self.generate(
            question=question,
            docs=docs,
            current_date=current_date,
        )

        retrieved_docs = [
            self.to_retrieved_document(doc)
            for doc in docs
        ]

        response = self._build_ask_response(
            question=question,
            answer=answer,
            documents=retrieved_docs,
        )

        # Sauvegarde de la nouvelle interaction dans la mémoire locale.
        self._save_interaction_in_memory(
            question=question,
            answer=answer,
            documents=retrieved_docs,
        )

        return response

    def is_index_loaded(self) -> bool:
        """
        Indique si l'index vectoriel est chargé en mémoire.

        Returns
        -------
        bool
            `True` si l'index FAISS est disponible, sinon `False`.
        """
        return self.vectorstore is not None