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
3. récupération des documents pertinents dans l'index vectoriel
4. construction d'un contexte combinant mémoire et documents
5. génération de la réponse finale
6. sauvegarde de l'échange dans la mémoire locale

L'objectif est de conserver le contexte documentaire comme source
principale d'information, tout en rendant les échanges plus fluides
dans une logique conversationnelle.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

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
        Nombre de documents récupérés par défaut lors du retrieval.
    """

    def __init__(
        self,
        documents: list[Document] | None = None,
        index_dir: str | Path = INDEX_DIR,
        embedding_model: str = "mistral-embed",
        llm_model: str = "mistral-small-latest",
        temperature: float = 0.2,
        default_k: int = 3,
    ) -> None:
        self.documents = documents or []
        self.index_dir = Path(index_dir)
        self.default_k = default_k

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

Tu dois répondre uniquement à partir du CONTEXTE DOCUMENTAIRE fourni.

Tu peux utiliser la MÉMOIRE seulement comme aide complémentaire si elle est pertinente,
mais tu ne dois jamais inventer d'événement qui ne serait pas présent dans le contexte documentaire.

Règles :
- N'invente aucun événement
- Utilise uniquement les événements présents dans le contexte
- Si plusieurs événements sont pertinents, présente-les séparément.
- Si aucune information n’est disponible, réponds exactement : "Je ne trouve pas d'événement correspondant, je suis un assistant qui ne peut vous conseiller que des événements culturels."
et ne propose rien.

Format de réponse OBLIGATOIRE :

Pour chaque événement trouvé, réponds sous cette forme :

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
            allow_dangerous_deserialization=True
        )

    def ensure_index_ready(self) -> None:
        """
        Vérifie que l'index vectoriel est prêt à être utilisé.

        Si l'index n'est pas encore chargé en mémoire, il est
        automatiquement chargé depuis le disque.
        """
        if self.vectorstore is None:
            self.load_index()

    def retrieve(self, question: str, k: int | None = None) -> list[Document]:
        """
        Récupère les documents les plus proches d'une question.

        La recherche est effectuée dans l'index vectoriel FAISS
        à partir de la représentation sémantique de la question.

        Parameters
        ----------
        question : str
            Question posée par l'utilisateur.
        k : int | None, default=None
            Nombre de documents à récupérer. Si None, utilise `default_k`.

        Returns
        -------
        list[Document]
            Liste des documents les plus proches.

        Raises
        ------
        ValueError
            Si la question est vide.
        """
        if not question or not question.strip():
            raise ValueError("La question ne peut pas être vide.")

        self.ensure_index_ready()

        final_k = self.default_k if k is None else k

        docs = self.vectorstore.similarity_search(
            question.strip(),
            k=final_k
        )

        return docs

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

            block = f"""
Événement {i}
Titre : {md.get('title', '')}
Lieu : {md.get('location_name', '')}
Ville : {md.get('city', '')}
Région : {md.get('region', '')}
Date de début : {md.get('first_date', '')}
Date de fin : {md.get('last_date', '')}
Type : {md.get('event_type', '')}
URL : {md.get('url', '')}
Contenu : {doc.page_content}
"""
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
            "memory_context": memory_context
        }

    def generate(self, question: str, docs: list[Document]) -> str:
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

        Returns
        -------
        str
            Réponse générée par le modèle de langage.
        """
        full_context = self.build_full_context(question=question, docs=docs)

        return self.chain.invoke(
            {
                "question": question.strip(),
                "context": full_context["context"],
                "memory_context": full_context["memory_context"]
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
            score=None
        )

    def ask(self, question: str, k: int | None = None) -> AskResponse:
        """
        Exécute le pipeline complet de question-réponse.

        Le traitement suit les étapes suivantes :
        1. vérification d'une éventuelle référence à un choix précédent
        2. recherche d'une question identique déjà présente en mémoire
        3. retrieval documentaire dans l'index vectoriel
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

        # Cas 1 : l'utilisateur fait référence à un choix précédent.
        choice_result = self.memory_service.build_choice_answer(question)
        if choice_result is not None:
            choice_docs = [
                RetrievedDocument(**doc_data)
                for doc_data in choice_result.get("documents", [])
            ]

            response = AskResponse(
                question=question,
                answer=choice_result.get("answer", ""),
                n_docs=len(choice_docs),
                documents=choice_docs
            )

            self.memory_service.add_entry(
                question=question,
                answer=response.answer,
                documents=[doc.model_dump() for doc in response.documents]
            )

            return response

        # Cas 2 : la même question existe déjà dans la mémoire.
        exact_memory = self.memory_service.find_exact_question(question)
        if exact_memory:
            memory_docs = [
                RetrievedDocument(**doc_data)
                for doc_data in exact_memory.get("documents", [])
            ]

            response = AskResponse(
                question=question,
                answer=exact_memory.get("answer", ""),
                n_docs=len(memory_docs),
                documents=memory_docs
            )

            self.memory_service.add_entry(
                question=question,
                answer=response.answer,
                documents=[doc.model_dump() for doc in response.documents]
            )

            return response

        # Cas 3 : exécution normale du pipeline RAG.
        docs = self.retrieve(question=question, k=k)
        answer = self.generate(question=question, docs=docs)

        retrieved_docs = [
            self.to_retrieved_document(doc)
            for doc in docs
        ]

        response = AskResponse(
            question=question,
            answer=answer,
            n_docs=len(docs),
            documents=retrieved_docs
        )

        # Sauvegarde de la nouvelle interaction dans la mémoire locale.
        self.memory_service.add_entry(
            question=question,
            answer=answer,
            documents=[doc.model_dump() for doc in retrieved_docs]
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