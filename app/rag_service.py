from __future__ import annotations

"""
Service principal du pipeline RAG.

Cette classe implémente un pipeline Retrieval-Augmented Generation
orienté recommandation d'événements culturels.

Le service combine trois briques principales :

- un index vectoriel FAISS pour retrouver les documents les plus proches
- une mémoire locale pour réutiliser certaines réponses passées
- un modèle de langage pour générer une réponse finale

Le fonctionnement général suit cette logique :

1. vérification d'un éventuel choix utilisateur déjà mémorisé
2. vérification d'une question déjà posée à l'identique
3. retrieval documentaire dans l'index vectoriel
4. construction d'un contexte combinant documents + mémoire
5. génération de la réponse finale
6. sauvegarde de l'échange en mémoire

L'objectif est de rendre le système plus fluide dans une conversation
tout en conservant le contexte documentaire comme source principale
de vérité.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from app.memory_service import MemoryService
from app.schemas import AskResponse, RetrievedDocument


class RAGService:
    """
    Service central du système RAG.

    Cette classe gère l'ensemble du pipeline de question-réponse,
    depuis le stockage des documents jusqu'à la génération finale.

    Elle s'appuie sur :
    - un stockage vectoriel FAISS pour la recherche sémantique
    - un service de mémoire locale pour réutiliser des échanges passés
    - un modèle de langage pour produire la réponse finale

    Parameters
    ----------
    documents : list[Document], optional
        Liste initiale de documents à indexer.
    index_dir : str, default="faiss_index_openagenda"
        Répertoire contenant l'index vectoriel FAISS.
    embedding_model : str, default="mistral-embed"
        Modèle utilisé pour générer les embeddings des documents.
    llm_model : str, default="mistral-small-latest"
        Modèle de langage utilisé pour la génération de réponses.
    temperature : float, default=0.2
        Température du modèle de langage.
    default_k : int, default=3
        Nombre de documents récupérés par défaut lors du retrieval.
    """

    def __init__(
        self,
        documents: list[Document] | None = None,
        index_dir: str = "faiss_index_openagenda",
        embedding_model: str = "mistral-embed",
        llm_model: str = "mistral-small-latest",
        temperature: float = 0.2,
        default_k: int = 3,
    ) -> None:
        self.documents = documents or []
        self.index_dir = Path(index_dir)
        self.default_k = default_k

        self.embeddings = MistralAIEmbeddings(model=embedding_model)

        self.llm = ChatMistralAI(
            model=llm_model,
            temperature=temperature,
        )

        self.memory_service = MemoryService()
        self.vectorstore: FAISS | None = None

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

        self.chain = self.prompt | self.llm | StrOutputParser()

    def set_documents(self, documents: list[Document]) -> None:
        """
        Définit la liste de documents à utiliser par le service.

        Cette méthode permet de remplacer les documents actuellement
        stockés en mémoire, par exemple après un rechargement ou
        une reconstruction de l'index.

        Parameters
        ----------
        documents : list[Document]
            Documents LangChain à utiliser.
        """
        self.documents = documents

    def build_index(self, documents: list[Document] | None = None) -> int:
        """
        Construit l'index vectoriel FAISS à partir des documents disponibles.

        Si une liste de documents est fournie, elle remplace les documents
        actuellement stockés dans le service.

        Cette opération génère les embeddings de tous les documents
        puis sauvegarde l'index sur disque.

        Parameters
        ----------
        documents : list[Document], optional
            Documents à indexer. Si None, utilise ceux déjà présents.

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
        sans recalculer les embeddings.

        Raises
        ------
        FileNotFoundError
            Si l'index n'existe pas dans le répertoire configuré.
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

        Si l'index n'est pas encore chargé en mémoire,
        il est automatiquement chargé depuis le disque.
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
        k : int, optional
            Nombre de documents à récupérer. Si None, utilise `default_k`.

        Returns
        -------
        list[Document]
            Documents les plus proches de la question.

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
        Formate les documents récupérés en contexte textuel.

        Chaque document est transformé en bloc structuré contenant
        les métadonnées principales de l'événement ainsi que
        son contenu textuel.

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
        - le contexte documentaire issu des documents retrouvés
        - un contexte mémoire issu des échanges précédents

        Si aucune mémoire pertinente n'est trouvée, un message par défaut
        est inséré.

        Parameters
        ----------
        question : str
            Question utilisateur.
        docs : list[Document]
            Documents récupérés par le retrieval.

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

        La réponse finale s'appuie sur :
        - les documents retrouvés dans l'index
        - la mémoire locale, utilisée comme aide complémentaire

        Parameters
        ----------
        question : str
            Question utilisateur.
        docs : list[Document]
            Documents retrouvés dans l'index.

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

        Parameters
        ----------
        doc : Document
            Document LangChain à convertir.

        Returns
        -------
        RetrievedDocument
            Représentation sérialisable du document pour l'API.
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
        2. recherche d'une question identique déjà stockée en mémoire
        3. retrieval documentaire classique dans l'index
        4. génération de la réponse
        5. sauvegarde du nouvel échange en mémoire

        Parameters
        ----------
        question : str
            Question utilisateur.
        k : int, optional
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

        # 1. Cas spécial : référence à un choix précédent
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

        # 2. Cas spécial : question exacte déjà présente en mémoire
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
                documents=[doc.model_dump() for doc in response.documents],
            )

            return response

        # 3. Pipeline RAG normal
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

        # 4. Sauvegarde mémoire
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
            True si l'index FAISS est disponible, sinon False.
        """
        return self.vectorstore is not None