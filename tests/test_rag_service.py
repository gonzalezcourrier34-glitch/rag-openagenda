import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.rag_service import RAGService


class FakeEmbeddings:
    def __init__(self, *args, **kwargs):
        pass


def fake_llm_factory(*args, **kwargs):
    return RunnableLambda(lambda x: "Réponse simulée")


class FakeChain:
    def __init__(self, answer="Réponse simulée"):
        self.answer = answer

    def invoke(self, payload):
        return self.answer


class FakeVectorStore:
    def __init__(self, documents=None):
        self.documents = documents or []

    def save_local(self, path):
        return None

    def similarity_search(self, query, k=3):
        return self.documents[:k]


class FakeMemoryService:
    def __init__(self, *args, **kwargs):
        self.entries = []

    def build_choice_answer(self, question):
        return None

    def find_exact_question(self, question):
        return None

    def build_memory_context(self, question):
        return ""

    def add_entry(self, question, answer, documents=None):
        entry = {
            "question": question,
            "answer": answer,
            "documents": documents or [],
        }
        self.entries.append(entry)
        return entry


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Exposition architecture à Montpellier",
            metadata={
                "title": "Expo Archi",
                "location_name": "Musée X",
                "city": "Montpellier",
                "region": "Occitanie",
                "first_date": "2026-03-01",
                "last_date": "2026-03-10",
                "event_type": "Exposition",
                "url": "http://test.com",
            },
        ),
        Document(
            page_content="Concert jazz à Sète",
            metadata={
                "title": "Jazz Night",
                "location_name": "Salle Y",
                "city": "Sète",
                "region": "Occitanie",
                "first_date": "2026-03-15",
                "last_date": "2026-03-15",
                "event_type": "Concert",
                "url": "http://concert.com",
            },
        ),
    ]


@pytest.fixture
def patched_rag(monkeypatch):
    monkeypatch.setattr("app.rag_service.MistralAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr("app.rag_service.ChatMistralAI", fake_llm_factory)
    monkeypatch.setattr("app.rag_service.MemoryService", FakeMemoryService)

    monkeypatch.setattr(
        "app.rag_service.extract_filters_from_question",
        lambda question, documents: {
            "cities": [],
            "locations": [],
            "event_types": [],
            "date_start": None,
            "date_end": None,
            "keywords": [],
        },
    )
    monkeypatch.setattr(
        "app.rag_service.doc_matches_filters",
        lambda doc, filters: True,
    )
    monkeypatch.setattr(
        "app.rag_service.score_document",
        lambda question, doc, filters: 0.0,
    )

    return monkeypatch


def test_set_documents(patched_rag, sample_documents):
    rag = RAGService()
    rag.set_documents(sample_documents)

    assert rag.documents == sample_documents


def test_is_index_loaded_false_by_default(patched_rag):
    rag = RAGService()

    assert rag.is_index_loaded() is False


def test_build_index(patched_rag, sample_documents):
    rag = RAGService()
    rag.set_documents(sample_documents)

    def fake_from_documents(documents, embeddings):
        return FakeVectorStore(documents)

    patched_rag.setattr("app.rag_service.FAISS.from_documents", fake_from_documents)

    n_docs = rag.build_index()

    assert n_docs == 2
    assert rag.is_index_loaded() is True


def test_build_index_without_documents(patched_rag):
    rag = RAGService()

    with pytest.raises(ValueError, match="Aucun document disponible"):
        rag.build_index()


def test_load_index_missing_dir_raises(patched_rag, tmp_path):
    rag = RAGService(index_dir=tmp_path / "missing_index")

    with pytest.raises(FileNotFoundError, match="Index introuvable"):
        rag.load_index()


def test_ensure_index_ready_loads_index_when_needed(patched_rag, sample_documents):
    rag = RAGService()
    fake_store = FakeVectorStore(sample_documents)

    def fake_load_local(path, embeddings, allow_dangerous_deserialization=True):
        return fake_store

    patched_rag.setattr("app.rag_service.FAISS.load_local", fake_load_local)

    rag.index_dir.mkdir(parents=True, exist_ok=True)
    rag.ensure_index_ready()

    assert rag.vectorstore is fake_store


def test_retrieve_empty_question_raises(patched_rag):
    rag = RAGService()

    with pytest.raises(ValueError, match="La question ne peut pas être vide"):
        rag.retrieve("   ")


def test_retrieve_returns_docs(patched_rag, sample_documents):
    rag = RAGService()
    rag.vectorstore = FakeVectorStore(sample_documents)

    docs = rag.retrieve("architecture", k=1)

    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Expo Archi"


def test_retrieve_uses_initial_fetch_k(patched_rag, sample_documents):
    rag = RAGService(default_k=2, initial_fetch_k=10)
    fake_store = FakeVectorStore(sample_documents)
    rag.vectorstore = fake_store

    captured = {}

    def fake_similarity_search(query, k=3):
        captured["query"] = query
        captured["k"] = k
        return sample_documents

    fake_store.similarity_search = fake_similarity_search

    docs = rag.retrieve("architecture", k=2)

    assert captured["query"] == "architecture"
    assert captured["k"] == 10
    assert len(docs) == 2


def test_retrieve_falls_back_to_raw_docs_when_all_filtered_out(
    patched_rag, sample_documents, monkeypatch
):
    rag = RAGService()
    rag.vectorstore = FakeVectorStore(sample_documents)

    monkeypatch.setattr(
        "app.rag_service.doc_matches_filters",
        lambda doc, filters: False,
    )

    docs = rag.retrieve("architecture", k=2)

    assert len(docs) == 2
    assert docs[0].metadata["title"] == "Expo Archi"


def test_format_docs_empty(patched_rag):
    rag = RAGService()

    result = rag.format_docs([])

    assert result == "Aucun événement trouvé."


def test_format_docs_returns_structured_text(patched_rag, sample_documents):
    rag = RAGService()

    result = rag.format_docs(sample_documents)

    assert "Événement 1" in result
    assert "Titre : Expo Archi" in result
    assert "Ville : Montpellier" in result
    assert "Contenu : Exposition architecture à Montpellier" in result


def test_build_full_context_without_memory(patched_rag, sample_documents):
    rag = RAGService()

    result = rag.build_full_context("question test", sample_documents)

    assert "context" in result
    assert "memory_context" in result
    assert result["memory_context"] == "Aucun souvenir pertinent trouvé."


def test_build_full_context_with_memory(patched_rag, sample_documents):
    rag = RAGService()
    rag.memory_service.build_memory_context = lambda question: "Souvenir utile"

    result = rag.build_full_context("question test", sample_documents)

    assert result["memory_context"] == "Souvenir utile"


def test_generate_uses_chain(patched_rag, sample_documents):
    rag = RAGService()
    rag.chain = FakeChain("Réponse générée")

    result = rag.generate(
        "question test",
        sample_documents,
        current_date="2026-03-18",
    )

    assert result == "Réponse générée"


def test_generate_returns_fallback_when_no_docs(patched_rag):
    rag = RAGService()

    result = rag.generate(
        "question test",
        [],
        current_date="2026-03-18",
    )

    assert result == rag.FALLBACK_NO_RESULT_MESSAGE


def test_to_retrieved_document(patched_rag, sample_documents):
    rag = RAGService()

    retrieved = rag.to_retrieved_document(sample_documents[0])

    assert retrieved.title == "Expo Archi"
    assert retrieved.city == "Montpellier"
    assert retrieved.url == "http://test.com"
    assert retrieved.score is None


def test_ask_empty_question_raises(patched_rag):
    rag = RAGService()

    with pytest.raises(ValueError, match="La question ne peut pas être vide"):
        rag.ask("   ")


def test_ask_choice_result_branch(patched_rag):
    rag = RAGService()

    rag.memory_service.build_choice_answer = lambda question: {
        "question": question,
        "answer": "Voici l'événement correspondant à votre choix : ...",
        "documents": [
            {
                "title": "Expo choix",
                "location_name": "Musée",
                "city": "Montpellier",
                "region": "Occitanie",
                "first_date": "2026-03-01",
                "last_date": "2026-03-01",
                "event_type": "Exposition",
                "url": "http://choix.com",
                "score": None,
            }
        ],
    }

    response = rag.ask("choix 1")

    assert response.question == "choix 1"
    assert response.answer.startswith("Voici l'événement correspondant")
    assert response.n_docs == 1
    assert response.documents[0].title == "Expo choix"
    assert len(rag.memory_service.entries) == 0


def test_ask_exact_memory_branch(patched_rag):
    rag = RAGService()

    rag.memory_service.find_exact_question = lambda question: {
        "question": question,
        "answer": "Réponse retrouvée en mémoire",
        "documents": [
            {
                "title": "Expo mémoire",
                "location_name": "Lieu mémoire",
                "city": "Montpellier",
                "region": "Occitanie",
                "first_date": "2026-03-02",
                "last_date": "2026-03-02",
                "event_type": "Exposition",
                "url": "http://memoire.com",
                "score": None,
            }
        ],
    }

    response = rag.ask("question déjà posée")

    assert response.answer == "Réponse retrouvée en mémoire"
    assert response.n_docs == 1
    assert response.documents[0].title == "Expo mémoire"
    assert len(rag.memory_service.entries) == 0


def test_ask_normal_rag_pipeline(patched_rag, sample_documents):
    rag = RAGService()
    rag.vectorstore = FakeVectorStore(sample_documents)
    rag.chain = FakeChain("Réponse pipeline RAG")

    response = rag.ask("Je cherche une exposition à Montpellier", k=2)

    assert response.question == "Je cherche une exposition à Montpellier"
    assert response.answer == "Réponse pipeline RAG"
    assert response.n_docs == 2
    assert response.documents[0].title == "Expo Archi"
    assert len(rag.memory_service.entries) == 1


def test_ask_normal_pipeline_with_default_k(patched_rag, sample_documents):
    rag = RAGService(default_k=1)
    rag.vectorstore = FakeVectorStore(sample_documents)
    rag.chain = FakeChain("Réponse avec default_k")

    response = rag.ask("architecture")

    assert response.answer == "Réponse avec default_k"
    assert response.n_docs == 1


def test_load_index_success(patched_rag, sample_documents, tmp_path):
    rag = RAGService(index_dir=tmp_path / "faiss_index")
    rag.index_dir.mkdir(parents=True, exist_ok=True)

    fake_store = FakeVectorStore(sample_documents)

    def fake_load_local(path, embeddings, allow_dangerous_deserialization=True):
        return fake_store

    patched_rag.setattr("app.rag_service.FAISS.load_local", fake_load_local)

    rag.load_index()

    assert rag.vectorstore is fake_store
    assert rag.is_index_loaded() is True