import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.rag_service import RAGService


# FAKE CLASSES
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


# FIXTURES
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
        lambda question, documents: {},
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


# TESTS
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

    with pytest.raises(ValueError, match="Aucun document"):
        rag.build_index()


def test_load_index_missing_dir_raises(patched_rag, tmp_path):
    rag = RAGService(index_dir=tmp_path / "missing_index")

    with pytest.raises(FileNotFoundError):
        rag.load_index()


def test_ensure_index_ready_loads_index_when_needed(patched_rag, sample_documents):
    rag = RAGService()
    fake_store = FakeVectorStore(sample_documents)

    def fake_load_local(*args, **kwargs):
        return fake_store

    patched_rag.setattr("app.rag_service.FAISS.load_local", fake_load_local)

    rag.index_dir.mkdir(parents=True, exist_ok=True)
    rag.ensure_index_ready()

    assert rag.vectorstore is fake_store


# RETRIEVE
def test_retrieve_empty_question_raises(patched_rag):
    rag = RAGService()

    with pytest.raises(ValueError, match="Question vide"):
        rag.retrieve("   ")


def test_retrieve_returns_docs(patched_rag, sample_documents):
    rag = RAGService()
    rag.vectorstore = FakeVectorStore(sample_documents)

    docs = rag.retrieve("architecture", k=1)

    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Expo Archi"


def test_retrieve_returns_empty_when_all_filtered_out(
    patched_rag, sample_documents, monkeypatch
):
    rag = RAGService()
    rag.vectorstore = FakeVectorStore(sample_documents)

    monkeypatch.setattr(
        "app.rag_service.doc_matches_filters",
        lambda doc, filters: False,
    )

    docs = rag.retrieve("architecture", k=2)

    assert docs == []


# FORMAT
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


# GENERATE
def test_generate_uses_chain(patched_rag, sample_documents):
    rag = RAGService()
    rag.chain = FakeChain("Réponse générée")

    result = rag.generate("question", sample_documents)

    assert result == "Réponse générée"


def test_generate_returns_fallback_when_no_docs(patched_rag):
    rag = RAGService()

    result = rag.generate("question", [])

    assert result == rag.FALLBACK_NO_RESULT_MESSAGE


# DTO
def test_to_retrieved_document(patched_rag, sample_documents):
    rag = RAGService()

    doc = rag.to_retrieved_document(sample_documents[0])

    assert doc.title == "Expo Archi"
    assert doc.city == "Montpellier"
    assert doc.url == "http://test.com"


# ASK
def test_ask_empty_question_raises(patched_rag):
    rag = RAGService()

    with pytest.raises(ValueError, match="Question vide"):
        rag.ask("   ")


def test_ask_normal_pipeline(patched_rag, sample_documents):
    rag = RAGService()
    rag.vectorstore = FakeVectorStore(sample_documents)
    rag.chain = FakeChain("Réponse pipeline")

    response = rag.ask("architecture", k=2)

    assert response.answer == "Réponse pipeline"
    assert response.n_docs == 2
    assert len(rag.memory_service.entries) == 1


def test_ask_with_default_k(patched_rag, sample_documents):
    rag = RAGService(default_k=1)
    rag.vectorstore = FakeVectorStore(sample_documents)
    rag.chain = FakeChain("Réponse default_k")

    response = rag.ask("architecture")

    assert response.n_docs == 1