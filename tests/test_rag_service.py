import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.rag_service import RAGService


# -------------------------------------------------------------------------
# Fakes
# -------------------------------------------------------------------------

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
    def __init__(self, documents=None, results_with_scores=None):
        self.documents = documents or []
        self.results_with_scores = results_with_scores or []

    def save_local(self, path):
        return None

    def similarity_search_with_score(self, query, k=3):
        return self.results_with_scores[:k]


class FakeFilterService:
    def __init__(self, docs=None):
        self.docs = docs or []

    def filter_documents_with_debug(self, question, docs, default_city=None):
        return {
            "filters": {"question": question, "default_city": default_city},
            "n_input_docs": len(docs),
            "n_after_city": len(self.docs),
            "n_after_type": len(self.docs),
            "n_after_music": len(self.docs),
            "n_after_cultural": len(self.docs),
            "n_after_audience": len(self.docs),
            "n_after_duration": len(self.docs),
            "n_after_date": len(self.docs),
            "n_after_price": len(self.docs),
            "docs": self.docs,
        }


class FakeRetrievalService:
    def __init__(self, ranked_docs=None, retrieval_debug=None):
        self.ranked_docs = ranked_docs
        self.retrieval_debug = retrieval_debug or []

    def rank_documents(self, question, raw_docs, top_k):
        if self.ranked_docs is not None:
            return self.ranked_docs[:top_k]
        return raw_docs[:top_k]

    def rank_documents_with_scores(self, question, raw_docs, top_k):
        if self.retrieval_debug:
            return self.retrieval_debug
        return [
            {
                "title": (doc.metadata or {}).get("title", ""),
                "final_score": (doc.metadata or {}).get("final_score"),
                "vector_score": (doc.metadata or {}).get("vector_score"),
            }
            for doc in raw_docs[:top_k]
        ]


class FakeTraceService:
    def __init__(self, *args, **kwargs):
        self.payloads = []

    def write_trace(self, payload):
        self.payloads.append(payload)


class FakeFAISSFactory:
    def __init__(self):
        self.created_from_documents = None
        self.loaded_store = None

    def from_documents(self, documents, embedding=None, embeddings=None, **kwargs):
        self.created_from_documents = documents
        return FakeVectorStore(documents=documents)

    def load_local(
        self,
        folder_path=None,
        embeddings=None,
        allow_dangerous_deserialization=True,
        **kwargs,
    ):
        return self.loaded_store


class FakeMemoryService:
    def __init__(self):
        self.messages = []

    def format_history_for_prompt(self, session_id: str) -> str:
        return ""

    def append_message(self, session_id: str, role: str, content: str) -> None:
        self.messages.append(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
            }
        )

    def get_history(self, session_id: str):
        return []

    def reset_memory(self) -> None:
        pass


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Contenu exposition architecture Montpellier",
            metadata={
                "title": "Expo Archi",
                "description": "Une exposition d'architecture",
                "location_name": "Musée X",
                "city": "Montpellier",
                "region": "Occitanie",
                "first_date": "2026-03-01",
                "last_date": "2026-03-10",
                "event_type": "Exposition",
                "canonical_event_type": "exposition",
                "music_genre": "",
                "price_info": "gratuit",
                "is_free": True,
                "keywords_title": ["expo", "archi"],
                "url": "http://test.com",
                "source_url": "http://test.com",
                "content_quality": 8,
                "search_text": "exposition architecture montpellier musee x",
            },
        ),
        Document(
            page_content="Contenu concert jazz Sète",
            metadata={
                "title": "Jazz Night",
                "description": "Concert de jazz",
                "location_name": "Salle Y",
                "city": "Sète",
                "region": "Occitanie",
                "first_date": "2026-03-15",
                "last_date": "2026-03-15",
                "event_type": "Concert",
                "canonical_event_type": "concert",
                "music_genre": "jazz",
                "price_info": "payant",
                "is_free": False,
                "keywords_title": ["jazz"],
                "url": "http://concert.com",
                "source_url": "http://concert.com",
                "content_quality": 8,
                "search_text": "concert jazz sete salle y",
            },
        ),
    ]


@pytest.fixture
def patched_rag(monkeypatch):
    fake_faiss = FakeFAISSFactory()

    monkeypatch.setattr("app.rag_service.MistralAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr("app.rag_service.ChatMistralAI", fake_llm_factory)
    monkeypatch.setattr("app.rag_service.FAISS.from_documents", fake_faiss.from_documents)
    monkeypatch.setattr("app.rag_service.FAISS.load_local", fake_faiss.load_local)
    monkeypatch.setattr("app.rag_service.FilterService", FakeFilterService)
    monkeypatch.setattr("app.rag_service.RetrievalService", FakeRetrievalService)
    monkeypatch.setattr("app.rag_service.TraceService", FakeTraceService)
    monkeypatch.setattr("app.rag_service.MemoryService", FakeMemoryService)

    return fake_faiss


# -------------------------------------------------------------------------
# Base
# -------------------------------------------------------------------------

def test_set_documents(patched_rag, sample_documents):
    rag = RAGService()
    rag.set_documents(sample_documents)

    assert rag.documents == sample_documents


def test_is_index_loaded_false_by_default(patched_rag):
    rag = RAGService()

    assert rag.is_index_loaded() is False


# -------------------------------------------------------------------------
# Construction / chargement index
# -------------------------------------------------------------------------

def test_build_index(patched_rag, sample_documents):
    rag = RAGService(documents=sample_documents)

    n_docs = rag.build_index()

    assert n_docs == 2
    assert rag.is_index_loaded() is True
    assert patched_rag.created_from_documents is not None
    assert len(patched_rag.created_from_documents) == 2
    assert (
        patched_rag.created_from_documents[0].page_content
        == sample_documents[0].metadata["search_text"]
    )


def test_build_index_without_documents_raises(patched_rag):
    rag = RAGService(documents=[])

    with pytest.raises(ValueError, match="aucun document n'est disponible"):
        rag.build_index()


def test_load_index_missing_dir_raises(patched_rag, tmp_path):
    rag = RAGService(index_dir=tmp_path / "missing_index")

    with pytest.raises(FileNotFoundError, match="Index FAISS introuvable"):
        rag.load_index()


def test_load_index_success(patched_rag, sample_documents, tmp_path):
    rag = RAGService(index_dir=tmp_path / "faiss_index")
    rag.index_dir.mkdir(parents=True, exist_ok=True)
    (rag.index_dir / "stub.faiss").write_text("ok", encoding="utf-8")

    fake_store = FakeVectorStore(documents=sample_documents)
    patched_rag.loaded_store = fake_store

    rag.load_index()

    assert rag.vectorstore is fake_store
    assert rag.is_index_loaded() is True


def test_ensure_index_ready_loads_existing_index_and_documents(
    patched_rag, sample_documents, monkeypatch, tmp_path
):
    rag = RAGService(index_dir=tmp_path / "faiss_index", documents=[])
    rag.index_dir.mkdir(parents=True, exist_ok=True)
    (rag.index_dir / "stub.faiss").write_text("ok", encoding="utf-8")

    fake_store = FakeVectorStore(documents=sample_documents)
    patched_rag.loaded_store = fake_store

    monkeypatch.setattr("app.rag_service.load_documents", lambda zone, scope: sample_documents)

    rag.ensure_index_ready()

    assert rag.vectorstore is fake_store
    assert rag.documents == sample_documents


def test_ensure_index_ready_builds_index_when_missing(
    patched_rag, sample_documents, monkeypatch, tmp_path
):
    rag = RAGService(index_dir=tmp_path / "faiss_index", documents=[])

    monkeypatch.setattr("app.rag_service.load_documents", lambda zone, scope: sample_documents)

    rag.ensure_index_ready()

    assert rag.is_index_loaded() is True
    assert rag.documents == sample_documents


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def test_build_docs_for_vector_index_uses_search_text(patched_rag, sample_documents):
    rag = RAGService()

    docs_for_index = rag._build_docs_for_vector_index(sample_documents)

    assert len(docs_for_index) == 2
    assert docs_for_index[0].page_content == sample_documents[0].metadata["search_text"]
    assert docs_for_index[1].page_content == sample_documents[1].metadata["search_text"]


def test_attach_vector_scores_sets_metadata(patched_rag, sample_documents):
    rag = RAGService()

    docs = rag._attach_vector_scores(
        [
            (sample_documents[0], 0.12),
            (sample_documents[1], 0.56),
        ]
    )

    assert docs[0].metadata["vector_score"] == 0.12
    assert docs[1].metadata["vector_score"] == 0.56


def test_is_reliable_document_true(patched_rag, sample_documents):
    rag = RAGService()

    assert rag._is_reliable_document(sample_documents[0]) is True


def test_is_reliable_document_false_when_too_poor(patched_rag):
    rag = RAGService()
    poor_doc = Document(
        page_content="x",
        metadata={
            "title": "",
            "first_date": "",
            "location_name": "",
            "city": "",
            "source_url": "",
            "content_quality": 1,
        },
    )

    assert rag._is_reliable_document(poor_doc) is False


def test_post_filter_ranked_docs_prefers_reliable_docs(patched_rag, sample_documents):
    rag = RAGService(top_k_final=1)

    poor_doc = Document(
        page_content="x",
        metadata={
            "title": "",
            "first_date": "",
            "location_name": "",
            "city": "",
            "source_url": "",
            "content_quality": 1,
        },
    )

    docs = rag._post_filter_ranked_docs([poor_doc, sample_documents[0]])

    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Expo Archi"


# -------------------------------------------------------------------------
# Retrieval
# -------------------------------------------------------------------------

def test_retrieve_empty_question_raises(patched_rag):
    rag = RAGService()

    with pytest.raises(ValueError, match="La question utilisateur ne peut pas être vide"):
        rag.retrieve("   ")


def test_run_vector_retrieval_returns_scored_docs(patched_rag, sample_documents, monkeypatch):
    rag = RAGService(top_k_retrieval=2)

    fake_local_vs = FakeVectorStore(
        results_with_scores=[
            (sample_documents[0], 0.11),
            (sample_documents[1], 0.22),
        ]
    )
    monkeypatch.setattr(rag, "_build_local_vectorstore", lambda docs: fake_local_vs)

    docs = rag._run_vector_retrieval("architecture", sample_documents)

    assert len(docs) == 2
    assert docs[0].metadata["vector_score"] == 0.11
    assert docs[1].metadata["vector_score"] == 0.22


def test_retrieve_returns_docs_from_pipeline(patched_rag, sample_documents, monkeypatch):
    rag = RAGService()
    monkeypatch.setattr(
        rag,
        "_run_pipeline",
        lambda question, session_id="default": {"docs": sample_documents[:1]},
    )

    docs = rag.retrieve("architecture")

    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Expo Archi"


def test_prefilter_with_fallback_uses_fallback_when_main_empty(patched_rag, sample_documents):
    rag = RAGService(documents=sample_documents)

    main_debug = {
        "filters": {},
        "n_input_docs": 2,
        "n_after_city": 0,
        "n_after_type": 0,
        "n_after_music": 0,
        "n_after_cultural": 0,
        "n_after_audience": 0,
        "n_after_duration": 0,
        "n_after_date": 0,
        "n_after_price": 0,
        "docs": [],
    }
    fallback_debug = {
        "filters": {},
        "n_input_docs": 2,
        "n_after_city": 1,
        "n_after_type": 1,
        "n_after_music": 1,
        "n_after_cultural": 1,
        "n_after_audience": 1,
        "n_after_duration": 1,
        "n_after_date": 1,
        "n_after_price": 1,
        "docs": [sample_documents[0]],
    }

    class LocalFilterService:
        def __init__(self):
            self.calls = 0

        def filter_documents_with_debug(self, question, docs, default_city=None):
            self.calls += 1
            return main_debug if self.calls == 1 else fallback_debug

    rag.filter_service = LocalFilterService()

    filter_debug, fallback_filter_debug, prefiltered_docs, fallback_used = rag._prefilter_with_fallback(
        "question test"
    )

    assert fallback_used is True
    assert filter_debug["docs"] == []
    assert fallback_filter_debug is not None
    assert len(prefiltered_docs) == 1
    assert prefiltered_docs[0].metadata["title"] == "Expo Archi"


# -------------------------------------------------------------------------
# Formatage
# -------------------------------------------------------------------------

def test_format_doc_returns_structured_text(patched_rag, sample_documents):
    rag = RAGService()

    result = rag.format_doc(sample_documents[0], index=1)

    assert "Événement 1" in result
    assert "Titre : Expo Archi" in result
    assert "Ville : Montpellier" in result
    assert "Description : Une exposition d'architecture" in result


def test_format_docs_empty_returns_empty_string(patched_rag):
    rag = RAGService()

    result = rag.format_docs([])

    assert result == ""


def test_format_docs_returns_structured_text(patched_rag, sample_documents):
    rag = RAGService()

    result = rag.format_docs(sample_documents)

    assert "Événement 1" in result
    assert "Titre : Expo Archi" in result
    assert "Événement 2" in result
    assert "Titre : Jazz Night" in result


def test_format_docs_list_returns_list(patched_rag, sample_documents):
    rag = RAGService()

    result = rag.format_docs_list(sample_documents)

    assert isinstance(result, list)
    assert len(result) == 2
    assert "Événement 1" in result[0]


# -------------------------------------------------------------------------
# Conversion API
# -------------------------------------------------------------------------

def test_build_retrieved_document(patched_rag, sample_documents):
    rag = RAGService()

    retrieved = rag._build_retrieved_document(sample_documents[0])

    assert retrieved.title == "Expo Archi"
    assert retrieved.city == "Montpellier"
    assert retrieved.url == "http://test.com"
    assert retrieved.price_info == "gratuit"
    assert retrieved.is_free is True
    assert retrieved.keywords_title == ["expo", "archi"]
    assert retrieved.score is None


def test_build_retrieved_documents_returns_list(patched_rag, sample_documents):
    rag = RAGService()

    retrieved_docs = rag.build_retrieved_documents(sample_documents)

    assert len(retrieved_docs) == 2
    assert retrieved_docs[0].title == "Expo Archi"


# -------------------------------------------------------------------------
# Génération
# -------------------------------------------------------------------------

def test_generate_answer_returns_empty_answer_when_no_docs(patched_rag):
    rag = RAGService()

    result = rag.generate_answer("question test", [], session_id="test-session")

    assert result == rag.EMPTY_ANSWER


def test_generate_answer_uses_chain_when_docs_exist(patched_rag, sample_documents):
    rag = RAGService()
    rag.chain = FakeChain("Réponse générée")

    result = rag.generate_answer("question test", sample_documents, session_id="test-session")

    assert result == "Réponse générée"


def test_generate_answer_returns_empty_answer_when_chain_returns_blank(
    patched_rag, sample_documents
):
    rag = RAGService()
    rag.chain = FakeChain("   ")

    result = rag.generate_answer("question test", sample_documents, session_id="test-session")

    assert result == rag.EMPTY_ANSWER


# -------------------------------------------------------------------------
# Pipeline complet
# -------------------------------------------------------------------------

def test_ask_empty_question_raises(patched_rag):
    rag = RAGService()

    with pytest.raises(ValueError, match="La question utilisateur ne peut pas être vide"):
        rag.ask("   ")


def test_run_pipeline_success(patched_rag, sample_documents, monkeypatch):
    rag = RAGService(documents=sample_documents, top_k_retrieval=2, top_k_final=1)
    rag.chain = FakeChain("Réponse pipeline")
    rag.filter_service = FakeFilterService(docs=sample_documents)
    rag.retrieval_service = FakeRetrievalService(ranked_docs=[sample_documents[0]])
    rag.trace_service = FakeTraceService()
    rag.memory_service = FakeMemoryService()
    rag.vectorstore = FakeVectorStore()

    monkeypatch.setattr(
        rag,
        "_build_local_vectorstore",
        lambda docs: FakeVectorStore(results_with_scores=[(sample_documents[0], 0.15)]),
    )
    monkeypatch.setattr(
        rag,
        "_rewrite_question_with_history",
        lambda question, session_id="default": question,
    )

    result = rag._run_pipeline("Je cherche une exposition", session_id="test-session")

    assert result["question"] == "Je cherche une exposition"
    assert result["effective_question"] == "Je cherche une exposition"
    assert result["session_id"] == "test-session"
    assert result["history"] == []
    assert result["answer"] == "Réponse pipeline"
    assert len(result["docs"]) == 1
    assert result["docs"][0].metadata["title"] == "Expo Archi"
    assert len(result["retrieved_documents"]) == 1
    assert len(result["retrieved_contexts"]) == 1
    assert result["fallback_used"] is False
    assert len(rag.trace_service.payloads) == 1


def test_ask_returns_ask_response(patched_rag, sample_documents, monkeypatch):
    rag = RAGService()

    monkeypatch.setattr(
        rag,
        "_run_pipeline",
        lambda question, session_id="default": {
            "question": question,
            "answer": "Réponse finale",
            "docs": sample_documents[:1],
            "retrieved_documents": rag.build_retrieved_documents(sample_documents[:1]),
            "session_id": session_id,
        },
    )

    response = rag.ask("architecture", session_id="test-session")

    assert response.question == "architecture"
    assert response.answer == "Réponse finale"
    assert response.n_docs == 1
    assert response.session_id == "test-session"
    assert response.documents[0].title == "Expo Archi"


def test_ask_debug_returns_debug_payload(patched_rag, sample_documents, monkeypatch):
    rag = RAGService()

    monkeypatch.setattr(
        rag,
        "_run_pipeline",
        lambda question, session_id="default": {
            "question": question,
            "effective_question": question,
            "session_id": session_id,
            "history": [],
            "answer": "Réponse debug",
            "docs": sample_documents[:1],
            "retrieved_documents": rag.build_retrieved_documents(sample_documents[:1]),
            "retrieved_contexts": ["Événement 1\nTitre : Expo Archi"],
            "fallback_used": False,
            "filter_debug": {"filters": {}, "n_input_docs": 2},
            "fallback_filter_debug": None,
            "retrieval_debug": {"rows": [{"title": "Expo Archi", "final_score": 0.8}]},
        },
    )

    response = rag.ask_debug("architecture", session_id="test-session")

    assert response["question"] == "architecture"
    assert response["effective_question"] == "architecture"
    assert response["session_id"] == "test-session"
    assert response["history"] == []
    assert response["answer"] == "Réponse debug"
    assert response["n_docs"] == 1
    assert response["documents"][0]["title"] == "Expo Archi"
    assert response["retrieved_contexts"] == ["Événement 1\nTitre : Expo Archi"]
    assert response["fallback_used"] is False
    assert "filter_debug" in response
    assert "retrieval_debug" in response