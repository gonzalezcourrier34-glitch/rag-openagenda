from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import (
    _initialize_rag_service,
    _raise_http_from_exception,
    app,
)
from app.security import require_api_key


app.dependency_overrides[require_api_key] = lambda: "test-key"


class FakeMemoryService:
    def reset_memory(self) -> None:
        pass


class FakeRAGAsk:
    def __init__(self):
        self.memory_service = FakeMemoryService()

    def ask(self, question, session_id="default"):
        return {
            "question": question,
            "answer": "Réponse test",
            "n_docs": 1,
            "documents": [
                {
                    "title": "Expo test",
                    "location_name": "Musée",
                    "city": "Montpellier",
                    "region": "Occitanie",
                    "first_date": "2026-03-01",
                    "last_date": "2026-03-01",
                    "event_type": "Exposition",
                    "url": "http://test.com",
                    "price_info": "gratuit",
                    "is_free": True,
                    "keywords_title": [],
                    "score": None,
                }
            ],
            "session_id": session_id,
        }


class FakeRAGAskDebug:
    def __init__(self):
        self.memory_service = FakeMemoryService()

    def ask_debug(self, question, session_id="default"):
        return {
            "question": question,
            "effective_question": question,
            "session_id": session_id,
            "history": [],
            "answer": "Réponse debug",
            "n_docs": 1,
            "documents": [
                {
                    "title": "Expo test",
                    "location_name": "Musée",
                    "city": "Montpellier",
                    "region": "Occitanie",
                    "first_date": "2026-03-01",
                    "last_date": "2026-03-01",
                    "event_type": "Exposition",
                    "url": "http://test.com",
                    "price_info": "gratuit",
                    "is_free": True,
                    "keywords_title": [],
                    "score": None,
                }
            ],
            "retrieved_contexts": ["Contenu test"],
            "fallback_used": False,
            "filter_debug": {"filters": {}, "n_input_docs": 1},
            "retrieval_debug": {},
        }

class FakeRAGHealth:
    def __init__(self):
        self.memory_service = FakeMemoryService()

    def is_index_loaded(self):
        return True


class FakeRAGRebuild:
    def __init__(self):
        self.memory_service = FakeMemoryService()
        self.zone = None
        self.scope = None
        self.documents = None

    def rebuild_index(self, documents):
        self.documents = documents
        return 2


def test_raise_http_from_exception_file_not_found():
    with pytest.raises(HTTPException) as exc_info:
        _raise_http_from_exception(
            FileNotFoundError("index absent"),
            user_log_message="Erreur utilisateur",
            server_log_message="Erreur serveur",
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "index absent"


def test_raise_http_from_exception_value_error():
    with pytest.raises(HTTPException) as exc_info:
        _raise_http_from_exception(
            ValueError("question invalide"),
            user_log_message="Erreur utilisateur",
            server_log_message="Erreur serveur",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "question invalide"


def test_raise_http_from_exception_unexpected():
    with pytest.raises(HTTPException) as exc_info:
        _raise_http_from_exception(
            RuntimeError("boom"),
            user_log_message="Erreur utilisateur",
            server_log_message="Erreur serveur",
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Erreur interne du serveur."


def test_initialize_rag_service_returns_when_already_loaded(monkeypatch):
    class FakeRAG:
        def __init__(self):
            self.memory_service = FakeMemoryService()
            self.index_dir = Path("unused")

        def is_index_loaded(self):
            return True

    monkeypatch.setattr("app.main.rag_service", FakeRAG())

    _initialize_rag_service()


def test_initialize_rag_service_loads_existing_index(monkeypatch, tmp_path):
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "stub.faiss").write_text("ok", encoding="utf-8")

    class FakeRAG:
        def __init__(self):
            self.memory_service = FakeMemoryService()
            self.index_dir = index_dir
            self.documents = None
            self.zone = None
            self.scope = None
            self.loaded = False

        def is_index_loaded(self):
            return False

        def load_index(self):
            self.loaded = True

        def set_documents(self, documents):
            self.documents = documents

    fake_rag = FakeRAG()

    monkeypatch.setattr("app.main.rag_service", fake_rag)
    monkeypatch.setattr("app.main.load_documents", lambda zone, scope: ["doc1", "doc2"])

    _initialize_rag_service()

    assert fake_rag.loaded is True
    assert fake_rag.documents == ["doc1", "doc2"]
    assert isinstance(fake_rag.zone, str)
    assert isinstance(fake_rag.scope, str)


def test_initialize_rag_service_warns_when_no_documents(monkeypatch, tmp_path):
    class FakeRAG:
        def __init__(self):
            self.memory_service = FakeMemoryService()
            self.index_dir = tmp_path / "missing_index"
            self.zone = None
            self.scope = None

        def is_index_loaded(self):
            return False

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    monkeypatch.setattr("app.main.load_documents", lambda zone, scope: [])

    _initialize_rag_service()


def test_initialize_rag_service_builds_index_when_documents_exist(monkeypatch, tmp_path):
    class FakeRAG:
        def __init__(self):
            self.memory_service = FakeMemoryService()
            self.index_dir = tmp_path / "missing_index"
            self.zone = None
            self.scope = None
            self.documents = None

        def is_index_loaded(self):
            return False

        def rebuild_index(self, documents):
            self.documents = documents
            return len(documents)

    fake_rag = FakeRAG()

    monkeypatch.setattr("app.main.rag_service", fake_rag)
    monkeypatch.setattr("app.main.load_documents", lambda zone, scope: ["doc1", "doc2", "doc3"])

    _initialize_rag_service()

    assert fake_rag.zone is not None
    assert fake_rag.scope is not None
    assert fake_rag.documents == ["doc1", "doc2", "doc3"]


def test_root_redirects_to_docs():
    client = TestClient(app)

    response = client.get("/", follow_redirects=False)

    assert response.status_code in (307, 308)
    assert response.headers["location"] == "/docs"


def test_health_ok(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGHealth())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["index_loaded"] is True


def test_ask_success(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGAsk())
    client = TestClient(app)

    response = client.post("/ask", json={"question": "test", "session_id": "test-session"})

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "test"
    assert data["answer"] == "Réponse test"
    assert data["n_docs"] == 1
    assert data["session_id"] == "test-session"
    assert data["documents"][0]["title"] == "Expo test"


def test_ask_file_not_found(monkeypatch):
    class FakeRAG:
        def __init__(self):
            self.memory_service = FakeMemoryService()

        def ask(self, question, session_id="default"):
            raise FileNotFoundError("index absent")

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    client = TestClient(app)

    response = client.post("/ask", json={"question": "test", "session_id": "test-session"})

    assert response.status_code == 503
    assert "index absent" in response.json()["detail"]


def test_ask_value_error(monkeypatch):
    class FakeRAG:
        def __init__(self):
            self.memory_service = FakeMemoryService()

        def ask(self, question, session_id="default"):
            raise ValueError("question invalide")

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    client = TestClient(app)

    response = client.post("/ask", json={"question": "test", "session_id": "test-session"})

    assert response.status_code == 400
    assert "question invalide" in response.json()["detail"]


def test_ask_unexpected_error(monkeypatch):
    class FakeRAG:
        def __init__(self):
            self.memory_service = FakeMemoryService()

        def ask(self, question, session_id="default"):
            raise RuntimeError("boom")

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    client = TestClient(app)

    response = client.post("/ask", json={"question": "test", "session_id": "test-session"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne du serveur."


def test_ask_debug_success(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGAskDebug())
    client = TestClient(app)

    response = client.post(
        "/ask/debug",
        json={"question": "test debug", "session_id": "debug-session"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "test debug"
    assert data["effective_question"] == "test debug"
    assert data["session_id"] == "debug-session"
    assert data["history"] == []
    assert data["answer"] == "Réponse debug"
    assert data["n_docs"] == 1
    assert data["retrieved_contexts"] == ["Contenu test"]
    assert data["documents"][0]["title"] == "Expo test"


def test_ask_debug_file_not_found(monkeypatch):
    class FakeRAG:
        def __init__(self):
            self.memory_service = FakeMemoryService()

        def ask_debug(self, question, session_id="default"):
            raise FileNotFoundError("index absent")

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    client = TestClient(app)

    response = client.post(
        "/ask/debug",
        json={"question": "test debug", "session_id": "debug-session"},
    )

    assert response.status_code == 503
    assert "index absent" in response.json()["detail"]


def test_ask_debug_value_error(monkeypatch):
    class FakeRAG:
        def __init__(self):
            self.memory_service = FakeMemoryService()

        def ask_debug(self, question, session_id="default"):
            raise ValueError("question invalide")

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    client = TestClient(app)

    response = client.post(
        "/ask/debug",
        json={"question": "test debug", "session_id": "debug-session"},
    )

    assert response.status_code == 400
    assert "question invalide" in response.json()["detail"]


def test_ask_debug_unexpected_error(monkeypatch):
    class FakeRAG:
        def __init__(self):
            self.memory_service = FakeMemoryService()

        def ask_debug(self, question, session_id="default"):
            raise RuntimeError("boom")

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    client = TestClient(app)

    response = client.post(
        "/ask/debug",
        json={"question": "test debug", "session_id": "debug-session"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne du serveur."


def test_ask_debug_empty_question_returns_422():
    client = TestClient(app)

    response = client.post("/ask/debug", json={"question": "", "session_id": "debug-session"})

    assert response.status_code == 422


def test_rebuild_success(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGRebuild())
    monkeypatch.setattr("app.main.load_documents", lambda zone, scope: ["doc1", "doc2"])
    client = TestClient(app)

    response = client.post(
        "/rebuild",
        json={"zone": "Montpellier", "scope": "city"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "Index reconstruit avec 2 documents."
    assert data["n_docs_indexed"] == 2


def test_rebuild_no_documents(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGRebuild())
    monkeypatch.setattr("app.main.load_documents", lambda zone, scope: [])
    client = TestClient(app)

    response = client.post(
        "/rebuild",
        json={"zone": "Montpellier", "scope": "city"},
    )

    assert response.status_code == 400
    assert "Aucun document" in response.json()["detail"]


def test_rebuild_uses_default_zone_and_scope(monkeypatch):
    captured = {}

    def fake_load_documents(*, zone, scope):
        captured["zone"] = zone
        captured["scope"] = scope
        return ["doc1", "doc2"]

    monkeypatch.setattr("app.main.rag_service", FakeRAGRebuild())
    monkeypatch.setattr("app.main.load_documents", fake_load_documents)
    client = TestClient(app)

    response = client.post("/rebuild", json={})

    assert response.status_code == 200
    assert isinstance(captured["zone"], str)
    assert isinstance(captured["scope"], str)


def test_rebuild_value_error(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGRebuild())

    def fake_load_documents(zone, scope):
        raise ValueError("zone invalide")

    monkeypatch.setattr("app.main.load_documents", fake_load_documents)
    client = TestClient(app)

    response = client.post(
        "/rebuild",
        json={"zone": "Montpellier", "scope": "city"},
    )

    assert response.status_code == 400
    assert "zone invalide" in response.json()["detail"]


def test_rebuild_unexpected_error(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGRebuild())

    def fake_load_documents(zone, scope):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.main.load_documents", fake_load_documents)
    client = TestClient(app)

    response = client.post(
        "/rebuild",
        json={"zone": "Montpellier", "scope": "city"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne du serveur."