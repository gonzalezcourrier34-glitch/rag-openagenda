from fastapi.testclient import TestClient

from app.main import app
from app.security import require_api_key


class FakeRAGService:
    def ask(self, question: str, k: int | None = None):
        from app.schemas import AskResponse, RetrievedDocument

        return AskResponse(
            question=question,
            answer="Voici une exposition d'architecture à Montpellier.",
            n_docs=1,
            documents=[
                RetrievedDocument(
                    title="Expo Archi",
                    city="Montpellier",
                    location_name="Musée X",
                    region="Occitanie",
                    first_date="2026-03-01",
                    last_date="2026-03-10",
                    event_type="Exposition",
                    url="http://test.com",
                    price_info="gratuit",
                    is_free=True,
                    keywords_title=[],
                    score=None,
                )
            ],
        )

    def ask_debug(self, question: str):
        return {
            "question": question,
            "answer": "Réponse debug",
            "n_docs": 1,
            "documents": [
                {
                    "title": "Expo Archi",
                    "location_name": "Musée X",
                    "city": "Montpellier",
                    "region": "Occitanie",
                    "first_date": "2026-03-01",
                    "last_date": "2026-03-10",
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
            "retrieval_debug": [],
        }

    def rebuild_index(self, documents):
        self.documents = documents
        return 3

    def is_index_loaded(self):
        return True


class FakeRAGServiceNoIndex:
    def is_index_loaded(self):
        return False


app.dependency_overrides[require_api_key] = lambda: "test-key"


def test_root_redirects_to_docs():
    client = TestClient(app)

    response = client.get("/", follow_redirects=False)

    assert response.status_code in (307, 308)
    assert response.headers["location"] == "/docs"


def test_health_ok(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGService())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["index_loaded"] is True


def test_health_index_not_loaded(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGServiceNoIndex())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["index_loaded"] is False


def test_ask(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGService())
    client = TestClient(app)

    question = "Je cherche une exposition d'architecture à Montpellier"

    response = client.post(
        "/ask",
        json={"question": question},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == question
    assert data["answer"] == "Voici une exposition d'architecture à Montpellier."
    assert data["n_docs"] == 1
    assert len(data["documents"]) == 1
    assert data["documents"][0]["title"] == "Expo Archi"
    assert data["documents"][0]["city"] == "Montpellier"


def test_ask_value_error(monkeypatch):
    class FakeRAG:
        def ask(self, question: str, k: int | None = None):
            raise ValueError("question invalide")

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    client = TestClient(app)

    response = client.post("/ask", json={"question": "bad question"})

    assert response.status_code == 400
    assert "question invalide" in response.json()["detail"]


def test_ask_file_not_found(monkeypatch):
    class FakeRAG:
        def ask(self, question: str, k: int | None = None):
            raise FileNotFoundError("index absent")

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    client = TestClient(app)

    response = client.post("/ask", json={"question": "test"})

    assert response.status_code == 503
    assert "index absent" in response.json()["detail"]


def test_ask_unexpected_error(monkeypatch):
    class FakeRAG:
        def ask(self, question: str, k: int | None = None):
            raise Exception("boom")

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    client = TestClient(app)

    response = client.post("/ask", json={"question": "test"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne du serveur."


def test_ask_debug(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGService())
    client = TestClient(app)

    response = client.post(
        "/ask/debug",
        json={"question": "debug exposition"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "debug exposition"
    assert data["answer"] == "Réponse debug"
    assert data["n_docs"] == 1
    assert data["retrieved_contexts"] == ["Contenu test"]
    assert data["documents"][0]["title"] == "Expo Archi"


def test_rebuild(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGService())
    monkeypatch.setattr(
        "app.main.load_documents",
        lambda *args, **kwargs: ["fake_doc_1", "fake_doc_2", "fake_doc_3"],
    )
    client = TestClient(app)

    response = client.post(
        "/rebuild",
        json={"zone": "Montpellier", "scope": "city"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["n_docs_indexed"] == 3


def test_rebuild_no_documents(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGService())
    monkeypatch.setattr(
        "app.main.load_documents",
        lambda *args, **kwargs: [],
    )
    client = TestClient(app)

    response = client.post(
        "/rebuild",
        json={"zone": "Montpellier", "scope": "city"},
    )

    assert response.status_code == 400
    assert "Aucun document" in response.json()["detail"]


def test_rebuild_value_error(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGService())

    def fake_load_documents(*args, **kwargs):
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
    monkeypatch.setattr("app.main.rag_service", FakeRAGService())

    def fake_load_documents(*args, **kwargs):
        raise Exception("boom")

    monkeypatch.setattr("app.main.load_documents", fake_load_documents)
    client = TestClient(app)

    response = client.post(
        "/rebuild",
        json={"zone": "Montpellier", "scope": "city"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne du serveur."