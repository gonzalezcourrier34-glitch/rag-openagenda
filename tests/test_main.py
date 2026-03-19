from fastapi.testclient import TestClient

from app.main import app
from app.security import require_api_key


app.dependency_overrides[require_api_key] = lambda: "test-key"


def test_root_redirects_to_docs():
    client = TestClient(app)

    response = client.get("/", follow_redirects=False)

    assert response.status_code in (307, 308)
    assert response.headers["location"] == "/docs"


def test_health_ok(monkeypatch):
    class FakeRAG:
        def is_index_loaded(self):
            return True

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["index_loaded"] is True


def test_ask_success(monkeypatch):
    class FakeRAG:
        def ask(self, question, k=None):
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
                        "score": None,
                    }
                ],
            }

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    client = TestClient(app)

    response = client.post("/ask", json={"question": "test"})

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "test"
    assert data["answer"] == "Réponse test"
    assert data["n_docs"] == 1
    assert data["documents"][0]["title"] == "Expo test"


def test_ask_file_not_found(monkeypatch):
    class FakeRAG:
        def ask(self, question, k=None):
            raise FileNotFoundError("index absent")

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    client = TestClient(app)

    response = client.post("/ask", json={"question": "test"})

    assert response.status_code == 503
    assert "index absent" in response.json()["detail"]


def test_ask_value_error(monkeypatch):
    class FakeRAG:
        def ask(self, question, k=None):
            raise ValueError("question invalide")

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    client = TestClient(app)

    response = client.post("/ask", json={"question": "test"})

    assert response.status_code == 400
    assert "question invalide" in response.json()["detail"]


def test_ask_unexpected_error(monkeypatch):
    class FakeRAG:
        def ask(self, question, k=None):
            raise Exception("boom")

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    client = TestClient(app)

    response = client.post("/ask", json={"question": "test"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne du serveur."


def test_ask_debug_success(monkeypatch):
    class FakeDoc:
        def __init__(self):
            self.page_content = "Contenu test"
            self.metadata = {"title": "Expo test"}

    class FakeRAG:
        def retrieve(self, question, k=None):
            return [FakeDoc()]

        def generate(self, question, docs, current_date):
            return "Réponse debug"

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    client = TestClient(app)

    response = client.post("/ask/debug", json={"question": "test debug"})

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "test debug"
    assert data["answer"] == "Réponse debug"
    assert data["contexts"] == ["Contenu test"]
    assert data["metadata"] == [{"title": "Expo test"}]


def test_ask_debug_value_error():
    client = TestClient(app)

    response = client.post("/ask/debug", json={"question": ""})

    assert response.status_code == 422


def test_rebuild_success(monkeypatch):
    class FakeRAG:
        def set_documents(self, documents):
            self.documents = documents

        def build_index(self):
            return 2

    fake_documents = ["doc1", "doc2"]

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    monkeypatch.setattr(
        "app.main.load_documents",
        lambda zone, scope, source="api": fake_documents,
    )
    client = TestClient(app)

    response = client.post(
        "/rebuild",
        json={"zone": "Montpellier", "scope": "city"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "reconstruite" in data["message"]
    assert data["n_docs_indexed"] == 2


def test_rebuild_no_documents(monkeypatch):
    class FakeRAG:
        def set_documents(self, documents):
            return None

        def build_index(self):
            return 0

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    monkeypatch.setattr(
        "app.main.load_documents",
        lambda zone, scope, source="api": [],
    )
    client = TestClient(app)

    response = client.post(
        "/rebuild",
        json={"zone": "Montpellier", "scope": "city"},
    )

    assert response.status_code == 400
    assert "Aucun document" in response.json()["detail"]


def test_rebuild_uses_default_zone_and_scope(monkeypatch):
    class FakeRAG:
        def set_documents(self, documents):
            return None

        def build_index(self):
            return 1

    captured = {}

    def fake_load_documents(zone, scope, source="api"):
        captured["zone"] = zone
        captured["scope"] = scope
        captured["source"] = source
        return ["doc"]

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    monkeypatch.setattr("app.main.load_documents", fake_load_documents)
    client = TestClient(app)

    response = client.post("/rebuild", json={})

    assert response.status_code == 200
    assert captured["source"] == "api"
    assert isinstance(captured["zone"], str)
    assert isinstance(captured["scope"], str)


def test_rebuild_unexpected_error(monkeypatch):
    class FakeRAG:
        def set_documents(self, documents):
            return None

        def build_index(self):
            return 0

    def fake_load_documents(zone, scope, source="api"):
        raise Exception("erreur chargement")

    monkeypatch.setattr("app.main.rag_service", FakeRAG())
    monkeypatch.setattr("app.main.load_documents", fake_load_documents)
    client = TestClient(app)

    response = client.post(
        "/rebuild",
        json={"zone": "Montpellier", "scope": "city"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne du serveur."