from fastapi.testclient import TestClient

from app.main import app
from app.security import require_api_key


app.dependency_overrides[require_api_key] = lambda: "test-key"


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
    assert "Erreur interne" in response.json()["detail"]


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
    assert "Erreur interne" in response.json()["detail"]