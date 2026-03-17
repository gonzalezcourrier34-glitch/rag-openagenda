from fastapi.testclient import TestClient

from app.main import app
from app.security import require_api_key


class FakeRAGService:
    def ask(self, question: str):
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
                    first_date="2026-03-01",
                    last_date="2026-03-10",
                    url="http://test.com",
                )
            ],
        )

    def build_index(self):
        return 3

    def is_index_loaded(self):
        return True

    def set_documents(self, documents):
        return None


app.dependency_overrides[require_api_key] = lambda: "test-key"


def test_health(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGService())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["index_loaded"] is True


def test_ask(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGService())
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={"question": "Je cherche une exposition d'architecture à Montpellier"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["n_docs"] == 1


def test_rebuild(monkeypatch):
    monkeypatch.setattr("app.main.rag_service", FakeRAGService())
    monkeypatch.setattr(
        "app.main.load_documents",
        lambda zone, scope: ["fake_doc_1", "fake_doc_2", "fake_doc_3"],
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