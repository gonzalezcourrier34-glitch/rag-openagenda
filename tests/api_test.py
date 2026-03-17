import pytest
from fastapi.testclient import TestClient

from app.main import app


# =========================
# Mock du RAGService
# =========================
class FakeRAGService:
    def ask(self, question: str):
        return {
            "question": question,
            "answer": "Voici une exposition d'architecture à Montpellier.",
            "n_docs": 2,
            "documents": [
                {
                    "title": "Expo Archi",
                    "city": "Montpellier",
                    "location_name": "Musée X",
                    "first_date": "2026-03-01",
                    "last_date": "2026-03-10",
                    "url": "http://test.com"
                }
            ]
        }

    def rebuild(self, zone: str, scope: str):
        return {
            "status": "success",
            "message": "Index reconstruit"
        }


# =========================
# Fixture pytest
# =========================
@pytest.fixture
def client(monkeypatch):
    fake_rag = FakeRAGService()

    # Remplace le rag_service réel par le fake
    monkeypatch.setattr("app.main.rag_service", fake_rag)

    return TestClient(app)


# =========================
# Tests
# =========================
def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data


def test_rebuild(client):
    response = client.post("/rebuild", json={
        "zone": "Montpellier",
        "scope": "city"
    })

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"


def test_ask(client):
    payload = {
        "question": "Je cherche une exposition d'architecture"
    }

    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert data["n_docs"] == 2
    assert isinstance(data["documents"], list)