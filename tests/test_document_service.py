from pathlib import Path

import pandas as pd
import pytest
from langchain_core.documents import Document

from app.document_service import (
    _safe,
    build_event_document,
    build_indexable_text,
    fetch_and_save_events,
    fetch_openagenda_events,
    load_documents,
    load_events_from_csv,
    load_events_from_json,
    normalize_events,
    save_events_to_csv,
    save_events_to_json,
    search_agendas_for_zone,
)


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP error {self.status_code}")

    def json(self):
        return self._payload


def test_safe_none():
    assert _safe(None) == ""


def test_safe_value():
    assert _safe(123) == "123"
    assert _safe("abc") == "abc"


def test_save_and_load_events_json(tmp_path: Path):
    events = [{"uid": "1", "title": {"fr": "Expo test"}}]
    output_path = tmp_path / "raw" / "events.json"

    save_events_to_json(events, output_path)
    loaded = load_events_from_json(output_path)

    assert output_path.exists()
    assert loaded == events


def test_load_events_from_json_file_not_found(tmp_path: Path):
    missing_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="Fichier JSON introuvable"):
        load_events_from_json(missing_path)


def test_save_and_load_events_csv(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "event_uid": "1",
                "title": "Expo test",
                "description": "Description",
            }
        ]
    )
    output_path = tmp_path / "processed" / "events.csv"

    save_events_to_csv(df, output_path)
    loaded = load_events_from_csv(output_path)

    assert output_path.exists()
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.loc[0, "title"] == "Expo test"


def test_load_events_from_csv_file_not_found(tmp_path: Path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Fichier CSV introuvable"):
        load_events_from_csv(missing_path)


def test_normalize_events_empty():
    df = normalize_events([])

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    expected_cols = [
        "event_uid",
        "agenda_uid",
        "title",
        "description",
        "location_name",
        "city",
        "region",
        "first_date",
        "last_date",
        "event_type",
        "source_url",
    ]
    assert list(df.columns) == expected_cols


def test_normalize_events_basic():
    events = [
        {
            "uid": "1",
            "agenda": {"uid": "42"},
            "title": {"fr": "Expo architecture"},
            "description": {"fr": " Une    belle   exposition "},
            "canonicalUrl": "http://test.com",
            "firstDate": "2026-03-01",
            "lastDate": "2026-03-10",
            "eventType": "Exposition",
            "location": {
                "name": "Musée Fabre",
                "city": "Montpellier",
                "region": "Occitanie",
            },
        }
    ]

    df = normalize_events(events)

    assert len(df) == 1
    assert df.loc[0, "event_uid"] == "1"
    assert df.loc[0, "agenda_uid"] == "42"
    assert df.loc[0, "title"] == "Expo architecture"
    assert df.loc[0, "description"] == "Une belle exposition"
    assert df.loc[0, "city"] == "Montpellier"
    assert df.loc[0, "region"] == "Occitanie"
    assert df.loc[0, "source_url"] == "http://test.com"


def test_build_indexable_text():
    df = pd.DataFrame(
        [
            {
                "title": "Expo",
                "description": "Architecture",
                "location_name": "Musée",
                "city": "Montpellier",
                "region": "Occitanie",
                "first_date": "2026-03-01",
                "last_date": "2026-03-10",
                "event_type": "Exposition",
            }
        ]
    )

    result = build_indexable_text(df)

    assert "text_for_embedding" in result.columns
    text = result.loc[0, "text_for_embedding"]
    assert "Expo" in text
    assert "Architecture" in text
    assert "Montpellier" in text


def test_build_event_document_with_prebuilt_text():
    event = {
        "event_uid": "1",
        "agenda_uid": "42",
        "title": "Expo",
        "description": "Architecture",
        "location_name": "Musée",
        "city": "Montpellier",
        "region": "Occitanie",
        "first_date": "2026-03-01",
        "last_date": "2026-03-10",
        "event_type": "Exposition",
        "source_url": "http://test.com",
        "text_for_embedding": "Texte déjà prêt",
    }

    doc = build_event_document(event)

    assert isinstance(doc, Document)
    assert doc.page_content == "Texte déjà prêt"
    assert doc.metadata["doc_id"] == "openagenda_1"
    assert doc.metadata["chunk_id"] == "openagenda_1_0"
    assert doc.metadata["title"] == "Expo"
    assert doc.metadata["url"] == "http://test.com"


def test_build_event_document_without_prebuilt_text():
    event = {
        "event_uid": "2",
        "agenda_uid": "99",
        "title": "Concert",
        "description": "Musique live",
        "location_name": "Salle A",
        "city": "Sète",
        "region": "Occitanie",
        "first_date": "2026-04-01",
        "last_date": "2026-04-01",
        "event_type": "Concert",
        "source_url": "http://concert.com",
    }

    doc = build_event_document(event)

    assert "Titre : Concert" in doc.page_content
    assert "Description : Musique live" in doc.page_content
    assert doc.metadata["event_uid"] == "2"
    assert doc.metadata["city"] == "Sète"


def test_search_agendas_for_zone_ok(monkeypatch):
    monkeypatch.setattr("app.document_service.OPENAGENDA_API_KEY", "fake-key")

    def fake_get(url, headers=None, params=None, timeout=None):
        assert "api.openagenda.com/v2/agendas" in url
        assert headers == {"key": "fake-key"}
        assert params["search"] == "Montpellier"
        return FakeResponse({"agendas": [{"uid": "123"}, {"uid": "456"}]})

    monkeypatch.setattr("app.document_service.requests.get", fake_get)

    agendas = search_agendas_for_zone("Montpellier")

    assert isinstance(agendas, list)
    assert len(agendas) == 2
    assert agendas[0]["uid"] == "123"


def test_search_agendas_for_zone_missing_api_key(monkeypatch):
    monkeypatch.setattr("app.document_service.OPENAGENDA_API_KEY", "")

    with pytest.raises(ValueError, match="OPENAGENDA_API_KEY manquante"):
        search_agendas_for_zone("Montpellier")


def test_search_agendas_for_zone_invalid_payload(monkeypatch):
    monkeypatch.setattr("app.document_service.OPENAGENDA_API_KEY", "fake-key")

    monkeypatch.setattr(
        "app.document_service.requests.get",
        lambda *args, **kwargs: FakeResponse({"unexpected": []}),
    )

    with pytest.raises(ValueError, match="Réponse inattendue"):
        search_agendas_for_zone("Montpellier")


def test_fetch_openagenda_events_ok(monkeypatch):
    monkeypatch.setattr("app.document_service.OPENAGENDA_API_KEY", "fake-key")

    page_1 = {
        "events": [{"uid": "1"}, {"uid": "2"}],
        "total": 3,
    }
    page_2 = {
        "events": [{"uid": "3"}],
        "total": 3,
    }

    calls = {"count": 0}

    def fake_get(url, params=None, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return FakeResponse(page_1)
        return FakeResponse(page_2)

    monkeypatch.setattr("app.document_service.requests.get", fake_get)

    events = fetch_openagenda_events(
        agenda_uid="123",
        zone="Montpellier",
        scope="city",
        limit=2,
        max_pages=10,
    )

    assert len(events) == 3
    assert events[0]["uid"] == "1"
    assert events[-1]["uid"] == "3"


def test_fetch_openagenda_events_missing_api_key(monkeypatch):
    monkeypatch.setattr("app.document_service.OPENAGENDA_API_KEY", "")

    with pytest.raises(ValueError, match="OPENAGENDA_API_KEY manquante"):
        fetch_openagenda_events("123", "Montpellier")


def test_fetch_openagenda_events_invalid_payload(monkeypatch):
    monkeypatch.setattr("app.document_service.OPENAGENDA_API_KEY", "fake-key")

    monkeypatch.setattr(
        "app.document_service.requests.get",
        lambda *args, **kwargs: FakeResponse({"unexpected": []}),
    )

    with pytest.raises(ValueError, match="Réponse inattendue de l'API"):
        fetch_openagenda_events("123", "Montpellier")


def test_fetch_and_save_events_returns_dataframe(monkeypatch, tmp_path: Path):
    json_path = tmp_path / "raw" / "events.json"
    csv_path = tmp_path / "processed" / "events.csv"

    monkeypatch.setattr(
        "app.document_service.search_agendas_for_zone",
        lambda zone: [{"uid": "111"}, {"uid": "222"}],
    )

    fake_events = [
        {
            "uid": "1",
            "agenda": {"uid": "111"},
            "title": {"fr": "Expo 1"},
            "description": {"fr": "Desc 1"},
            "canonicalUrl": "http://test1.com",
            "firstDate": "2026-03-01",
            "lastDate": "2026-03-10",
            "eventType": "Exposition",
            "location": {
                "name": "Lieu 1",
                "city": "Montpellier",
                "region": "Occitanie",
            },
        }
    ]

    monkeypatch.setattr(
        "app.document_service.fetch_openagenda_events",
        lambda agenda_uid, zone, scope: fake_events,
    )

    df = fetch_and_save_events(
        zone="Montpellier",
        scope="city",
        json_path=json_path,
        csv_path=csv_path,
    )

    assert not df.empty
    assert "text_for_embedding" in df.columns
    assert json_path.exists()
    assert csv_path.exists()


def test_fetch_and_save_events_no_agendas(monkeypatch):
    monkeypatch.setattr("app.document_service.search_agendas_for_zone", lambda zone: [])

    df = fetch_and_save_events(zone="Montpellier", scope="city")

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_load_documents_from_api(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "event_uid": "1",
                "agenda_uid": "42",
                "title": "Expo",
                "description": "Description",
                "location_name": "Musée",
                "city": "Montpellier",
                "region": "Occitanie",
                "first_date": "2026-03-01",
                "last_date": "2026-03-10",
                "event_type": "Exposition",
                "source_url": "http://test.com",
                "text_for_embedding": "Texte 1",
            }
        ]
    )

    monkeypatch.setattr(
        "app.document_service.fetch_and_save_events",
        lambda **kwargs: df.copy(),
    )

    docs = load_documents(source="api")

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].metadata["title"] == "Expo"


def test_load_documents_from_json(tmp_path: Path):
    json_path = tmp_path / "events.json"
    events = [
        {
            "uid": "1",
            "agenda": {"uid": "42"},
            "title": {"fr": "Expo JSON"},
            "description": {"fr": "Desc JSON"},
            "canonicalUrl": "http://json.com",
            "firstDate": "2026-03-01",
            "lastDate": "2026-03-10",
            "eventType": "Exposition",
            "location": {
                "name": "Musée",
                "city": "Montpellier",
                "region": "Occitanie",
            },
        }
    ]
    save_events_to_json(events, json_path)

    docs = load_documents(source="json", json_path=json_path)

    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Expo JSON"


def test_load_documents_from_csv(tmp_path: Path):
    csv_path = tmp_path / "events.csv"
    df = pd.DataFrame(
        [
            {
                "event_uid": "1",
                "agenda_uid": "42",
                "title": "Expo CSV",
                "description": "Desc CSV",
                "location_name": "Musée",
                "city": "Montpellier",
                "region": "Occitanie",
                "first_date": "2026-03-01",
                "last_date": "2026-03-10",
                "event_type": "Exposition",
                "source_url": "http://csv.com",
                "text_for_embedding": "Texte CSV",
            }
        ]
    )
    save_events_to_csv(df, csv_path)

    docs = load_documents(source="csv", csv_path=csv_path)

    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Expo CSV"


def test_load_documents_invalid_source():
    with pytest.raises(ValueError, match="source doit valoir 'api', 'json' ou 'csv'"):
        load_documents(source="invalid")