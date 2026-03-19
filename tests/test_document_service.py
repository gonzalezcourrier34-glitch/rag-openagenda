from pathlib import Path

import pandas as pd
import pytest
import requests
from langchain_core.documents import Document

from app.document_service import (
    _safe,
    build_event_document,
    build_indexable_text,
    doc_matches_filters,
    extract_filters_from_question,
    fetch_and_save_events,
    fetch_openagenda_events,
    get_known_metadata_values,
    load_documents,
    load_events_from_csv,
    load_events_from_json,
    normalize_events,
    normalize_text,
    parse_date_filters,
    save_events_to_csv,
    save_events_to_json,
    score_document,
    search_agendas_for_zone,
)


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP error {self.status_code}")

    def json(self):
        return self._payload


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Exposition architecture à Montpellier",
            metadata={
                "title": "Expo Archi",
                "location_name": "Musée Fabre",
                "city": "Montpellier",
                "region": "Occitanie",
                "first_date": "2025-09-20",
                "last_date": "2025-09-21",
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
                "first_date": "2025-09-15",
                "last_date": "2025-09-15",
                "event_type": "Concert",
                "url": "http://concert.com",
            },
        ),
    ]


def test_safe_none():
    assert _safe(None) == ""


def test_safe_value():
    assert _safe(123) == "123"
    assert _safe("abc") == "abc"


def test_normalize_text_basic():
    assert normalize_text("  Musée Fabre  ") == "musee fabre"
    assert normalize_text("Événement   Culturel") == "evenement culturel"


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
    assert "Titre : Expo" in text
    assert "Description : Architecture" in text
    assert "Ville : Montpellier" in text


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


def test_get_known_metadata_values(sample_documents):
    values = get_known_metadata_values(sample_documents)

    assert "montpellier" in values["cities"]
    assert "sete" in values["cities"]
    assert "musee fabre" in values["locations"]
    assert "exposition" in values["event_types"]
    assert "concert" in values["event_types"]


def test_parse_date_filters_month():
    start, end = parse_date_filters("Quels événements à Montpellier en septembre 2025 ?")

    assert start == "2025-09-01"
    assert end == "2025-09-30"


def test_parse_date_filters_exact_date():
    start, end = parse_date_filters("Quels événements le 20 septembre 2025 ?")

    assert start == "2025-09-20"
    assert end == "2025-09-20"


def test_parse_date_filters_range():
    start, end = parse_date_filters("Quels événements du 20 au 21 septembre 2025 ?")

    assert start == "2025-09-20"
    assert end == "2025-09-21"


def test_parse_date_filters_numeric():
    start, end = parse_date_filters("Quels événements le 20/09/2025 ?")

    assert start == "2025-09-20"
    assert end == "2025-09-20"


def test_parse_date_filters_today():
    start, end = parse_date_filters("Quels événements aujourd'hui ?")

    assert start is not None
    assert end is not None
    assert start == end


def test_extract_filters_from_question(sample_documents):
    filters = extract_filters_from_question(
        question="Y a-t-il une exposition au Musée Fabre à Montpellier en septembre 2025 ?",
        documents=sample_documents,
    )

    assert "montpellier" in filters["cities"]
    assert "musee fabre" in filters["locations"]
    assert "exposition" in filters["event_types"]
    assert filters["date_start"] == "2025-09-01"
    assert filters["date_end"] == "2025-09-30"


def test_extract_filters_from_question_fuzzy_city(sample_documents):
    filters = extract_filters_from_question(
        question="Je cherche une exposition à Montpelier",
        documents=sample_documents,
    )

    assert "montpellier" in filters["cities"]


def test_doc_matches_filters_true(sample_documents):
    filters = {
        "cities": ["montpellier"],
        "locations": ["musee fabre"],
        "event_types": ["exposition"],
        "date_start": "2025-09-01",
        "date_end": "2025-09-30",
        "keywords": [],
    }

    assert doc_matches_filters(sample_documents[0], filters) is True


def test_doc_matches_filters_false_city(sample_documents):
    filters = {
        "cities": ["montpellier"],
        "locations": [],
        "event_types": [],
        "date_start": None,
        "date_end": None,
        "keywords": [],
    }

    assert doc_matches_filters(sample_documents[1], filters) is False


def test_doc_matches_filters_false_date(sample_documents):
    filters = {
        "cities": [],
        "locations": [],
        "event_types": [],
        "date_start": "2025-10-01",
        "date_end": "2025-10-31",
        "keywords": [],
    }

    assert doc_matches_filters(sample_documents[0], filters) is False


def test_score_document_prefers_matching_doc(sample_documents):
    filters = {
        "cities": ["montpellier"],
        "locations": ["musee fabre"],
        "event_types": ["exposition"],
        "date_start": "2025-09-01",
        "date_end": "2025-09-30",
        "keywords": ["architecture"],
    }

    score_0 = score_document(
        question="Je cherche une exposition d'architecture au Musée Fabre à Montpellier",
        doc=sample_documents[0],
        filters=filters,
    )
    score_1 = score_document(
        question="Je cherche une exposition d'architecture au Musée Fabre à Montpellier",
        doc=sample_documents[1],
        filters=filters,
    )

    assert score_0 > score_1


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


def test_search_agendas_for_zone_request_error(monkeypatch):
    monkeypatch.setattr("app.document_service.OPENAGENDA_API_KEY", "fake-key")

    def fake_get(*args, **kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr("app.document_service.requests.get", fake_get)

    with pytest.raises(RuntimeError, match="Erreur réseau lors de la recherche des agendas"):
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


def test_fetch_openagenda_events_request_error(monkeypatch):
    monkeypatch.setattr("app.document_service.OPENAGENDA_API_KEY", "fake-key")

    def fake_get(*args, **kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr("app.document_service.requests.get", fake_get)

    with pytest.raises(RuntimeError, match="Erreur réseau lors de la récupération des événements"):
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


def test_fetch_and_save_events_skips_runtime_error_and_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "app.document_service.search_agendas_for_zone",
        lambda zone: [{"uid": "111"}],
    )

    def fake_fetch(*args, **kwargs):
        raise RuntimeError("erreur réseau")

    monkeypatch.setattr("app.document_service.fetch_openagenda_events", fake_fetch)

    df = fetch_and_save_events(zone="Montpellier", scope="city")

    assert isinstance(df, pd.DataFrame)
    assert df.empty


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


def test_load_documents_filters_poor_events_from_json(tmp_path: Path):
    json_path = tmp_path / "events.json"
    events = [
        {
            "uid": "1",
            "agenda": {"uid": "42"},
            "title": {"fr": ""},
            "description": {"fr": ""},
            "canonicalUrl": "http://json.com",
            "firstDate": "2026-03-01",
            "lastDate": "2026-03-10",
            "eventType": "Exposition",
            "location": {
                "name": "",
                "city": "Montpellier",
                "region": "Occitanie",
            },
        }
    ]
    save_events_to_json(events, json_path)

    docs = load_documents(source="json", json_path=json_path)

    assert docs == []


def test_load_documents_invalid_source():
    with pytest.raises(ValueError, match="source doit valoir 'api', 'json' ou 'csv'"):
        load_documents(source="invalid")