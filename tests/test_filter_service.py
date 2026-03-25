"""
Tests unitaires du module filter_service.

Objectifs :
- valider les utilitaires internes
- valider l'extraction des filtres depuis la question
- valider les fonctions de matching documentaires
- valider le préfiltrage global
- valider le mode debug
"""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.documents import Document

from app.filter_service import FilterService


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def service():
    return FilterService()


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="Exposition architecture à Montpellier",
            metadata={
                "title": "Expo Archi",
                "description": "Une belle exposition d'architecture",
                "location_name": "Musée Fabre",
                "city": "Montpellier",
                "city_norm": "montpellier",
                "region": "Occitanie",
                "first_date": "2026-03-28",
                "last_date": "2026-03-29",
                "event_type": "Exposition",
                "event_type_norm": "exposition",
                "canonical_event_type": "exposition",
                "music_genre": "",
                "music_genre_norm": "",
                "search_text": "exposition architecture montpellier musee fabre culture",
                "is_free": True,
                "is_single_day": False,
                "duration_days": 2,
                "derived_event_terms": ["exposition", "expo"],
                "derived_music_terms": [],
                "audience_terms": ["tout public"],
                "price_info": "gratuit",
                "keywords_title": ["expo", "architecture"],
                "source_url": "http://expo.com",
                "url": "http://expo.com",
            },
        ),
        Document(
            page_content="Concert jazz à Sète",
            metadata={
                "title": "Jazz Night",
                "description": "Concert live",
                "location_name": "Salle Y",
                "city": "Sète",
                "city_norm": "sete",
                "region": "Occitanie",
                "first_date": "2026-03-29",
                "last_date": "2026-03-29",
                "event_type": "Concert",
                "event_type_norm": "concert",
                "canonical_event_type": "concert",
                "music_genre": "jazz",
                "music_genre_norm": "jazz",
                "search_text": "concert jazz sete live musique",
                "is_free": False,
                "is_single_day": True,
                "duration_days": 1,
                "derived_event_terms": ["concert", "live"],
                "derived_music_terms": ["jazz"],
                "audience_terms": ["adulte"],
                "price_info": "payant",
                "keywords_title": ["jazz"],
                "source_url": "http://concert.com",
                "url": "http://concert.com",
            },
        ),
        Document(
            page_content="Atelier pour enfants à Montpellier",
            metadata={
                "title": "Atelier Kids",
                "description": "Atelier créatif",
                "location_name": "Maison pour tous",
                "city": "Montpellier",
                "city_norm": "montpellier",
                "region": "Occitanie",
                "first_date": "2026-04-10",
                "last_date": "2026-04-10",
                "event_type": "Atelier",
                "event_type_norm": "atelier",
                "canonical_event_type": "atelier",
                "music_genre": "",
                "music_genre_norm": "",
                "search_text": "atelier enfants montpellier creatif",
                "is_free": None,
                "is_single_day": True,
                "duration_days": 1,
                "derived_event_terms": ["atelier"],
                "derived_music_terms": [],
                "audience_terms": ["enfant", "famille"],
                "price_info": "inconnu",
                "keywords_title": ["atelier"],
                "source_url": "http://atelier.com",
                "url": "http://atelier.com",
            },
        ),
    ]


# -------------------------------------------------------------------------
# Utilitaires de base
# -------------------------------------------------------------------------

def test_safe_none(service):
    assert service._safe(None) == ""


def test_safe_value(service):
    assert service._safe(123) == "123"


def test_normalize_text(service):
    assert service.normalize_text("  Musée Fabre  ") == "musee fabre"


def test_parse_iso_date_valid(service):
    parsed = service.parse_iso_date("2026-03-28")
    assert parsed == date(2026, 3, 28)


def test_parse_iso_date_invalid(service):
    assert service.parse_iso_date("bad-date") is None
    assert service.parse_iso_date(None) is None


def test_build_date_valid(service):
    assert service._build_date(2026, 3, 28) == date(2026, 3, 28)


def test_build_date_invalid(service):
    assert service._build_date(2026, 2, 31) is None


# -------------------------------------------------------------------------
# Lecture des métadonnées documentaires
# -------------------------------------------------------------------------

def test_doc_dates(service, sample_docs):
    first_date, last_date = service._doc_dates(sample_docs[0])
    assert first_date == date(2026, 3, 28)
    assert last_date == date(2026, 3, 29)


def test_doc_city(service, sample_docs):
    assert service._doc_city(sample_docs[0]) == "montpellier"


def test_doc_event_type(service, sample_docs):
    assert service._doc_event_type(sample_docs[0]) == "exposition"


def test_doc_music_genre(service, sample_docs):
    assert service._doc_music_genre(sample_docs[1]) == "jazz"


def test_doc_search_text_prefers_search_text(service, sample_docs):
    text = service._doc_search_text(sample_docs[0])
    assert "montpellier" in text
    assert "exposition" in text


def test_doc_is_free(service, sample_docs):
    assert service._doc_is_free(sample_docs[0]) is True
    assert service._doc_is_free(sample_docs[2]) is None


def test_doc_is_single_day(service, sample_docs):
    assert service._doc_is_single_day(sample_docs[1]) is True


def test_doc_duration_days(service, sample_docs):
    assert service._doc_duration_days(sample_docs[0]) == 2
    assert service._doc_duration_days(sample_docs[1]) == 1


def test_doc_derived_event_terms(service, sample_docs):
    assert "exposition" in service._doc_derived_event_terms(sample_docs[0])


def test_doc_derived_music_terms(service, sample_docs):
    assert "jazz" in service._doc_derived_music_terms(sample_docs[1])


def test_doc_audience_terms(service, sample_docs):
    assert "enfant" in service._doc_audience_terms(sample_docs[2])


# -------------------------------------------------------------------------
# Extraction des contraintes
# -------------------------------------------------------------------------

def test_extract_city(service):
    assert service._extract_city("je cherche un concert a montpellier") == "montpellier"


def test_extract_duration_filter_single_day(service):
    assert service._extract_duration_filter("evenement sur une journee") == "single_day"


def test_extract_duration_filter_multi_day(service):
    assert service._extract_duration_filter("festival sur plusieurs jours") == "multi_day"


def test_get_month_bounds(service):
    start, end = service._get_month_bounds(2026, 3)
    assert start == date(2026, 3, 1)
    assert end == date(2026, 4, 1)


def test_get_year_bounds(service):
    start, end = service._get_year_bounds(2026)
    assert start == date(2026, 1, 1)
    assert end == date(2027, 1, 1)


def test_extract_explicit_weekend_range(service):
    start, end = service._extract_explicit_weekend_range("week-end du 28 et 29 mars 2026")
    assert start == date(2026, 3, 28)
    assert end == date(2026, 3, 29)


def test_extract_date_filters_exact_date(service):
    filters = service._extract_date_filters("concert le 29 mars 2026")
    assert filters["exact_date"] == date(2026, 3, 29)
    assert filters["time_mode"] == "exact_date"


def test_extract_date_filters_range(service):
    filters = service._extract_date_filters("du 28 au 29 mars 2026")
    assert filters["start_date"] == date(2026, 3, 28)
    assert filters["end_date"] == date(2026, 3, 29)
    assert filters["time_mode"] == "date_range"


def test_extract_date_filters_month_year(service):
    filters = service._extract_date_filters("evenements en mars 2026")
    assert filters["month"] == 3
    assert filters["year"] == 2026
    assert filters["time_mode"] == "month_year"


def test_extract_date_filters_year(service):
    filters = service._extract_date_filters("evenements en 2026")
    assert filters["year"] == 2026
    assert filters["time_mode"] == "year"


def test_extract_filters_with_default_city(service):
    filters = service.extract_filters("je cherche une exposition", default_city="Montpellier")
    assert filters["city"] == "montpellier"


def test_extract_filters_with_explicit_city(service):
    filters = service.extract_filters(
        "je cherche une exposition à Sète",
        default_city="Montpellier",
    )
    assert filters["explicit_city"] == "sete"
    assert filters["city"] == "sete"


# -------------------------------------------------------------------------
# Matching
# -------------------------------------------------------------------------

def test_matches_city_true(service, sample_docs):
    filters = {"city": "montpellier"}
    assert service.matches_city(sample_docs[0], filters) is True


def test_matches_city_false(service, sample_docs):
    filters = {"city": "montpellier"}
    assert service.matches_city(sample_docs[1], filters) is False


def test_matches_event_type_true(service, sample_docs):
    filters = {"event_type": "exposition"}
    assert service.matches_event_type(sample_docs[0], filters) is True


def test_matches_event_type_false(service, sample_docs):
    filters = {"event_type": "conference"}
    assert service.matches_event_type(sample_docs[0], filters) is False


def test_matches_music_genre_true(service, sample_docs):
    filters = {"music_genre": "jazz"}
    assert service.matches_music_genre(sample_docs[1], filters) is True


def test_matches_music_genre_false(service, sample_docs):
    filters = {"music_genre": "rock"}
    assert service.matches_music_genre(sample_docs[1], filters) is False


def test_matches_cultural_scope_true(service, sample_docs):
    filters = {"is_cultural_query": True}
    assert service.matches_cultural_scope(sample_docs[0], filters) is True


def test_matches_duration_single_day(service, sample_docs):
    filters = {"duration_filter": "single_day"}
    assert service.matches_duration(sample_docs[1], filters) is True
    assert service.matches_duration(sample_docs[0], filters) is False


def test_matches_duration_multi_day(service, sample_docs):
    filters = {"duration_filter": "multi_day"}
    assert service.matches_duration(sample_docs[0], filters) is True
    assert service.matches_duration(sample_docs[1], filters) is False


def test_matches_date_exact(service, sample_docs):
    filters = {
        "exact_date": date(2026, 3, 29),
        "month": None,
        "year": None,
        "start_date": None,
        "end_date": None,
    }
    assert service.matches_date(sample_docs[0], filters) is True
    assert service.matches_date(sample_docs[1], filters) is True


def test_matches_date_range(service, sample_docs):
    filters = {
        "exact_date": None,
        "month": None,
        "year": None,
        "start_date": date(2026, 3, 28),
        "end_date": date(2026, 3, 29),
        "time_mode": "date_range",
    }
    assert service.matches_date(sample_docs[0], filters) is True
    assert service.matches_date(sample_docs[2], filters) is False


def test_matches_price_gratuit(service, sample_docs):
    filters = {"price_filter": "gratuit"}
    assert service.matches_price(sample_docs[0], filters) is True
    assert service.matches_price(sample_docs[1], filters) is False
    assert service.matches_price(sample_docs[2], filters) is True


def test_matches_price_payant(service, sample_docs):
    filters = {"price_filter": "payant"}
    assert service.matches_price(sample_docs[1], filters) is True
    assert service.matches_price(sample_docs[0], filters) is False
    assert service.matches_price(sample_docs[2], filters) is True


def test_matches_audience_true(service, sample_docs):
    filters = {"audience_terms": ["enfant"]}
    assert service.matches_audience(sample_docs[2], filters) is True


def test_matches_audience_true_when_doc_has_no_audience(service):
    doc = Document(page_content="x", metadata={})
    filters = {"audience_terms": ["enfant"]}
    assert service.matches_audience(doc, filters) is True


# -------------------------------------------------------------------------
# Pilotage global
# -------------------------------------------------------------------------

def test_has_strong_filters_false_with_only_default_city(service):
    filters = {
        "explicit_city": None,
        "event_type": None,
        "music_genre": None,
        "is_cultural_query": False,
        "duration_filter": None,
        "audience_terms": [],
        "exact_date": None,
        "month": None,
        "year": None,
        "start_date": None,
        "end_date": None,
        "price_filter": None,
    }
    assert service.has_strong_filters(filters) is False


def test_has_strong_filters_true(service):
    filters = {
        "explicit_city": "montpellier",
        "event_type": None,
        "music_genre": None,
        "is_cultural_query": False,
        "duration_filter": None,
        "audience_terms": [],
        "exact_date": None,
        "month": None,
        "year": None,
        "start_date": None,
        "end_date": None,
        "price_filter": None,
    }
    assert service.has_strong_filters(filters) is True


def test_run_filter_pipeline_city_only(service, sample_docs):
    filters = service.extract_filters("Que faire à Montpellier ?", default_city="Montpellier")
    filters["is_cultural_query"] = False

    pipeline = service._run_filter_pipeline(filters, sample_docs)

    assert len(pipeline["after_city"]) == 2
    assert len(pipeline["after_price"]) == 2


def test_run_filter_pipeline_returns_empty_after_city_when_no_city_match(service, sample_docs):
    filters = {
        "city": "paris",
        "explicit_city": "paris",
        "event_type": None,
        "music_genre": None,
        "is_cultural_query": False,
        "duration_filter": None,
        "audience_terms": [],
        "exact_date": None,
        "month": None,
        "year": None,
        "start_date": None,
        "end_date": None,
        "price_filter": None,
    }

    pipeline = service._run_filter_pipeline(filters, sample_docs)

    assert pipeline["after_city"] == []
    assert pipeline["after_price"] == []


def test_filter_documents_empty(service):
    assert service.filter_documents("question", []) == []


def test_filter_documents_city_only(service, sample_docs, monkeypatch):
    original_extract_filters = service.extract_filters

    def fake_extract_filters(question, default_city=None):
        filters = original_extract_filters(question, default_city)
        filters["is_cultural_query"] = False
        return filters

    monkeypatch.setattr(service, "extract_filters", fake_extract_filters)

    docs = service.filter_documents(
        question="Que faire à Montpellier ?",
        docs=sample_docs,
        default_city="Montpellier",
    )
    assert len(docs) == 2
    assert all((doc.metadata or {}).get("city") == "Montpellier" for doc in docs)


def test_filter_documents_event_type(service, sample_docs, monkeypatch):
    original_extract_filters = service.extract_filters

    def fake_extract_filters(question, default_city=None):
        filters = original_extract_filters(question, default_city)
        filters["is_cultural_query"] = False
        return filters

    monkeypatch.setattr(service, "extract_filters", fake_extract_filters)

    docs = service.filter_documents(
        question="Je cherche une exposition à Montpellier",
        docs=sample_docs,
        default_city="Montpellier",
    )
    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Expo Archi"


def test_filter_documents_music_genre(service, sample_docs, monkeypatch):
    original_extract_filters = service.extract_filters

    def fake_extract_filters(question, default_city=None):
        filters = original_extract_filters(question, default_city)
        filters["is_cultural_query"] = False
        return filters

    monkeypatch.setattr(service, "extract_filters", fake_extract_filters)

    docs = service.filter_documents(
        question="Je cherche un concert de jazz à Sète",
        docs=sample_docs,
        default_city=None,
    )
    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Jazz Night"


def test_filter_documents_date_range(service, sample_docs, monkeypatch):
    original_extract_filters = service.extract_filters

    def fake_extract_filters(question, default_city=None):
        filters = original_extract_filters(question, default_city)
        filters["is_cultural_query"] = False
        return filters

    monkeypatch.setattr(service, "extract_filters", fake_extract_filters)

    docs = service.filter_documents(
        question="Que faire du 28 au 29 mars 2026 à Montpellier ?",
        docs=sample_docs,
        default_city=None,
    )
    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Expo Archi"


def test_filter_documents_price(service, sample_docs, monkeypatch):
    original_extract_filters = service.extract_filters

    def fake_extract_filters(question, default_city=None):
        filters = original_extract_filters(question, default_city)
        filters["is_cultural_query"] = False
        return filters

    monkeypatch.setattr(service, "extract_filters", fake_extract_filters)

    docs = service.filter_documents(
        question="Je cherche un événement gratuit à Montpellier",
        docs=sample_docs,
        default_city=None,
    )
    assert len(docs) >= 1
    assert all((doc.metadata or {}).get("city") == "Montpellier" for doc in docs)


def test_filter_documents_audience(service, sample_docs, monkeypatch):
    original_extract_filters = service.extract_filters

    def fake_extract_filters(question, default_city=None):
        filters = original_extract_filters(question, default_city)
        filters["is_cultural_query"] = False
        return filters

    monkeypatch.setattr(service, "extract_filters", fake_extract_filters)

    docs = service.filter_documents(
        question="Je cherche un atelier pour enfant à Montpellier",
        docs=sample_docs,
        default_city=None,
    )
    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Atelier Kids"


# -------------------------------------------------------------------------
# Debug
# -------------------------------------------------------------------------

def test_doc_to_debug_row(service, sample_docs):
    row = service._doc_to_debug_row(sample_docs[0])

    assert row["title"] == "Expo Archi"
    assert row["city"] == "Montpellier"
    assert row["canonical_event_type"] == "exposition"
    assert row["url"] == "http://expo.com"


def test_filter_documents_with_debug(service, sample_docs):
    result = service.filter_documents_with_debug(
        question="Je cherche une exposition à Montpellier",
        docs=sample_docs,
        default_city="Montpellier",
    )

    assert "filters" in result
    assert "n_input_docs" in result
    assert "n_after_city" in result
    assert "n_after_type" in result
    assert "docs" in result
    assert "docs_debug" in result

    assert result["n_input_docs"] == 3
    assert isinstance(result["docs"], list)
    assert isinstance(result["docs_debug"], list)