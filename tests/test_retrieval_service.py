"""
Tests unitaires du module retrieval_service.

Objectifs :
- valider les utilitaires internes
- valider l'extraction des signaux depuis la question
- valider les fonctions de matching métier
- valider le scoring principal
- valider le ranking final et le mode debug
"""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.documents import Document

from app.retrieval_service import RetrievalService


@pytest.fixture
def service():
    return RetrievalService()


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
                "search_text": "exposition architecture montpellier musee fabre culture art",
                "is_free": True,
                "is_single_day": False,
                "duration_days": 2,
                "derived_event_terms": ["exposition", "expo"],
                "derived_music_terms": [],
                "audience_terms": ["tout public"],
                "price_info": "gratuit",
                "keywords_title": ["expo", "architecture"],
                "content_quality": 8,
                "has_long_description": True,
                "vector_score": 0.20,
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
                "content_quality": 7,
                "has_long_description": False,
                "vector_score": 0.10,
                "source_url": "http://concert.com",
                "url": "http://concert.com",
            },
        ),
        Document(
            page_content="Atelier pour enfants à Montpellier",
            metadata={
                "title": "Atelier Kids",
                "description": "Atelier créatif pour enfant",
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
                "search_text": "atelier enfants montpellier creatif culture",
                "is_free": None,
                "is_single_day": True,
                "duration_days": 1,
                "derived_event_terms": ["atelier"],
                "derived_music_terms": [],
                "audience_terms": ["enfant", "famille"],
                "price_info": "inconnu",
                "keywords_title": ["atelier"],
                "content_quality": 6,
                "has_long_description": False,
                "vector_score": 0.30,
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
    assert service.parse_iso_date("2026-03-28") == date(2026, 3, 28)


def test_parse_iso_date_invalid(service):
    assert service.parse_iso_date("bad-date") is None
    assert service.parse_iso_date(None) is None


# -------------------------------------------------------------------------
# Lecture documentaire
# -------------------------------------------------------------------------

def test_doc_dates(service, sample_docs):
    first_date, last_date = service._doc_dates(sample_docs[0])
    assert first_date == date(2026, 3, 28)
    assert last_date == date(2026, 3, 29)


def test_doc_text_prefers_search_text(service, sample_docs):
    text = service._doc_text(sample_docs[0])
    assert "montpellier" in text
    assert "exposition" in text


def test_doc_title_keywords(service, sample_docs):
    assert "expo" in service._doc_title_keywords(sample_docs[0])


def test_doc_derived_terms(service, sample_docs):
    assert "exposition" in service._doc_derived_terms(sample_docs[0])


def test_doc_derived_music_terms(service, sample_docs):
    assert "jazz" in service._doc_derived_music_terms(sample_docs[1])


def test_doc_city(service, sample_docs):
    assert service._doc_city(sample_docs[0]) == "montpellier"


def test_doc_event_type(service, sample_docs):
    assert service._doc_event_type(sample_docs[0]) == "exposition"


def test_doc_music_genre(service, sample_docs):
    assert service._doc_music_genre(sample_docs[1]) == "jazz"


def test_doc_duration_days(service, sample_docs):
    assert service._doc_duration_days(sample_docs[0]) == 2
    assert service._doc_duration_days(sample_docs[1]) == 1


def test_doc_is_single_day(service, sample_docs):
    assert service._doc_is_single_day(sample_docs[1]) is True


def test_doc_content_quality(service, sample_docs):
    assert service._doc_content_quality(sample_docs[0]) == 8


def test_doc_has_long_description(service, sample_docs):
    assert service._doc_has_long_description(sample_docs[0]) is True
    assert service._doc_has_long_description(sample_docs[1]) is False


# -------------------------------------------------------------------------
# Extraction de signaux
# -------------------------------------------------------------------------

def test_extract_city(service):
    assert service._extract_city("je cherche un concert a montpellier") == "montpellier"


def test_extract_price_filter(service):
    assert service._extract_price_filter("evenement gratuit") == "gratuit"
    assert service._extract_price_filter("concert payant") == "payant"
    assert service._extract_price_filter("concert") is None


def test_extract_date_filters_exact(service):
    signals = service._extract_date_filters("concert le 29 mars 2026")
    assert signals["exact_date"] == date(2026, 3, 29)
    assert signals["date_start"] == date(2026, 3, 29)
    assert signals["date_end"] == date(2026, 3, 29)


def test_extract_date_filters_range(service):
    signals = service._extract_date_filters("du 28 au 29 mars 2026")
    assert signals["date_start"] == date(2026, 3, 28)
    assert signals["date_end"] == date(2026, 3, 29)
    assert signals["month"] == 3
    assert signals["year"] == 2026


def test_extract_date_filters_month_year(service):
    signals = service._extract_date_filters("evenements en mars 2026")
    assert signals["month"] == 3
    assert signals["year"] == 2026


def test_extract_signals_basic(service):
    signals = service.extract_signals("Je cherche un concert de jazz gratuit à Sète")
    assert signals["city"] == "sete"
    assert signals["event_type"] in {"concert", None, ""}
    assert signals["music_genre"] in {"jazz", None, ""}
    assert signals["price_filter"] == "gratuit"
    assert isinstance(signals["keywords"], list)
    assert isinstance(signals["strong_keywords"], list)


# -------------------------------------------------------------------------
# Matching métier / helpers
# -------------------------------------------------------------------------

def test_vector_score_to_bonus(service, sample_docs):
    bonus = service._vector_score_to_bonus(sample_docs[0])
    assert bonus > 0


def test_supports_any_variant_true(service, sample_docs):
    doc_text = service._doc_text(sample_docs[0])
    doc_title_keywords = service._doc_title_keywords(sample_docs[0])
    doc_derived_terms = service._doc_derived_terms(sample_docs[0])

    assert service._supports_any_variant(
        doc_text=doc_text,
        doc_title_keywords=doc_title_keywords,
        doc_derived_terms=doc_derived_terms,
        variants=["expo", "vernissage"],
    ) is True


def test_date_overlaps_true(service):
    assert service._date_overlaps(
        event_start=date(2026, 3, 28),
        event_end=date(2026, 3, 29),
        query_start=date(2026, 3, 29),
        query_end=date(2026, 3, 30),
    ) is True


def test_date_overlaps_false(service):
    assert service._date_overlaps(
        event_start=date(2026, 3, 28),
        event_end=date(2026, 3, 29),
        query_start=date(2026, 4, 1),
        query_end=date(2026, 4, 2),
    ) is False


def test_keyword_text_score(service):
    score = service._keyword_text_score("concert jazz montpellier", ["concert", "jazz"])
    assert score > 0


def test_keyword_title_score(service):
    score, present, absent = service._keyword_title_score({"concert", "jazz"}, ["concert", "rock"])
    assert score > 0
    assert present == 1
    assert absent == 1


def test_derived_terms_score(service):
    score = service._derived_terms_score({"concert", "live"}, ["concert", "photo"])
    assert score > 0


def test_strong_keywords_score(service):
    score = service._strong_keywords_score(
        doc_text="concert jazz montpellier",
        doc_title_keywords={"concert"},
        doc_derived_terms={"jazz"},
        strong_keywords=["concert", "jazz"],
    )
    assert score > 0


def test_content_quality_score(service, sample_docs):
    score = service._content_quality_score(sample_docs[0])
    assert score > 0


def test_is_musical_document_true(service, sample_docs):
    doc = sample_docs[1]
    assert service._is_musical_document(
        doc_text=service._doc_text(doc),
        doc_event_type=service._doc_event_type(doc),
        doc_music_genre=service._doc_music_genre(doc),
        doc_derived_terms=service._doc_derived_terms(doc),
        doc_derived_music_terms=service._doc_derived_music_terms(doc),
    ) is True


def test_is_cultural_document_true(service, sample_docs):
    doc = sample_docs[0]
    assert service._is_cultural_document(
        doc_text=service._doc_text(doc),
        doc_event_type=service._doc_event_type(doc),
        doc_derived_terms=service._doc_derived_terms(doc),
    ) is True


def test_event_type_match_level_exact(service, sample_docs):
    doc = sample_docs[0]
    level = service._event_type_match_level(
        requested_event_type="exposition",
        doc_event_type=service._doc_event_type(doc),
        doc_text=service._doc_text(doc),
        doc_title_keywords=service._doc_title_keywords(doc),
        doc_derived_terms=service._doc_derived_terms(doc),
    )
    assert level == "exact"


def test_event_type_match_level_mismatch(service, sample_docs):
    doc = sample_docs[0]
    level = service._event_type_match_level(
        requested_event_type="conference",
        doc_event_type=service._doc_event_type(doc),
        doc_text=service._doc_text(doc),
        doc_title_keywords=service._doc_title_keywords(doc),
        doc_derived_terms=service._doc_derived_terms(doc),
    )
    assert level == "mismatch"


def test_is_doc_compatible_with_query_true(service, sample_docs):
    signals = service.extract_signals("Je cherche une exposition à Montpellier")
    assert service._is_doc_compatible_with_query(sample_docs[0], signals) is True


def test_is_doc_compatible_with_query_false(service, sample_docs):
    signals = service.extract_signals("Je cherche un concert de jazz")
    assert service._is_doc_compatible_with_query(sample_docs[0], signals) is False


def test_duration_penalty(service, sample_docs):
    signals = service.extract_signals("Que faire le 29 mars 2026 ?")
    penalty = service._duration_penalty(sample_docs[0], signals)
    assert penalty <= 0


# -------------------------------------------------------------------------
# Scoring principal
# -------------------------------------------------------------------------

def test_score_document_prefers_matching_doc(service, sample_docs):
    signals = service.extract_signals("Je cherche une exposition gratuite à Montpellier")
    score_expo = service.score_document(sample_docs[0], signals)
    score_concert = service.score_document(sample_docs[1], signals)

    assert score_expo > score_concert


def test_score_document_sets_debug_metadata(service, sample_docs):
    signals = service.extract_signals("Je cherche une exposition gratuite à Montpellier")
    score = service.score_document(sample_docs[0], signals)

    assert isinstance(score, float)
    assert "matched_title_keywords" in sample_docs[0].metadata
    assert "detected_city_signal" in sample_docs[0].metadata
    assert "detected_event_type_signal" in sample_docs[0].metadata


# -------------------------------------------------------------------------
# Tri et diversification
# -------------------------------------------------------------------------

def test_recency_anchor_date(service):
    anchor = service._recency_anchor_date(date(2026, 3, 28), date(2026, 3, 29))
    assert anchor == date(2026, 3, 28)


def test_temporal_distance_days_exact(service, sample_docs):
    signals = service.extract_signals("evenement le 29 mars 2026")
    distance = service._temporal_distance_days(sample_docs[0], signals)
    assert distance == 0


def test_sort_key(service, sample_docs):
    signals = service.extract_signals("evenement le 29 mars 2026")
    key = service._sort_key(sample_docs[0], 10.0, signals)
    assert isinstance(key, tuple)
    assert len(key) == 4


def test_doc_similarity_signature(service, sample_docs):
    signature = service._doc_similarity_signature(sample_docs[0])
    assert isinstance(signature, set)
    assert "expo" in signature or "archi" in signature or "musee" in signature


def test_apply_diversification_returns_limited_docs(service, sample_docs):
    scored_docs = [
        (sample_docs[0], 10.0),
        (sample_docs[1], 9.0),
        (sample_docs[2], 8.0),
    ]
    signals = service.extract_signals("Que faire à Montpellier ?")

    docs = service._apply_diversification(scored_docs, signals, top_k=2)

    assert len(docs) == 2


def test_apply_strict_post_filter_returns_compatible_docs(service, sample_docs):
    signals = service.extract_signals("Je cherche un concert de jazz")
    docs = service._apply_strict_post_filter(sample_docs, signals)

    assert len(docs) >= 1
    assert docs[0].metadata["title"] == "Jazz Night"


# -------------------------------------------------------------------------
# API publique
# -------------------------------------------------------------------------

def test_rank_documents_empty(service):
    assert service.rank_documents("question", []) == []


def test_rank_documents_returns_sorted_docs(service, sample_docs):
    docs = service.rank_documents(
        question="Je cherche une exposition gratuite à Montpellier",
        raw_docs=sample_docs,
        top_k=2,
    )

    assert len(docs) <= 2
    assert docs[0].metadata["title"] == "Expo Archi"
    assert "final_score" in docs[0].metadata


def test_rank_documents_with_scores_empty(service):
    assert service.rank_documents_with_scores("question", []) == []


def test_rank_documents_with_scores_returns_rows(service, sample_docs):
    rows = service.rank_documents_with_scores(
        question="Je cherche une exposition gratuite à Montpellier",
        raw_docs=sample_docs,
        top_k=2,
    )

    assert len(rows) <= 2
    assert rows[0]["title"] == "Expo Archi"
    assert "final_score" in rows[0]
    assert "vector_score" in rows[0]
    assert "temporal_distance_days" in rows[0]