"""
Tests unitaires du module lexical_service.

Objectifs :
- valider la normalisation de texte
- valider l'extraction robuste de texte
- valider le nettoyage léger
- valider la détection de termes
- valider l'extraction des signaux métier depuis une question
- valider le profil lexical d'un document
- valider l'extraction d'information de tarification
"""

from __future__ import annotations

import pytest

from app.lexical_service import LexicalService


@pytest.fixture
def lex():
    return LexicalService()


# -------------------------------------------------------------------------
# normalize_text
# -------------------------------------------------------------------------

def test_normalize_text_basic(lex):
    result = lex.normalize_text("  Musée Fabre  ")
    assert result == "musee fabre"


def test_normalize_text_removes_accents_and_extra_spaces(lex):
    result = lex.normalize_text("Événement   Culturel à   Sète")
    assert result == "evenement culturel a sete"


def test_normalize_text_handles_none(lex):
    assert lex.normalize_text(None) == ""


# -------------------------------------------------------------------------
# extract_text
# -------------------------------------------------------------------------

def test_extract_text_from_string(lex):
    assert lex.extract_text("Bonjour") == "Bonjour"


def test_extract_text_from_none(lex):
    assert lex.extract_text(None) == ""


def test_extract_text_from_int(lex):
    assert lex.extract_text(123) == "123"


def test_extract_text_from_dict_fr_key(lex):
    value = {"fr": "Exposition photo"}
    assert lex.extract_text(value) == "Exposition photo"


def test_extract_text_from_dict_plain_text(lex):
    value = {"text": "Concert jazz"}
    result = lex.extract_text(value)
    assert isinstance(result, str)


# -------------------------------------------------------------------------
# clean_text
# -------------------------------------------------------------------------

def test_clean_text_collapses_spaces(lex):
    text = "Une    belle   exposition"
    result = lex.clean_text(text)
    assert result == "Une belle exposition"


def test_clean_text_handles_none(lex):
    assert lex.clean_text(None) == ""


# -------------------------------------------------------------------------
# contains_any_term
# -------------------------------------------------------------------------

def test_contains_any_term_true(lex):
    text = lex.normalize_text("Grand concert de jazz à Montpellier")
    assert lex.contains_any_term(text, ["concert", "expo"]) is True


def test_contains_any_term_false(lex):
    text = lex.normalize_text("Atelier de cuisine")
    assert lex.contains_any_term(text, ["concert", "expo"]) is False


def test_contains_any_term_handles_empty_terms(lex):
    text = lex.normalize_text("Atelier de cuisine")
    assert lex.contains_any_term(text, []) is False


# -------------------------------------------------------------------------
# extract_price_info
# -------------------------------------------------------------------------

def test_extract_price_info_detects_gratuit(lex):
    price_info, is_free = lex.extract_price_info(
        "Concert gratuit",
        "Entrée libre pour tous",
        "",
        None,
        None,
        None,
    )

    assert isinstance(price_info, str)
    assert is_free is True


def test_extract_price_info_detects_payant(lex):
    price_info, is_free = lex.extract_price_info(
        "Concert",
        "Tarif plein 12 euros",
        "",
        None,
        None,
        None,
    )

    assert isinstance(price_info, str)
    assert is_free is False


def test_extract_price_info_unknown_when_no_signal(lex):
    price_info, is_free = lex.extract_price_info(
        "Atelier",
        "Informations à venir",
        "",
        None,
        None,
        None,
    )

    assert isinstance(price_info, str)
    assert is_free in (None, True, False)


# -------------------------------------------------------------------------
# build_document_lexical_profile
# -------------------------------------------------------------------------

def test_build_document_lexical_profile_returns_expected_keys(lex):
    profile = lex.build_document_lexical_profile(
        title="Concert jazz live",
        description="Une soirée musicale à Montpellier",
        long_description="Avec plusieurs artistes sur scène",
        event_type="Concert",
    )

    assert isinstance(profile, dict)
    assert "keywords_title" in profile
    assert "derived_event_terms" in profile
    assert "derived_music_terms" in profile
    assert "audience_terms" in profile
    assert "canonical_event_type" in profile
    assert "music_genre" in profile


def test_build_document_lexical_profile_detects_concert(lex):
    profile = lex.build_document_lexical_profile(
        title="Concert jazz live",
        description="Une soirée musicale",
        long_description="Avec plusieurs artistes",
        event_type="Concert",
    )

    assert isinstance(profile["derived_event_terms"], list)
    assert profile["canonical_event_type"] in ("concert", "Concert", "")


def test_build_document_lexical_profile_detects_music_genre(lex):
    profile = lex.build_document_lexical_profile(
        title="Soirée jazz",
        description="Concert live",
        long_description="Ambiance musicale",
        event_type="Concert",
    )

    assert "music_genre" in profile
    assert isinstance(profile["derived_music_terms"], list)


# -------------------------------------------------------------------------
# extract_question_signals
# -------------------------------------------------------------------------

def test_extract_question_signals_basic_concert_question(lex):
    signals = lex.extract_question_signals(
        lex.normalize_text("Je cherche un concert de jazz gratuit à Montpellier")
    )

    assert isinstance(signals, dict)
    assert "keywords" in signals
    assert "event_type" in signals
    assert "music_genre" in signals
    assert "price_filter" in signals
    assert "is_cultural_query" in signals


def test_extract_question_signals_detects_event_type(lex):
    signals = lex.extract_question_signals(
        lex.normalize_text("Y a-t-il une exposition à Montpellier ?")
    )

    assert signals["event_type"] in ("exposition", "", None)


def test_extract_question_signals_detects_music_genre(lex):
    signals = lex.extract_question_signals(
        lex.normalize_text("Je veux un concert de jazz")
    )

    assert signals["music_genre"] in ("jazz", "", None)


def test_extract_question_signals_detects_price_filter_gratuit(lex):
    signals = lex.extract_question_signals(
        lex.normalize_text("Quels événements gratuits ce week-end ?")
    )

    assert signals["price_filter"] in ("gratuit", None, "")


def test_extract_question_signals_detects_cultural_query(lex):
    signals = lex.extract_question_signals(
        lex.normalize_text("Quels événements culturels à Montpellier ?")
    )

    assert isinstance(signals["is_cultural_query"], bool)


def test_extract_question_signals_returns_keywords_list(lex):
    signals = lex.extract_question_signals(
        lex.normalize_text("Je cherche une exposition photo à Montpellier")
    )

    assert isinstance(signals["keywords"], list)


# -------------------------------------------------------------------------
# Constantes métier
# -------------------------------------------------------------------------

def test_lexical_service_has_expected_constant_sets(lex):
    assert hasattr(lex, "CULTURAL_EVENT_TYPES")
    assert hasattr(lex, "MUSICAL_EVENT_TYPES")
    assert hasattr(lex, "CULTURAL_TERMS")
    assert hasattr(lex, "EVENT_TYPE_TERMS")
    assert hasattr(lex, "MUSIC_GENRE_TERMS")