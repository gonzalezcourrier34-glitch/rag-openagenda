"""
Service de ranking métier des documents après la recherche vectorielle.

Ce module intervient après la recherche FAISS afin d'affiner les
documents candidats avant la génération finale.

Objectifs :
- extraire des signaux métier utiles depuis la question utilisateur
- calculer un score hybride combinant :
  - un signal vectoriel issu de FAISS
  - un score métier explicable
- mieux respecter les contraintes fortes de la question :
  - ville
  - type d'événement
  - tarification
  - date exacte
  - période
  - mois / année
  - genre musical
  - intention culturelle large
- limiter les faux positifs sémantiques
- légèrement diversifier les résultats finaux

Important :
Le préfiltrage structuré fort peut être fait en amont par `filter_service`.
Cette classe conserve cependant quelques garde-fous métier afin
d'éviter qu'un document incompatible remonte trop haut dans le classement.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from langchain_core.documents import Document

from app.lexical_service import LexicalService


class RetrievalService:
    """
    Service de ranking métier après la recherche vectorielle.

    Cette classe complète la recherche FAISS avec un scoring hybride
    plus strict que la simple similarité sémantique.

    Le score final repose notamment sur :
    - le score vectoriel initial
    - la ville si explicitement demandée
    - le type d'événement demandé
    - le genre musical demandé
    - la compatibilité culturelle globale de la requête
    - la tarification si explicitement demandée
    - la compatibilité temporelle :
      - date exacte
      - intervalle
      - mois / année
    - quelques mots-clés discriminants
    - la richesse documentaire

    Le tri final privilégie :
    1. le score final le plus élevé
    2. la proximité temporelle si la question contient une contrainte de date
    3. la date la plus récente
    4. la durée la plus courte
    """

    MONTHS = {
        "janvier": 1,
        "fevrier": 2,
        "mars": 3,
        "avril": 4,
        "mai": 5,
        "juin": 6,
        "juillet": 7,
        "aout": 8,
        "septembre": 9,
        "octobre": 10,
        "novembre": 11,
        "decembre": 12,
    }

    KNOWN_CITIES = {
        "montpellier",
        "paris",
        "sete",
        "toulouse",
        "lyon",
        "marseille",
        "bordeaux",
        "lille",
        "nantes",
    }

    STRONG_KEYWORDS = {
        "rock",
        "jazz",
        "blues",
        "electro",
        "folk",
        "rap",
        "photo",
        "photographie",
        "vinyl",
        "gratuit",
        "payant",
        "enfant",
        "enfants",
        "famille",
        "architecture",
        "atelier",
        "concert",
        "expo",
        "exposition",
        "vernissage",
    }

    SCORE_WEIGHTS = {
        # Bonus principaux
        "vector": 4.0,
        "city_bonus": 1.5,
        "event_type_exact_match": 12.0,
        "event_type_variant_match": 7.5,
        "music_genre_match": 7.0,
        "price_match": 4.5,
        "exact_date_match": 8.0,
        "period_match": 5.0,
        "month_match": 3.0,
        "keyword_text": 0.20,
        "keyword_title_present": 1.0,
        "derived_term_present": 1.2,
        "strong_keyword_present": 1.75,
        "content_quality": 0.20,
        "long_description_bonus": 0.15,
        "single_day_bonus": 0.10,
        "cultural_doc_bonus": 3.0,
        "musical_doc_bonus": 3.0,

        # Malus métier
        "wrong_city": -15.0,
        "event_type_soft_mismatch": -8.0,
        "event_type_hard_mismatch": -18.0,
        "music_genre_missing": -10.0,
        "music_doc_missing": -12.0,
        "price_mismatch": -9.0,
        "date_mismatch": -14.0,
        "month_mismatch": -8.0,
        "strong_keyword_absent": -1.75,
        "non_cultural_doc": -12.0,
        "empty_event_type_on_cultural_query": -5.0,
        "empty_event_type_on_typed_query": -8.0,

        # Diversification
        "similarity_penalty": 1.0,
    }

    def __init__(self) -> None:
        """
        Initialise le service de ranking avec son service lexical partagé.
        """
        self.lexical_service = LexicalService()

    # -------------------------------------------------------------------------
    # Utilitaires généraux
    # -------------------------------------------------------------------------

    def _safe(self, value: object) -> str:
        """
        Convertit une valeur potentiellement nulle en chaîne.
        """
        return "" if value is None else str(value)

    def normalize_text(self, text: object) -> str:
        """
        Normalise un texte pour faciliter les comparaisons lexicales.
        """
        return self.lexical_service.normalize_text(text)

    def parse_iso_date(self, value: str | None) -> date | None:
        """
        Convertit une date ISO de type YYYY-MM-DD en objet `date`.

        Parameters
        ----------
        value : str | None
            Date brute issue des métadonnées.

        Returns
        -------
        date | None
            Date parsée si possible, sinon None.
        """
        value = self._safe(value).strip()
        if not value:
            return None

        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    # Lecture / consolidation des métadonnées documentaires
    # -------------------------------------------------------------------------

    def _doc_dates(self, doc: Document) -> tuple[date | None, date | None]:
        """
        Retourne les dates début / fin du document.
        """
        md = doc.metadata or {}
        first_date = self.parse_iso_date(md.get("first_date"))
        last_date = self.parse_iso_date(md.get("last_date"))

        if first_date and not last_date:
            last_date = first_date

        return first_date, last_date

    def _doc_text(self, doc: Document) -> str:
        """
        Retourne un texte documentaire consolidé et normalisé.

        La méthode privilégie `search_text` lorsqu'il est disponible,
        car ce champ a vocation à centraliser le texte utile pour
        l'embedding, le matching lexical et le ranking.
        """
        md = doc.metadata or {}
        search_text = md.get("search_text", "")

        if search_text:
            return self.normalize_text(search_text)

        text = (
            f"{md.get('title', '')} "
            f"{md.get('description', '')} "
            f"{md.get('location_name', '')} "
            f"{md.get('city', '')} "
            f"{md.get('event_type', '')} "
            f"{md.get('canonical_event_type', '')} "
            f"{md.get('music_genre', '')} "
            f"{md.get('price_info', '')} "
            f"{doc.page_content}"
        )
        return self.normalize_text(text)

    def _doc_title_keywords(self, doc: Document) -> set[str]:
        """
        Retourne les mots-clés du titre sous forme d'ensemble normalisé.
        """
        md = doc.metadata or {}
        keywords = md.get("keywords_title", [])

        if not isinstance(keywords, list):
            return set()

        return {
            self.normalize_text(keyword)
            for keyword in keywords
            if self.normalize_text(keyword)
        }

    def _doc_derived_terms(self, doc: Document) -> set[str]:
        """
        Retourne les termes métier dérivés du document.
        """
        md = doc.metadata or {}
        terms = md.get("derived_event_terms", [])

        if not isinstance(terms, list):
            return set()

        return {
            self.normalize_text(term)
            for term in terms
            if self.normalize_text(term)
        }

    def _doc_derived_music_terms(self, doc: Document) -> set[str]:
        """
        Retourne les termes musicaux dérivés du document.
        """
        md = doc.metadata or {}
        terms = md.get("derived_music_terms", [])

        if not isinstance(terms, list):
            return set()

        return {
            self.normalize_text(term)
            for term in terms
            if self.normalize_text(term)
        }

    def _doc_city(self, doc: Document) -> str:
        """
        Retourne la ville normalisée du document.
        """
        md = doc.metadata or {}
        return md.get("city_norm") or self.normalize_text(md.get("city", ""))

    def _doc_event_type(self, doc: Document) -> str:
        """
        Retourne le type d'événement normalisé.

        On privilégie `canonical_event_type` lorsqu'il est disponible.
        """
        md = doc.metadata or {}
        return (
            self.normalize_text(md.get("canonical_event_type", ""))
            or md.get("event_type_norm")
            or self.normalize_text(md.get("event_type", ""))
        )

    def _doc_music_genre(self, doc: Document) -> str:
        """
        Retourne le genre musical normalisé.
        """
        md = doc.metadata or {}
        return md.get("music_genre_norm") or self.normalize_text(md.get("music_genre", ""))

    def _doc_duration_days(self, doc: Document) -> int:
        """
        Retourne la durée de l'événement en jours.

        Si la durée n'est pas fournie en métadonnée, elle est
        recalculée à partir des dates début / fin.
        """
        md = doc.metadata or {}
        value = md.get("duration_days")

        try:
            if value is not None:
                return max(1, int(value))
        except (TypeError, ValueError):
            pass

        first_date, last_date = self._doc_dates(doc)

        if not first_date:
            return 999999

        if not last_date:
            last_date = first_date

        return max(1, (last_date - first_date).days + 1)

    def _doc_is_single_day(self, doc: Document) -> bool | None:
        """
        Indique si l'événement est mono-jour.
        """
        md = doc.metadata or {}
        value = md.get("is_single_day")

        if isinstance(value, bool):
            return value

        return None

    def _doc_content_quality(self, doc: Document) -> int:
        """
        Retourne un score simple de richesse documentaire.
        """
        md = doc.metadata or {}
        value = md.get("content_quality")

        try:
            return max(0, int(value)) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    def _doc_has_long_description(self, doc: Document) -> bool:
        """
        Indique si le document dispose d'une description longue.
        """
        md = doc.metadata or {}
        return bool(md.get("has_long_description"))

    # -------------------------------------------------------------------------
    # Extraction des signaux de question
    # -------------------------------------------------------------------------

    def _extract_city(self, question: str) -> str | None:
        """
        Détecte une ville connue dans la question normalisée.
        """
        for city in sorted(self.KNOWN_CITIES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(city)}\b", question):
                return city
        return None

    def _extract_price_filter(self, question: str) -> str | None:
        """
        Détecte si la question cible explicitement des événements
        gratuits ou payants.
        """
        if re.search(r"\bgratuit(?:e|s)?\b", question):
            return "gratuit"

        if re.search(r"\bpayant(?:e|s)?\b", question):
            return "payant"

        return None

    def _extract_date_filters(self, question: str) -> dict[str, Any]:
        """
        Extrait des signaux temporels depuis la question.

        Cas gérés :
        - date exacte : "20 septembre 2025"
        - mois + année : "septembre 2025"
        - intervalle :
          - "du 20 au 21 septembre 2025"
          - "du 28 au 29 mars 2026"
          - "28 et 29 mars 2026"
          - "week-end du 20 au 21 septembre 2025"
        """
        result = {
            "exact_date": None,
            "date_start": None,
            "date_end": None,
            "month": None,
            "year": None,
        }

        month_pattern = (
            r"(janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|"
            r"octobre|novembre|decembre)"
        )

        pattern_range_du_au = (
            rf"\bdu\s+(\d{{1,2}})\s+au\s+(\d{{1,2}})\s+{month_pattern}\s+(\d{{4}})\b"
        )
        match = re.search(pattern_range_du_au, question)
        if match:
            day_start, day_end, month_name, year_str = match.groups()
            year = int(year_str)
            month = self.MONTHS[month_name]
            result["date_start"] = date(year, month, int(day_start))
            result["date_end"] = date(year, month, int(day_end))
            result["month"] = month
            result["year"] = year
            return result

        pattern_range_et = rf"\b(\d{{1,2}})\s+et\s+(\d{{1,2}})\s+{month_pattern}\s+(\d{{4}})\b"
        match = re.search(pattern_range_et, question)
        if match:
            day_start, day_end, month_name, year_str = match.groups()
            year = int(year_str)
            month = self.MONTHS[month_name]
            result["date_start"] = date(year, month, int(day_start))
            result["date_end"] = date(year, month, int(day_end))
            result["month"] = month
            result["year"] = year
            return result

        pattern_exact = rf"\b(\d{{1,2}})\s+{month_pattern}\s+(\d{{4}})\b"
        match = re.search(pattern_exact, question)
        if match:
            day_str, month_name, year_str = match.groups()
            exact_date = date(int(year_str), self.MONTHS[month_name], int(day_str))
            result["exact_date"] = exact_date
            result["date_start"] = exact_date
            result["date_end"] = exact_date
            result["month"] = exact_date.month
            result["year"] = exact_date.year
            return result

        for month_name, month_num in self.MONTHS.items():
            match = re.search(rf"\b{month_name}\s+(\d{{4}})\b", question)
            if match:
                result["month"] = month_num
                result["year"] = int(match.group(1))
                return result

        return result

    def extract_signals(self, question: str) -> dict[str, Any]:
        """
        Extrait les signaux utiles au ranking depuis la question utilisateur.
        """
        q = self.normalize_text(question)
        date_filters = self._extract_date_filters(q)
        lexical_signals = self.lexical_service.extract_question_signals(q)

        strong_keywords = [
            kw for kw in lexical_signals["keywords"]
            if kw in self.STRONG_KEYWORDS
        ]

        is_cultural_query = bool(
            lexical_signals.get("is_cultural_query", False)
            or self.lexical_service.contains_any_term(q, self.lexical_service.CULTURAL_TERMS)
        )

        has_time_constraint = bool(
            date_filters["exact_date"]
            or date_filters["date_start"]
            or date_filters["date_end"]
            or (date_filters["month"] and date_filters["year"])
        )

        has_type_constraint = bool(lexical_signals["event_type"])
        has_price_constraint = bool(self._extract_price_filter(q))
        has_music_constraint = bool(lexical_signals["music_genre"])

        is_broad_query = not (
            has_time_constraint
            or has_type_constraint
            or has_price_constraint
            or has_music_constraint
        )

        return {
            "question_norm": q,
            "city": self._extract_city(q),
            "event_type": lexical_signals["event_type"],
            "music_genre": lexical_signals["music_genre"],
            "price_filter": self._extract_price_filter(q),
            "exact_date": date_filters["exact_date"],
            "date_start": date_filters["date_start"],
            "date_end": date_filters["date_end"],
            "month": date_filters["month"],
            "year": date_filters["year"],
            "keywords": lexical_signals["keywords"],
            "strong_keywords": strong_keywords,
            "is_cultural_query": is_cultural_query,
            "has_time_constraint": has_time_constraint,
            "has_type_constraint": has_type_constraint,
            "has_price_constraint": has_price_constraint,
            "has_music_constraint": has_music_constraint,
            "is_broad_query": is_broad_query,
        }
    # -------------------------------------------------------------------------
    # Matching métier
    # -------------------------------------------------------------------------

    def _vector_score_to_bonus(self, doc: Document) -> float:
        """
        Convertit le score brut FAISS en bonus exploitable.

        Hypothèse :
        plus `vector_score` est faible, plus le document est proche.

        Ici on borne volontairement l'impact du score vectoriel
        pour éviter qu'il domine les signaux métier.
        """
        md = doc.metadata or {}
        raw_score = md.get("vector_score")

        if raw_score is None:
            return 0.0

        try:
            raw_score = float(raw_score)
        except (TypeError, ValueError):
            return 0.0

        return max(0.0, 4.0 - raw_score * 4.0)

    def _supports_any_variant(
        self,
        doc_text: str,
        doc_title_keywords: set[str],
        doc_derived_terms: set[str],
        variants: list[str],
    ) -> bool:
        """
        Vérifie si au moins une variante métier est supportée par le document.
        """
        support_text = f"{doc_text} {' '.join(doc_title_keywords)} {' '.join(doc_derived_terms)}"
        return self.lexical_service.contains_any_term(support_text, variants)

    def _date_overlaps(
        self,
        event_start: date | None,
        event_end: date | None,
        query_start: date | None,
        query_end: date | None,
    ) -> bool:
        """
        Vérifie si deux périodes se chevauchent.
        """
        if not event_start:
            return False

        if not event_end:
            event_end = event_start

        if not query_start:
            return False

        if not query_end:
            query_end = query_start

        return event_start <= query_end and event_end >= query_start

    def _keyword_text_score(self, doc_text: str, keywords: list[str]) -> float:
        """
        Calcule un bonus léger à partir des mots-clés retrouvés dans
        le texte consolidé du document.
        """
        if not keywords:
            return 0.0

        hits = sum(1 for kw in keywords if kw in doc_text)
        return hits * self.SCORE_WEIGHTS["keyword_text"]

    def _keyword_title_score(
        self,
        doc_title_keywords: set[str],
        keywords: list[str],
    ) -> tuple[float, int, int]:
        """
        Calcule un bonus léger à partir des mots-clés retrouvés dans le titre.
        """
        if not keywords:
            return 0.0, 0, 0

        present = sum(1 for kw in keywords if kw in doc_title_keywords)
        absent = len(keywords) - present
        score = present * self.SCORE_WEIGHTS["keyword_title_present"]

        return score, present, absent

    def _derived_terms_score(
        self,
        doc_derived_terms: set[str],
        keywords: list[str],
    ) -> float:
        """
        Calcule un bonus lorsque des mots-clés de la question correspondent
        à des termes métier dérivés du document.
        """
        if not keywords or not doc_derived_terms:
            return 0.0

        hits = sum(1 for kw in keywords if kw in doc_derived_terms)
        return hits * self.SCORE_WEIGHTS["derived_term_present"]

    def _strong_keywords_score(
        self,
        doc_text: str,
        doc_title_keywords: set[str],
        doc_derived_terms: set[str],
        strong_keywords: list[str],
    ) -> float:
        """
        Calcule un bonus ou un malus léger sur les mots-clés forts.
        """
        if not strong_keywords:
            return 0.0

        score = 0.0
        support_text = f"{doc_text} {' '.join(doc_title_keywords)} {' '.join(doc_derived_terms)}"

        for kw in strong_keywords:
            if kw in support_text:
                score += self.SCORE_WEIGHTS["strong_keyword_present"]
            else:
                score += self.SCORE_WEIGHTS["strong_keyword_absent"]

        return score

    def _content_quality_score(self, doc: Document) -> float:
        """
        Attribue un léger bonus aux documents les plus riches.
        """
        score = 0.0
        score += self._doc_content_quality(doc) * self.SCORE_WEIGHTS["content_quality"]

        if self._doc_has_long_description(doc):
            score += self.SCORE_WEIGHTS["long_description_bonus"]

        if self._doc_is_single_day(doc) is True:
            score += self.SCORE_WEIGHTS["single_day_bonus"]

        return score

    def _is_musical_document(
        self,
        doc_text: str,
        doc_event_type: str,
        doc_music_genre: str,
        doc_derived_terms: set[str],
        doc_derived_music_terms: set[str],
    ) -> bool:
        """
        Détermine si le document semble réellement musical.
        """
        if doc_event_type in self.lexical_service.MUSICAL_EVENT_TYPES:
            return True

        if doc_music_genre:
            return True

        if doc_derived_music_terms:
            return True

        support_text = f"{doc_text} {' '.join(doc_derived_terms)} {' '.join(doc_derived_music_terms)}"
        return self.lexical_service.contains_any_term(
            support_text,
            [
                "concert",
                "live",
                "dj set",
                "showcase",
                "musique",
                "musical",
                "musicien",
                "groupe",
                "chanteur",
                "duo",
                "trio",
                "scene",
                "scène",
            ],
        )

    def _looks_too_generic_for_cultural_query(
        self,
        doc_text: str,
        doc_event_type: str,
        doc_derived_terms: set[str],
    ) -> bool:
        """
        Détecte des documents trop vagues pour une requête culturelle large.
        """
        if doc_event_type:
            return False

        if doc_derived_terms:
            return False

        generic_terms = [
            "journee portes ouvertes",
            "braderie",
            "destockage",
            "concours",
            "reparation",
            "velo",
            "familles",
            "patronage",
        ]

        return self.lexical_service.contains_any_term(doc_text, generic_terms)

    def _is_cultural_document(
        self,
        doc_text: str,
        doc_event_type: str,
        doc_derived_terms: set[str],
    ) -> bool:
        """
        Détermine si le document semble relever d'une proposition culturelle.
        """
        if doc_event_type in self.lexical_service.CULTURAL_EVENT_TYPES:
            return True

        if doc_derived_terms & self.lexical_service.CULTURAL_EVENT_TYPES:
            return True

        if self._looks_too_generic_for_cultural_query(
            doc_text=doc_text,
            doc_event_type=doc_event_type,
            doc_derived_terms=doc_derived_terms,
        ):
            return False

        support_text = f"{doc_text} {' '.join(doc_derived_terms)}"

        if self.lexical_service.contains_any_term(
            support_text,
            [
                "concert",
                "spectacle",
                "theatre",
                "théâtre",
                "projection",
                "cinema",
                "cinéma",
                "exposition",
                "expo",
                "vernissage",
                "festival",
                "lecture",
                "conte",
                "conference",
                "conférence",
                "atelier",
                "musique",
                "danse",
                "art",
                "photographie",
                "photo",
            ],
        ):
            return True

        return False
    
    def _event_type_match_level(
        self,
        requested_event_type: str,
        doc_event_type: str,
        doc_text: str,
        doc_title_keywords: set[str],
        doc_derived_terms: set[str],
    ) -> str:
        """
        Retourne le niveau de compatibilité entre le type demandé
        et le document.

        Returns
        -------
        str
            - "exact"
            - "variant"
            - "mismatch"
        """
        if not requested_event_type:
            return "variant"

        if requested_event_type == doc_event_type:
            return "exact"

        variants = self.lexical_service.EVENT_TYPE_TERMS.get(requested_event_type, [])
        if self._supports_any_variant(
            doc_text=doc_text,
            doc_title_keywords=doc_title_keywords,
            doc_derived_terms=doc_derived_terms,
            variants=variants,
        ):
            return "variant"

        if requested_event_type in doc_derived_terms:
            return "variant"

        return "mismatch"

    def _is_doc_compatible_with_query(
        self,
        doc: Document,
        signals: dict[str, Any],
    ) -> bool:
        """
        Vérifie si un document reste compatible avec les contraintes fortes
        de la question avant sélection finale.
        """
        doc_text = self._doc_text(doc)
        doc_title_keywords = self._doc_title_keywords(doc)
        doc_derived_terms = self._doc_derived_terms(doc)
        doc_derived_music_terms = self._doc_derived_music_terms(doc)
        doc_event_type = self._doc_event_type(doc)
        doc_music_genre = self._doc_music_genre(doc)

        if signals["event_type"]:
            match_level = self._event_type_match_level(
                requested_event_type=signals["event_type"],
                doc_event_type=doc_event_type,
                doc_text=doc_text,
                doc_title_keywords=doc_title_keywords,
                doc_derived_terms=doc_derived_terms,
            )
            if match_level == "mismatch":
                return False

        if signals["music_genre"]:
            genre_variants = self.lexical_service.MUSIC_GENRE_TERMS.get(
                signals["music_genre"], []
            )
            genre_supported = (
                signals["music_genre"] == doc_music_genre
                or signals["music_genre"] in doc_derived_music_terms
                or self._supports_any_variant(
                    doc_text=doc_text,
                    doc_title_keywords=doc_title_keywords,
                    doc_derived_terms=doc_derived_terms | doc_derived_music_terms,
                    variants=genre_variants,
                )
            )
            if not genre_supported:
                return False

            if not self._is_musical_document(
                doc_text=doc_text,
                doc_event_type=doc_event_type,
                doc_music_genre=doc_music_genre,
                doc_derived_terms=doc_derived_terms,
                doc_derived_music_terms=doc_derived_music_terms,
            ):
                return False

        if signals["is_cultural_query"]:
            if not self._is_cultural_document(
                doc_text=doc_text,
                doc_event_type=doc_event_type,
                doc_derived_terms=doc_derived_terms,
            ):
                return False

        return True

    def _duration_penalty(self, doc: Document, signals: dict[str, Any]) -> float:
        """
        Pénalise les événements très longs, surtout pour les requêtes datées.
        """
        duration_days = self._doc_duration_days(doc)

        if duration_days == 999999:
            return -4.0

        if signals["has_time_constraint"]:
            if duration_days > 120:
                return -12.0
            if duration_days > 60:
                return -8.0
            if duration_days > 30:
                return -4.0
            return 0.0

        if signals["is_broad_query"]:
            if duration_days > 180:
                return -6.0
            if duration_days > 90:
                return -3.0

        return 0.0

    # -------------------------------------------------------------------------
    # Scoring principal
    # -------------------------------------------------------------------------

    def score_document(self, doc: Document, signals: dict[str, Any]) -> float:
        """
        Attribue un score hybride à un document candidat.
        """
        md = doc.metadata or {}

        doc_text = self._doc_text(doc)
        doc_title_keywords = self._doc_title_keywords(doc)
        doc_derived_terms = self._doc_derived_terms(doc)
        doc_derived_music_terms = self._doc_derived_music_terms(doc)
        doc_city = self._doc_city(doc)
        doc_event_type = self._doc_event_type(doc)
        doc_music_genre = self._doc_music_genre(doc)

        first_date, last_date = self._doc_dates(doc)
        score = 0.0

        # 1. Signal vectoriel de base
        score += self._vector_score_to_bonus(doc)

        # 2. Ville
        if signals["city"]:
            if signals["city"] == doc_city:
                score += self.SCORE_WEIGHTS["city_bonus"]
            else:
                score += self.SCORE_WEIGHTS["wrong_city"]

        # 3. Type d'événement
        if signals["event_type"]:
            match_level = self._event_type_match_level(
                requested_event_type=signals["event_type"],
                doc_event_type=doc_event_type,
                doc_text=doc_text,
                doc_title_keywords=doc_title_keywords,
                doc_derived_terms=doc_derived_terms,
            )

            if match_level == "exact":
                score += self.SCORE_WEIGHTS["event_type_exact_match"]
            elif match_level == "variant":
                score += self.SCORE_WEIGHTS["event_type_variant_match"]
            else:
                score += self.SCORE_WEIGHTS["event_type_hard_mismatch"]

            if not doc_event_type:
                score += self.SCORE_WEIGHTS["empty_event_type_on_typed_query"]

        # 4. Genre musical
        if signals["music_genre"]:
            genre_variants = self.lexical_service.MUSIC_GENRE_TERMS.get(
                signals["music_genre"], []
            )
            genre_supported = (
                signals["music_genre"] == doc_music_genre
                or signals["music_genre"] in doc_derived_music_terms
                or self._supports_any_variant(
                    doc_text=doc_text,
                    doc_title_keywords=doc_title_keywords,
                    doc_derived_terms=doc_derived_terms | doc_derived_music_terms,
                    variants=genre_variants,
                )
            )

            if genre_supported:
                score += self.SCORE_WEIGHTS["music_genre_match"]
            else:
                score += self.SCORE_WEIGHTS["music_genre_missing"]

            if self._is_musical_document(
                doc_text=doc_text,
                doc_event_type=doc_event_type,
                doc_music_genre=doc_music_genre,
                doc_derived_terms=doc_derived_terms,
                doc_derived_music_terms=doc_derived_music_terms,
            ):
                score += self.SCORE_WEIGHTS["musical_doc_bonus"]
            else:
                score += self.SCORE_WEIGHTS["music_doc_missing"]

        # 5. Garde-fou croisé sur les concerts
        if signals["event_type"] == "concert":
            is_musical_doc = self._is_musical_document(
                doc_text=doc_text,
                doc_event_type=doc_event_type,
                doc_music_genre=doc_music_genre,
                doc_derived_terms=doc_derived_terms,
                doc_derived_music_terms=doc_derived_music_terms,
            )

            if not is_musical_doc:
                score += self.SCORE_WEIGHTS["event_type_hard_mismatch"]
            elif doc_event_type != "concert":
                score += self.SCORE_WEIGHTS["event_type_soft_mismatch"]

        # 6. Requête culturelle large
        if signals["is_cultural_query"]:
            if self._is_cultural_document(
                doc_text=doc_text,
                doc_event_type=doc_event_type,
                doc_derived_terms=doc_derived_terms,
            ):
                score += self.SCORE_WEIGHTS["cultural_doc_bonus"]
            else:
                score += self.SCORE_WEIGHTS["non_cultural_doc"]

            if not doc_event_type and not doc_derived_terms:
                score += self.SCORE_WEIGHTS["empty_event_type_on_cultural_query"]

        # 7. Tarification
        if signals["price_filter"] == "gratuit":
            if md.get("is_free") is True:
                score += self.SCORE_WEIGHTS["price_match"] + 2.0
            elif md.get("is_free") is False:
                score += self.SCORE_WEIGHTS["price_mismatch"]
            else:
                score -= 3.0

        if signals["price_filter"] == "payant":
            if md.get("is_free") is False:
                score += self.SCORE_WEIGHTS["price_match"] + 2.0
            elif md.get("is_free") is True:
                score += self.SCORE_WEIGHTS["price_mismatch"]
            else:
                score -= 3.0

        # 8. Date exacte ou intervalle
        if signals["date_start"] or signals["date_end"]:
            overlaps = self._date_overlaps(
                event_start=first_date,
                event_end=last_date,
                query_start=signals["date_start"],
                query_end=signals["date_end"],
            )
            if overlaps:
                if signals["exact_date"]:
                    score += self.SCORE_WEIGHTS["exact_date_match"]
                else:
                    score += self.SCORE_WEIGHTS["period_match"]
            else:
                score += self.SCORE_WEIGHTS["date_mismatch"]

        # 9. Mois / année
        elif signals["month"] and signals["year"] and first_date:
            if last_date is None:
                last_date = first_date

            month_start = date(signals["year"], signals["month"], 1)
            if signals["month"] == 12:
                month_end = date(signals["year"] + 1, 1, 1)
            else:
                month_end = date(signals["year"], signals["month"] + 1, 1)

            if first_date < month_end and last_date >= month_start:
                score += self.SCORE_WEIGHTS["month_match"]
            else:
                score += self.SCORE_WEIGHTS["month_mismatch"]

        # 10. Mots-clés et signaux faibles
        score += self._keyword_text_score(
            doc_text=doc_text,
            keywords=signals["keywords"],
        )

        keyword_title_score, n_present, n_absent = self._keyword_title_score(
            doc_title_keywords=doc_title_keywords,
            keywords=signals["keywords"],
        )
        score += keyword_title_score

        score += self._derived_terms_score(
            doc_derived_terms=doc_derived_terms,
            keywords=signals["keywords"],
        )

        score += self._strong_keywords_score(
            doc_text=doc_text,
            doc_title_keywords=doc_title_keywords,
            doc_derived_terms=doc_derived_terms | doc_derived_music_terms,
            strong_keywords=signals["strong_keywords"],
        )

        # 11. Qualité documentaire
        score += self._content_quality_score(doc)

        # 11 bis. Durée excessive
        score += self._duration_penalty(doc, signals)

        # 12. Traces de debug
        if doc.metadata is None:
            doc.metadata = {}

        doc.metadata["matched_title_keywords"] = n_present
        doc.metadata["missing_title_keywords"] = n_absent
        doc.metadata["detected_city_signal"] = signals["city"]
        doc.metadata["detected_event_type_signal"] = signals["event_type"]
        doc.metadata["detected_music_genre_signal"] = signals["music_genre"]
        doc.metadata["detected_price_signal"] = signals["price_filter"]
        doc.metadata["detected_date_start_signal"] = (
            str(signals["date_start"]) if signals["date_start"] else None
        )
        doc.metadata["detected_date_end_signal"] = (
            str(signals["date_end"]) if signals["date_end"] else None
        )
        doc.metadata["detected_is_cultural_query"] = signals["is_cultural_query"]

        return score

    # -------------------------------------------------------------------------
    # Tri et diversification
    # -------------------------------------------------------------------------

    def _recency_anchor_date(
        self,
        first_date: date | None,
        last_date: date | None,
    ) -> date:
        """
        Retourne la date de référence utilisée pour la récence.
        """
        if first_date:
            return first_date
        if last_date:
            return last_date
        return date.min

    def _temporal_distance_days(
        self,
        doc: Document,
        signals: dict[str, Any],
    ) -> int:
        """
        Calcule une distance temporelle simple entre le document
        et la contrainte temporelle de la question.
        """
        first_date, last_date = self._doc_dates(doc)

        if not first_date:
            return 999999

        if not last_date:
            last_date = first_date

        if signals.get("exact_date"):
            target = signals["exact_date"]
            if first_date <= target <= last_date:
                return 0
            if target < first_date:
                return (first_date - target).days
            return (target - last_date).days

        if signals.get("date_start") or signals.get("date_end"):
            start = signals.get("date_start") or signals.get("date_end")
            end = signals.get("date_end") or signals.get("date_start")
            if first_date <= end and last_date >= start:
                return 0
            if end < first_date:
                return (first_date - end).days
            return (start - last_date).days

        return 999999

    def _sort_key(
        self,
        doc: Document,
        final_score: float,
        signals: dict[str, Any],
    ) -> tuple[float, int, date, int]:
        """
        Construit la clé de tri finale.

        Ordre :
        1. score final
        2. proximité temporelle si contrainte temporelle
        3. date la plus récente
        4. durée la plus courte
        """
        first_date, last_date = self._doc_dates(doc)
        recency_date = self._recency_anchor_date(first_date, last_date)
        duration_days = self._doc_duration_days(doc)
        temporal_distance = self._temporal_distance_days(doc, signals)

        return (
            final_score,
            -temporal_distance,
            recency_date,
            -duration_days,
        )

    def _doc_similarity_signature(self, doc: Document) -> set[str]:
        """
        Construit une signature lexicale légère pour détecter
        les quasi-doublons.
        """
        md = doc.metadata or {}
        title = self.normalize_text(md.get("title", ""))
        location = self.normalize_text(md.get("location_name", ""))

        title_tokens = {
            token
            for token in re.findall(r"\b[a-z0-9]{3,}\b", title)
            if token not in self.lexical_service.STOPWORDS
        }

        location_tokens = {
            token
            for token in re.findall(r"\b[a-z0-9]{3,}\b", location)
            if token not in self.lexical_service.STOPWORDS
        }

        return title_tokens | location_tokens

    def _apply_diversification(
        self,
        scored_docs: list[tuple[Document, float]],
        signals: dict[str, Any],
        top_k: int,
    ) -> list[Document]:
        """
        Sélectionne les documents finaux avec une légère diversification.
        """
        if not scored_docs:
            return []

        selected: list[tuple[Document, float]] = []

        for doc, score in scored_docs:
            penalty = 0.0
            candidate_signature = self._doc_similarity_signature(doc)

            for selected_doc, selected_score in selected:
                selected_signature = self._doc_similarity_signature(selected_doc)

                if candidate_signature and selected_signature:
                    overlap = len(candidate_signature & selected_signature)
                    if overlap >= 3:
                        if score <= selected_score :
                            penalty += self.SCORE_WEIGHTS["similarity_penalty"]

            adjusted_score = score - penalty

            if doc.metadata is None:
                doc.metadata = {}

            doc.metadata["diversified_score"] = float(adjusted_score)

            selected.append((doc, adjusted_score))
            selected.sort(
                key=lambda item: self._sort_key(item[0], item[1], signals),
                reverse=True,
            )

            if len(selected) > top_k:
                selected = selected[:top_k]

        return [doc for doc, _ in selected]

    def _apply_strict_post_filter(
        self,
        raw_docs: list[Document],
        signals: dict[str, Any],
    ) -> list[Document]:
        """
        Applique un post-filtrage de cohérence sur les contraintes fortes.

        Principe :
        - si au moins un document satisfait les contraintes fortes,
          on ne garde que ces documents compatibles
        - sinon on conserve les candidats initiaux pour éviter
          un effondrement total du rappel
        """
        if not raw_docs:
            return []

        compatible_docs = [
            doc for doc in raw_docs
            if self._is_doc_compatible_with_query(doc, signals)
        ]

        if compatible_docs:
            return compatible_docs

        # Fallback doux seulement si requête très large
        if signals["is_broad_query"]:
            broad_docs = []
            for doc in raw_docs:
                doc_text = self._doc_text(doc)
                doc_event_type = self._doc_event_type(doc)
                doc_derived_terms = self._doc_derived_terms(doc)

                if self._is_cultural_document(
                    doc_text=doc_text,
                    doc_event_type=doc_event_type,
                    doc_derived_terms=doc_derived_terms,
                ):
                    broad_docs.append(doc)

            if broad_docs:
                return broad_docs

        return raw_docs

    # -------------------------------------------------------------------------
    # API publique
    # -------------------------------------------------------------------------

    def rank_documents(
        self,
        question: str,
        raw_docs: list[Document],
        top_k: int = 3,
    ) -> list[Document]:
        """
        Trie les documents candidats selon un ranking multi-critères.
        """
        if not raw_docs:
            return []

        signals = self.extract_signals(question)
        candidate_docs = self._apply_strict_post_filter(raw_docs, signals)

        scored_docs: list[tuple[Document, float]] = []
        for doc in candidate_docs:
            final_score = self.score_document(doc, signals)

            if doc.metadata is None:
                doc.metadata = {}

            first_date, last_date = self._doc_dates(doc)
            doc.metadata["final_score"] = float(final_score)
            doc.metadata["sort_recency_date"] = str(
                self._recency_anchor_date(first_date, last_date)
            )
            doc.metadata["duration_days"] = self._doc_duration_days(doc)

            scored_docs.append((doc, final_score))

        scored_docs.sort(
            key=lambda item: self._sort_key(item[0], item[1], signals),
            reverse=True,
        )

        return self._apply_diversification(
            scored_docs=scored_docs,
            signals=signals,
            top_k=top_k,
        )

    def rank_documents_with_scores(
        self,
        question: str,
        raw_docs: list[Document],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Retourne les documents triés avec leurs scores détaillés pour le debug.
        """
        if not raw_docs:
            return []

        signals = self.extract_signals(question)
        candidate_docs = self._apply_strict_post_filter(raw_docs, signals)

        rows: list[dict[str, Any]] = []
        for doc in candidate_docs:
            md = doc.metadata or {}
            score = self.score_document(doc, signals)

            first_date, last_date = self._doc_dates(doc)
            recency_date = self._recency_anchor_date(first_date, last_date)
            duration_days = self._doc_duration_days(doc)
            temporal_distance = self._temporal_distance_days(doc, signals)

            rows.append(
                {
                    "title": md.get("title", ""),
                    "location_name": md.get("location_name", ""),
                    "city": md.get("city", ""),
                    "region": md.get("region", ""),
                    "first_date": md.get("first_date", ""),
                    "last_date": md.get("last_date", ""),
                    "event_type": md.get("canonical_event_type", "") or md.get("event_type", ""),
                    "music_genre": md.get("music_genre", ""),
                    "price_info": md.get("price_info", ""),
                    "is_free": md.get("is_free"),
                    "keywords_title": md.get("keywords_title", []),
                    "derived_event_terms": md.get("derived_event_terms", []),
                    "derived_music_terms": md.get("derived_music_terms", []),
                    "content_quality": md.get("content_quality"),
                    "has_long_description": md.get("has_long_description"),
                    "is_single_day": md.get("is_single_day"),
                    "vector_score": md.get("vector_score"),
                    "final_score": score,
                    "recency_date": str(recency_date),
                    "temporal_distance_days": temporal_distance,
                    "duration_days": duration_days,
                    "matched_title_keywords": md.get("matched_title_keywords", 0),
                    "missing_title_keywords": md.get("missing_title_keywords", 0),
                    "detected_city_signal": md.get("detected_city_signal"),
                    "detected_event_type_signal": md.get("detected_event_type_signal"),
                    "detected_music_genre_signal": md.get("detected_music_genre_signal"),
                    "detected_price_signal": md.get("detected_price_signal"),
                    "detected_date_start_signal": md.get("detected_date_start_signal"),
                    "detected_date_end_signal": md.get("detected_date_end_signal"),
                    "detected_is_cultural_query": md.get("detected_is_cultural_query"),
                    "url": md.get("source_url", "") or md.get("url", ""),
                }
            )

        rows = sorted(
            rows,
            key=lambda row: (
                row["final_score"],
                -row["temporal_distance_days"],
                row["recency_date"],
                -row["duration_days"],
            ),
            reverse=True,
        )
        return rows[:top_k]