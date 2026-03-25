"""
Service de filtrage structuré pour le pipeline RAG OpenAgenda.

Ce module intervient avant la recherche vectorielle afin de réduire
le corpus candidat lorsque la question contient des contraintes fortes.

Rôle du service
---------------
Ce service permet de :

- normaliser la question utilisateur
- extraire des filtres structurés depuis cette question
- appliquer ces filtres sur les documents candidats
- produire, si besoin, une version détaillée du filtrage pour le debug

Philosophie
-----------
Le filtrage doit rester utile sans devenir trop agressif :

- lorsqu'une contrainte est explicite, on filtre fortement
- lorsqu'une information documentaire est absente, on reste plutôt souple
- on évite de supprimer trop tôt un bon document à cause d'une métadonnée incomplète

Ce service ne calcule aucun score métier.
Il se contente de sélectionner les documents compatibles avec les
contraintes explicites de la question.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

from langchain_core.documents import Document

from app.lexical_service import LexicalService


class FilterService:
    """
    Service de préfiltrage structuré des documents.

    Cette classe est utilisée avant la recherche vectorielle afin de
    réduire le bruit documentaire lorsque l'utilisateur exprime des
    contraintes fortes dans sa question.

    L'idée n'est pas de classer les documents, mais simplement de
    retirer ceux qui semblent manifestement incompatibles avec la demande.
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

    def __init__(self) -> None:
        """
        Initialise le service de filtrage avec son service lexical partagé.
        """
        self.lexical_service = LexicalService()

    # ------------------------------------------------------------------
    # Utilitaires de base
    # ------------------------------------------------------------------

    def _safe(self, value: object) -> str:
        """
        Convertit une valeur en chaîne de caractères.

        Parameters
        ----------
        value : object
            Valeur potentiellement nulle.

        Returns
        -------
        str
            Chaîne vide si la valeur est nulle, sinon conversion en texte.
        """
        return "" if value is None else str(value)

    def normalize_text(self, text: object) -> str:
        """
        Normalise un texte pour faciliter les comparaisons.

        Cette méthode délègue la normalisation au service lexical partagé.

        Parameters
        ----------
        text : object
            Texte brut à normaliser.

        Returns
        -------
        str
            Texte normalisé.
        """
        return self.lexical_service.normalize_text(text)

    def parse_iso_date(self, value: object) -> date | None:
        """
        Convertit une date ISO en objet `date`.

        Parameters
        ----------
        value : object
            Date brute au format supposé ISO.

        Returns
        -------
        date | None
            Date convertie si possible, sinon None.
        """
        raw = self._safe(value).strip()
        if not raw:
            return None

        try:
            return datetime.strptime(raw[:10], "%Y-%m-%d").date()
        except ValueError:
            return None

    def _build_date(self, year: int, month: int, day: int) -> date | None:
        """
        Construit une date de manière sécurisée.

        Parameters
        ----------
        year : int
            Année.
        month : int
            Mois.
        day : int
            Jour.

        Returns
        -------
        date | None
            Date valide si la combinaison est correcte, sinon None.
        """
        try:
            return date(year, month, day)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Lecture des métadonnées documentaires
    # ------------------------------------------------------------------

    def _doc_dates(self, doc: Document) -> tuple[date | None, date | None]:
        """
        Retourne les bornes de dates d'un document.

        Si la date de début existe mais pas la date de fin, on suppose
        que l'événement se déroule sur une seule journée.

        Parameters
        ----------
        doc : Document
            Document à inspecter.

        Returns
        -------
        tuple[date | None, date | None]
            Date de début, date de fin.
        """
        md = doc.metadata or {}
        first_date = self.parse_iso_date(md.get("first_date"))
        last_date = self.parse_iso_date(md.get("last_date"))

        if first_date and not last_date:
            last_date = first_date

        return first_date, last_date

    def _doc_city(self, doc: Document) -> str:
        """
        Retourne la ville normalisée d'un document.

        Parameters
        ----------
        doc : Document
            Document à inspecter.

        Returns
        -------
        str
            Ville normalisée.
        """
        md = doc.metadata or {}
        return md.get("city_norm") or self.normalize_text(md.get("city", ""))

    def _doc_event_type(self, doc: Document) -> str:
        """
        Retourne le type d'événement normalisé d'un document.

        On privilégie le champ `canonical_event_type` lorsqu'il est disponible.

        Parameters
        ----------
        doc : Document
            Document à inspecter.

        Returns
        -------
        str
            Type d'événement normalisé.
        """
        md = doc.metadata or {}
        return (
            self.normalize_text(md.get("canonical_event_type", ""))
            or md.get("event_type_norm")
            or self.normalize_text(md.get("event_type", ""))
        )

    def _doc_music_genre(self, doc: Document) -> str:
        """
        Retourne le genre musical normalisé d'un document.

        Parameters
        ----------
        doc : Document
            Document à inspecter.

        Returns
        -------
        str
            Genre musical normalisé.
        """
        md = doc.metadata or {}
        return md.get("music_genre_norm") or self.normalize_text(md.get("music_genre", ""))

    def _doc_search_text(self, doc: Document) -> str:
        """
        Retourne le texte de recherche consolidé et normalisé du document.

        Ce texte sert de filet de secours pour le matching lexical, même
        si certains champs structurés sont absents ou incomplets.

        Parameters
        ----------
        doc : Document
            Document à inspecter.

        Returns
        -------
        str
            Texte documentaire consolidé.
        """
        md = doc.metadata or {}
        search_text = md.get("search_text")
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
            f"{doc.page_content}"
        )
        return self.normalize_text(text)

    def _doc_is_free(self, doc: Document) -> bool | None:
        """
        Retourne l'information de gratuité d'un document.

        Parameters
        ----------
        doc : Document
            Document à inspecter.

        Returns
        -------
        bool | None
            True si gratuit, False si payant, None si inconnu.
        """
        md = doc.metadata or {}
        return md.get("is_free")

    def _doc_is_single_day(self, doc: Document) -> bool | None:
        """
        Indique si l'événement se déroule sur une seule journée.

        Parameters
        ----------
        doc : Document
            Document à inspecter.

        Returns
        -------
        bool | None
            Booléen si l'information est fiable, sinon None.
        """
        md = doc.metadata or {}
        value = md.get("is_single_day")

        if isinstance(value, bool):
            return value

        return None

    def _doc_duration_days(self, doc: Document) -> int | None:
        """
        Retourne la durée de l'événement en jours.

        Si la durée n'est pas disponible directement, elle est recalculée
        à partir des dates de début et de fin.

        Parameters
        ----------
        doc : Document
            Document à inspecter.

        Returns
        -------
        int | None
            Durée en jours, ou None si elle ne peut pas être estimée.
        """
        md = doc.metadata or {}
        value = md.get("duration_days")

        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            pass

        first_date, last_date = self._doc_dates(doc)
        if not first_date:
            return None

        if not last_date:
            last_date = first_date

        return max(1, (last_date - first_date).days + 1)

    def _doc_derived_event_terms(self, doc: Document) -> set[str]:
        """
        Retourne les termes métier dérivés pour le document.

        Parameters
        ----------
        doc : Document
            Document à inspecter.

        Returns
        -------
        set[str]
            Ensemble de termes normalisés.
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
        Retourne les termes musicaux dérivés pour le document.

        Parameters
        ----------
        doc : Document
            Document à inspecter.

        Returns
        -------
        set[str]
            Ensemble de termes normalisés.
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

    def _doc_audience_terms(self, doc: Document) -> set[str]:
        """
        Retourne les termes de public cible du document.

        Parameters
        ----------
        doc : Document
            Document à inspecter.

        Returns
        -------
        set[str]
            Ensemble de termes normalisés.
        """
        md = doc.metadata or {}
        terms = md.get("audience_terms", [])
        if not isinstance(terms, list):
            return set()

        return {
            self.normalize_text(term)
            for term in terms
            if self.normalize_text(term)
        }

    # ------------------------------------------------------------------
    # Extraction des contraintes depuis la question
    # ------------------------------------------------------------------

    def _extract_city(self, question_norm: str) -> str | None:
        """
        Détecte une ville connue dans la question normalisée.

        Parameters
        ----------
        question_norm : str
            Question déjà normalisée.

        Returns
        -------
        str | None
            Ville détectée si elle est explicitement mentionnée.
        """
        for city in sorted(self.KNOWN_CITIES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(city)}\b", question_norm):
                return city
        return None

    def _extract_duration_filter(self, question_norm: str) -> str | None:
        """
        Détecte une contrainte explicite sur la durée de l'événement.

        Parameters
        ----------
        question_norm : str
            Question déjà normalisée.

        Returns
        -------
        str | None
            - "single_day" pour une journée
            - "multi_day" pour plusieurs jours
            - None sinon
        """
        if re.search(
            r"\b(sur une journee|en une journee|a la journee|journee unique|ponctuel)\b",
            question_norm,
        ):
            return "single_day"

        if re.search(
            r"\b(sur plusieurs jours|plusieurs jours)\b",
            question_norm,
        ):
            return "multi_day"

        return None

    # ------------------------------------------------------------------
    # Utilitaires temporels
    # ------------------------------------------------------------------

    def _get_month_bounds(self, year: int, month: int) -> tuple[date, date]:
        """
        Retourne les bornes d'un mois sous forme d'intervalle semi-ouvert.

        Returns
        -------
        tuple[date, date]
            Date de début du mois, date du mois suivant.
        """
        month_start = date(year, month, 1)

        if month == 12:
            month_end = date(year + 1, 1, 1)
        else:
            month_end = date(year, month + 1, 1)

        return month_start, month_end

    def _get_year_bounds(self, year: int) -> tuple[date, date]:
        """
        Retourne les bornes d'une année sous forme d'intervalle semi-ouvert.
        """
        return date(year, 1, 1), date(year + 1, 1, 1)

    def _get_weekend_range(
        self,
        reference: date,
        next_weekend: bool = False,
    ) -> tuple[date, date]:
        """
        Retourne l'intervalle du week-end correspondant à une date de référence.

        Parameters
        ----------
        reference : date
            Date de départ utilisée comme repère.
        next_weekend : bool, default=False
            Si True, renvoie le week-end suivant.

        Returns
        -------
        tuple[date, date]
            Samedi, dimanche.
        """
        weekday = reference.weekday()

        if weekday == 5:
            saturday = reference
        elif weekday == 6:
            saturday = reference - timedelta(days=1)
        else:
            days_until_saturday = 5 - weekday
            saturday = reference + timedelta(days=days_until_saturday)

        if next_weekend:
            saturday = saturday + timedelta(days=7)

        sunday = saturday + timedelta(days=1)
        return saturday, sunday

    def _extract_explicit_weekend_range(
        self,
        question_norm: str,
    ) -> tuple[date | None, date | None]:
        """
        Extrait un week-end explicitement formulé dans la question.

        Exemples :
        - "week-end du 28 et 29 mars 2026"
        - "week-end du 28 mars 2026"

        Returns
        -------
        tuple[date | None, date | None]
            Date de début et date de fin du week-end détecté.
        """
        month_names = "|".join(self.MONTHS.keys())

        patterns = [
            rf"\bweek(?:\s|-)?end\s+(?:du|des)\s+(\d{{1,2}})\s*(?:et|au|a|-)\s*(\d{{1,2}})\s+({month_names})\s+(\d{{4}})\b",
            rf"\bweek(?:\s|-)?end\s+(?:du|des)\s+(\d{{1,2}})\s+({month_names})\s+(\d{{4}})\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, question_norm)
            if not match:
                continue

            groups = match.groups()

            if len(groups) == 4:
                day_start = int(groups[0])
                day_end = int(groups[1])
                month_name = groups[2]
                year_num = int(groups[3])

                month_num = self.MONTHS[month_name]
                start_date = self._build_date(year_num, month_num, day_start)
                end_date = self._build_date(year_num, month_num, day_end)

                if start_date and end_date:
                    if start_date <= end_date:
                        return start_date, end_date
                    return end_date, start_date

            if len(groups) == 3:
                day_num = int(groups[0])
                month_name = groups[1]
                year_num = int(groups[2])

                month_num = self.MONTHS[month_name]
                single_date = self._build_date(year_num, month_num, day_num)

                if single_date:
                    if single_date.weekday() == 5:
                        return single_date, single_date + timedelta(days=1)
                    if single_date.weekday() == 6:
                        return single_date - timedelta(days=1), single_date

                    saturday = single_date + timedelta(days=(5 - single_date.weekday()))
                    return saturday, saturday + timedelta(days=1)

        return None, None

    def _matches_weekend_with_max_span(
        self,
        doc: Document,
        start_date: date,
        end_date: date,
        max_span_days: int = 3,
    ) -> bool:
        """
        Vérifie qu'un document chevauche le week-end demandé
        et que sa durée totale reste raisonnable.

        Cette règle évite de faire remonter de très longs événements
        pour une demande très ciblée sur un week-end précis.
        """
        first_date, last_date = self._doc_dates(doc)

        if not first_date:
            return False

        if last_date is None:
            last_date = first_date

        overlaps = first_date <= end_date and last_date >= start_date
        if not overlaps:
            return False

        duration_days = max(1, (last_date - first_date).days + 1)
        return duration_days <= max_span_days

    def _extract_date_filters(self, question_norm: str) -> dict[str, Any]:
        """
        Extrait les filtres temporels présents dans la question.

        Cas gérés :
        - week-end explicite
        - ce week-end
        - week-end prochain
        - mois prochain
        - intervalle de dates
        - date exacte
        - mois + année
        - année seule
        """
        filters: dict[str, Any] = {
            "exact_date": None,
            "month": None,
            "year": None,
            "start_date": None,
            "end_date": None,
            "time_mode": None,
        }

        today = date.today()
        month_names = "|".join(self.MONTHS.keys())

        explicit_weekend_start, explicit_weekend_end = self._extract_explicit_weekend_range(
            question_norm
        )
        if explicit_weekend_start and explicit_weekend_end:
            filters["start_date"] = explicit_weekend_start
            filters["end_date"] = explicit_weekend_end
            filters["month"] = explicit_weekend_start.month
            filters["year"] = explicit_weekend_start.year
            filters["time_mode"] = "weekend_explicit"
            return filters

        if re.search(r"\bce\s+week(?:\s|-)?end\b", question_norm):
            start_date, end_date = self._get_weekend_range(
                reference=today,
                next_weekend=False,
            )
            filters["start_date"] = start_date
            filters["end_date"] = end_date
            filters["time_mode"] = "weekend_this"
            return filters

        if re.search(r"\bweek(?:\s|-)?end\s+prochain\b", question_norm):
            start_date, end_date = self._get_weekend_range(
                reference=today,
                next_weekend=True,
            )
            filters["start_date"] = start_date
            filters["end_date"] = end_date
            filters["time_mode"] = "weekend_next"
            return filters

        if re.search(r"\bmois\s+prochain\b", question_norm):
            if today.month == 12:
                next_month = 1
                next_year = today.year + 1
            else:
                next_month = today.month + 1
                next_year = today.year

            filters["month"] = next_month
            filters["year"] = next_year
            filters["time_mode"] = "next_month"
            return filters

        range_pattern = (
            rf"\b(?:du\s+)?(\d{{1,2}})\s*(?:au|a|et|-)\s*(\d{{1,2}})\s+"
            rf"({month_names})\s+(\d{{4}})\b"
        )
        match_range = re.search(range_pattern, question_norm)
        if match_range:
            day_start, day_end, month_name, year_str = match_range.groups()
            month_num = self.MONTHS[month_name]
            year_num = int(year_str)

            start_date = self._build_date(year_num, month_num, int(day_start))
            end_date = self._build_date(year_num, month_num, int(day_end))

            if start_date and end_date:
                if start_date > end_date:
                    start_date, end_date = end_date, start_date

                filters["start_date"] = start_date
                filters["end_date"] = end_date
                filters["month"] = month_num
                filters["year"] = year_num
                filters["time_mode"] = "date_range"
                return filters

        exact_pattern = rf"\b(\d{{1,2}})\s+({month_names})\s+(\d{{4}})\b"
        match_exact = re.search(exact_pattern, question_norm)
        if match_exact:
            day_str, month_name, year_str = match_exact.groups()
            month_num = self.MONTHS[month_name]
            year_num = int(year_str)

            exact_date = self._build_date(year_num, month_num, int(day_str))
            if exact_date:
                filters["exact_date"] = exact_date
                filters["month"] = month_num
                filters["year"] = year_num
                filters["time_mode"] = "exact_date"
                return filters

        for month_name, month_num in self.MONTHS.items():
            match_month = re.search(rf"\b{month_name}\s+(\d{{4}})\b", question_norm)
            if match_month:
                filters["month"] = month_num
                filters["year"] = int(match_month.group(1))
                filters["time_mode"] = "month_year"
                return filters

        match_year = re.search(r"\b(20\d{2})\b", question_norm)
        if match_year:
            filters["year"] = int(match_year.group(1))
            filters["time_mode"] = "year"
            return filters

        return filters

    def extract_filters(
        self,
        question: str,
        default_city: str | None = None,
    ) -> dict[str, Any]:
        """
        Extrait les filtres structurés depuis la question utilisateur.

        Parameters
        ----------
        question : str
            Question brute de l'utilisateur.
        default_city : str | None, default=None
            Ville par défaut à appliquer si la question ne cite pas explicitement de ville.

        Returns
        -------
        dict[str, Any]
            Ensemble des filtres structurés détectés.
        """
        question_norm = self.normalize_text(question)
        date_filters = self._extract_date_filters(question_norm)

        explicit_city = self._extract_city(question_norm)
        city = (
            explicit_city
            if explicit_city
            else self.normalize_text(default_city) if default_city else None
        )

        lexical_signals = self.lexical_service.extract_question_signals(question_norm)

        return {
            "question_norm": question_norm,
            "keywords": lexical_signals["keywords"],
            "city": city,
            "explicit_city": explicit_city,
            "event_type": lexical_signals["event_type"],
            "music_genre": lexical_signals["music_genre"],
            "audience_terms": lexical_signals.get("audience_terms", []),
            "is_cultural_query": lexical_signals.get("is_cultural_query", False),
            "duration_filter": self._extract_duration_filter(question_norm),
            "exact_date": date_filters["exact_date"],
            "month": date_filters["month"],
            "year": date_filters["year"],
            "start_date": date_filters["start_date"],
            "end_date": date_filters["end_date"],
            "time_mode": date_filters["time_mode"],
            "price_filter": lexical_signals.get("price_filter"),
        }

    # ------------------------------------------------------------------
    # Fonctions de matching
    # ------------------------------------------------------------------

    def matches_city(self, doc: Document, filters: dict[str, Any]) -> bool:
        """
        Vérifie si un document respecte le filtre de ville.
        """
        city = filters.get("city")
        if not city:
            return True

        return self._doc_city(doc) == city

    def _supports_any_variant(
        self,
        doc: Document,
        variants: list[str],
    ) -> bool:
        """
        Vérifie si un document supporte au moins une variante lexicale.
        """
        if not variants:
            return False

        doc_text = self._doc_search_text(doc)
        return self.lexical_service.contains_any_term(doc_text, variants)

    def _event_type_match_level(self, doc: Document, requested_event_type: str) -> str:
        """
        Retourne le niveau de compatibilité du document avec le type demandé.

        Returns
        -------
        str
            - "exact"
            - "variant"
            - "mismatch"
        """
        if not requested_event_type:
            return "variant"

        doc_event_type = self._doc_event_type(doc)
        if doc_event_type == requested_event_type:
            return "exact"

        variants = self.lexical_service.EVENT_TYPE_TERMS.get(requested_event_type, [])
        if self._supports_any_variant(doc, variants):
            return "variant"

        derived_terms = self._doc_derived_event_terms(doc)
        if requested_event_type in derived_terms:
            return "variant"

        return "mismatch"

    def _is_musical_document(self, doc: Document) -> bool:
        """
        Indique si le document semble réellement musical.
        """
        doc_event_type = self._doc_event_type(doc)
        if doc_event_type in self.lexical_service.MUSICAL_EVENT_TYPES:
            return True

        if self._doc_music_genre(doc):
            return True

        derived_music = self._doc_derived_music_terms(doc)
        if derived_music:
            return True

        doc_text = self._doc_search_text(doc)
        return self.lexical_service.contains_any_term(
            doc_text,
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

    def _is_cultural_document(self, doc: Document) -> bool:
        """
        Indique si le document semble culturel.
        """
        doc_event_type = self._doc_event_type(doc)
        if doc_event_type in self.lexical_service.CULTURAL_EVENT_TYPES:
            return True

        derived_terms = self._doc_derived_event_terms(doc)
        if derived_terms & self.lexical_service.CULTURAL_EVENT_TYPES:
            return True

        doc_text = self._doc_search_text(doc)
        return self.lexical_service.contains_any_term(
            doc_text,
            self.lexical_service.CULTURAL_TERMS,
        )

    def matches_event_type(self, doc: Document, filters: dict[str, Any]) -> bool:
        """
        Vérifie si un document respecte un type d'événement explicite.
        """
        event_type = filters.get("event_type")
        if not event_type:
            return True

        return self._event_type_match_level(doc, event_type) != "mismatch"

    def matches_music_genre(self, doc: Document, filters: dict[str, Any]) -> bool:
        """
        Vérifie si un document respecte un genre musical explicite.
        """
        music_genre = filters.get("music_genre")
        if not music_genre:
            return True

        doc_music_genre = self._doc_music_genre(doc)
        if music_genre and doc_music_genre == music_genre:
            return self._is_musical_document(doc)

        derived_music = self._doc_derived_music_terms(doc)
        if music_genre in derived_music:
            return self._is_musical_document(doc)

        variants = self.lexical_service.MUSIC_GENRE_TERMS.get(music_genre, [])
        if self._supports_any_variant(doc, variants):
            return self._is_musical_document(doc)

        return False

    def matches_cultural_scope(self, doc: Document, filters: dict[str, Any]) -> bool:
        """
        Vérifie si un document respecte une requête culturelle large.
        """
        if not filters.get("is_cultural_query"):
            return True

        return self._is_cultural_document(doc)

    def matches_duration(self, doc: Document, filters: dict[str, Any]) -> bool:
        """
        Vérifie si un document respecte une contrainte explicite de durée.
        """
        duration_filter = filters.get("duration_filter")
        if not duration_filter:
            return True

        is_single_day = self._doc_is_single_day(doc)
        duration_days = self._doc_duration_days(doc)

        if duration_filter == "single_day":
            if is_single_day is not None:
                return is_single_day is True
            if duration_days is not None:
                return duration_days == 1
            return True

        if duration_filter == "multi_day":
            if is_single_day is not None:
                return is_single_day is False
            if duration_days is not None:
                return duration_days > 1
            return True

        return True

    def matches_date(self, doc: Document, filters: dict[str, Any]) -> bool:
        """
        Vérifie si un document respecte le filtre temporel.

        La logique repose principalement sur le chevauchement
        de période lorsque cela est possible.
        """
        exact_date = filters.get("exact_date")
        month = filters.get("month")
        year = filters.get("year")
        start_date = filters.get("start_date")
        end_date = filters.get("end_date")

        if not any([exact_date, month, year, start_date, end_date]):
            return True

        first_date, last_date = self._doc_dates(doc)

        if not first_date:
            return False

        if last_date is None:
            last_date = first_date

        time_mode = filters.get("time_mode")

        if start_date and end_date:
            if time_mode in {"weekend_explicit", "weekend_this", "weekend_next"}:
                return self._matches_weekend_with_max_span(
                    doc=doc,
                    start_date=start_date,
                    end_date=end_date,
                    max_span_days=3,
                )

            return first_date <= end_date and last_date >= start_date

        if exact_date:
            return first_date <= exact_date <= last_date

        if month and year:
            month_start, month_end = self._get_month_bounds(year, month)
            return first_date < month_end and last_date >= month_start

        if year:
            year_start, year_end = self._get_year_bounds(year)
            return first_date < year_end and last_date >= year_start

        return True

    def matches_price(self, doc: Document, filters: dict[str, Any]) -> bool:
        """
        Vérifie si un document respecte le filtre de tarification.

        Règle retenue :
        - si la gratuité/payant est connue, on filtre strictement
        - si l'information est absente, on reste souple
        """
        price_filter = filters.get("price_filter")
        if not price_filter:
            return True

        is_free = self._doc_is_free(doc)

        if price_filter == "gratuit":
            if is_free is True:
                return True
            if is_free is False:
                return False
            return True

        if price_filter == "payant":
            if is_free is False:
                return True
            if is_free is True:
                return False
            return True

        return True

    def matches_audience(self, doc: Document, filters: dict[str, Any]) -> bool:
        """
        Vérifie si un document respecte une contrainte simple de public.
        """
        requested_terms = filters.get("audience_terms") or []
        if not requested_terms:
            return True

        doc_terms = self._doc_audience_terms(doc)
        if not doc_terms:
            return True

        requested_norm = {
            self.normalize_text(term)
            for term in requested_terms
            if self.normalize_text(term)
        }

        return bool(doc_terms & requested_norm)

    # ------------------------------------------------------------------
    # Pilotage global
    # ------------------------------------------------------------------

    def has_strong_filters(self, filters: dict[str, Any]) -> bool:
        """
        Indique si la question contient au moins une contrainte forte.

        Important :
        - la ville par défaut ne doit pas compter comme contrainte forte
        - seule une ville explicitement citée doit activer ce levier
        """
        return any(
            [
                filters.get("explicit_city"),
                filters.get("event_type"),
                filters.get("music_genre"),
                filters.get("is_cultural_query"),
                filters.get("duration_filter"),
                filters.get("audience_terms"),
                filters.get("exact_date"),
                filters.get("month"),
                filters.get("year"),
                filters.get("start_date"),
                filters.get("end_date"),
                filters.get("price_filter"),
            ]
        )

    def _run_filter_pipeline(
        self,
        filters: dict[str, Any],
        docs: list[Document],
    ) -> dict[str, list[Document]]:
        """
        Exécute le pipeline complet de filtrage étape par étape.

        Cette méthode centralise la logique commune entre :

        - `filter_documents`
        - `filter_documents_with_debug`

        Elle permet de conserver exactement la même logique de filtrage
        tout en évitant la duplication du code.

        Returns
        -------
        dict[str, list[Document]]
            Documents intermédiaires à chaque étape du pipeline.
        """
        docs_input = docs[:]
        docs_after_city = docs_input[:]
        docs_after_type = docs_after_city[:]
        docs_after_music = docs_after_type[:]
        docs_after_cultural = docs_after_music[:]
        docs_after_audience = docs_after_cultural[:]
        docs_after_duration = docs_after_audience[:]
        docs_after_date = docs_after_duration[:]
        docs_after_price = docs_after_date[:]

        if filters.get("city"):
            docs_after_city = [
                doc for doc in docs_after_city
                if self.matches_city(doc, filters)
            ]

        if not docs_after_city:
            return {
                "input": docs_input,
                "after_city": [],
                "after_type": [],
                "after_music": [],
                "after_cultural": [],
                "after_audience": [],
                "after_duration": [],
                "after_date": [],
                "after_price": [],
            }

        if not self.has_strong_filters(filters):
            return {
                "input": docs_input,
                "after_city": docs_after_city,
                "after_type": docs_after_city,
                "after_music": docs_after_city,
                "after_cultural": docs_after_city,
                "after_audience": docs_after_city,
                "after_duration": docs_after_city,
                "after_date": docs_after_city,
                "after_price": docs_after_city,
            }

        if filters.get("event_type"):
            docs_after_type = [
                doc for doc in docs_after_city
                if self.matches_event_type(doc, filters)
            ]
        else:
            docs_after_type = docs_after_city

        if not docs_after_type:
            return {
                "input": docs_input,
                "after_city": docs_after_city,
                "after_type": [],
                "after_music": [],
                "after_cultural": [],
                "after_audience": [],
                "after_duration": [],
                "after_date": [],
                "after_price": [],
            }

        if filters.get("music_genre"):
            docs_after_music = [
                doc for doc in docs_after_type
                if self.matches_music_genre(doc, filters)
            ]
        else:
            docs_after_music = docs_after_type

        if not docs_after_music:
            return {
                "input": docs_input,
                "after_city": docs_after_city,
                "after_type": docs_after_type,
                "after_music": [],
                "after_cultural": [],
                "after_audience": [],
                "after_duration": [],
                "after_date": [],
                "after_price": [],
            }

        if filters.get("is_cultural_query"):
            docs_after_cultural = [
                doc for doc in docs_after_music
                if self.matches_cultural_scope(doc, filters)
            ]
        else:
            docs_after_cultural = docs_after_music

        if not docs_after_cultural:
            return {
                "input": docs_input,
                "after_city": docs_after_city,
                "after_type": docs_after_type,
                "after_music": docs_after_music,
                "after_cultural": [],
                "after_audience": [],
                "after_duration": [],
                "after_date": [],
                "after_price": [],
            }

        if filters.get("audience_terms"):
            audience_docs = [
                doc for doc in docs_after_cultural
                if self.matches_audience(doc, filters)
            ]
            docs_after_audience = audience_docs if audience_docs else docs_after_cultural
        else:
            docs_after_audience = docs_after_cultural

        if filters.get("duration_filter"):
            duration_docs = [
                doc for doc in docs_after_audience
                if self.matches_duration(doc, filters)
            ]
            docs_after_duration = duration_docs if duration_docs else docs_after_audience
        else:
            docs_after_duration = docs_after_audience

        if any(
            [
                filters.get("exact_date"),
                filters.get("month"),
                filters.get("year"),
                filters.get("start_date"),
                filters.get("end_date"),
            ]
        ):
            docs_after_date = [
                doc for doc in docs_after_duration
                if self.matches_date(doc, filters)
            ]
        else:
            docs_after_date = docs_after_duration

        if not docs_after_date:
            return {
                "input": docs_input,
                "after_city": docs_after_city,
                "after_type": docs_after_type,
                "after_music": docs_after_music,
                "after_cultural": docs_after_cultural,
                "after_audience": docs_after_audience,
                "after_duration": docs_after_duration,
                "after_date": [],
                "after_price": [],
            }

        if filters.get("price_filter"):
            price_docs = [
                doc for doc in docs_after_date
                if self.matches_price(doc, filters)
            ]
            docs_after_price = price_docs if price_docs else docs_after_date
        else:
            docs_after_price = docs_after_date

        return {
            "input": docs_input,
            "after_city": docs_after_city,
            "after_type": docs_after_type,
            "after_music": docs_after_music,
            "after_cultural": docs_after_cultural,
            "after_audience": docs_after_audience,
            "after_duration": docs_after_duration,
            "after_date": docs_after_date,
            "after_price": docs_after_price,
        }

    def filter_documents(
        self,
        question: str,
        docs: list[Document],
        default_city: str | None = None,
    ) -> list[Document]:
        """
        Préfiltre les documents avant la recherche vectorielle.

        Cette méthode constitue l'entrée principale utilisée par le pipeline RAG.

        Parameters
        ----------
        question : str
            Question utilisateur.
        docs : list[Document]
            Documents candidats.
        default_city : str | None, default=None
            Ville par défaut appliquée si nécessaire.

        Returns
        -------
        list[Document]
            Documents retenus après filtrage.
        """
        if not docs:
            return []

        filters = self.extract_filters(
            question=question,
            default_city=default_city,
        )

        pipeline = self._run_filter_pipeline(filters, docs)
        return pipeline["after_price"]

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def _doc_to_debug_row(self, doc: Document) -> dict[str, Any]:
        """
        Convertit un document en ligne de debug légère.

        Cette structure permet d'inspecter facilement les documents
        retenus sans réafficher tout le contenu complet.
        """
        md = doc.metadata or {}
        return {
            "title": md.get("title", ""),
            "city": md.get("city", ""),
            "canonical_event_type": md.get("canonical_event_type", ""),
            "event_type": md.get("event_type", ""),
            "music_genre": md.get("music_genre", ""),
            "first_date": md.get("first_date", ""),
            "last_date": md.get("last_date", ""),
            "duration_days": md.get("duration_days"),
            "is_single_day": md.get("is_single_day"),
            "price_info": md.get("price_info", ""),
            "is_free": md.get("is_free"),
            "keywords_title": md.get("keywords_title", []),
            "derived_event_terms": md.get("derived_event_terms", []),
            "derived_music_terms": md.get("derived_music_terms", []),
            "audience_terms": md.get("audience_terms", []),
            "url": md.get("source_url", "") or md.get("url", ""),
        }

    def filter_documents_with_debug(
        self,
        question: str,
        docs: list[Document],
        default_city: str | None = None,
    ) -> dict[str, Any]:
        """
        Préfiltre les documents et retourne des informations de debug détaillées.

        Cette méthode est utile pour :
        - comprendre pourquoi certains documents disparaissent
        - inspecter les filtres extraits
        - mesurer l'effet de chaque étape du pipeline

        Returns
        -------
        dict[str, Any]
            Résultat structuré contenant les filtres, les volumes intermédiaires
            et la liste finale des documents retenus.
        """
        filters = self.extract_filters(
            question=question,
            default_city=default_city,
        )

        pipeline = self._run_filter_pipeline(filters, docs)
        final_docs = pipeline["after_price"]

        return {
            "filters": filters,
            "n_input_docs": len(pipeline["input"]),
            "n_after_city": len(pipeline["after_city"]),
            "n_after_type": len(pipeline["after_type"]),
            "n_after_music": len(pipeline["after_music"]),
            "n_after_cultural": len(pipeline["after_cultural"]),
            "n_after_audience": len(pipeline["after_audience"]),
            "n_after_duration": len(pipeline["after_duration"]),
            "n_after_date": len(pipeline["after_date"]),
            "n_after_price": len(final_docs),
            "docs": final_docs,
            "docs_debug": [self._doc_to_debug_row(doc) for doc in final_docs],
        }