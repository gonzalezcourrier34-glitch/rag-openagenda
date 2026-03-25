"""
Service lexical partagé pour le pipeline RAG OpenAgenda.

Ce module centralise les opérations lexicales réutilisables dans
plusieurs services applicatifs, notamment :

- document_service
- filter_service
- retrieval_service

Il permet de :
- normaliser les textes
- extraire des tokens et des mots-clés
- détecter des types d'événements
- détecter des genres musicaux
- détecter quelques signaux métier simples dans une question
- dériver des termes métier utiles pour l'indexation et le retrieval
- fournir des helpers communs de matching lexical

Ce service ne gère pas :
- le parsing des dates
- le scoring métier
- la recherche vectorielle
- la génération de réponse
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


class LexicalService:
    """
    Service lexical partagé.

    Philosophie :
    - fournir une logique de normalisation cohérente pour tout le pipeline
    - limiter les divergences entre documents et questions
    - rester explicable et simple à maintenir
    - préférer des règles prudentes à des déductions trop agressives
    """

    STOPWORDS = {
        "le",
        "la",
        "les",
        "de",
        "du",
        "des",
        "un",
        "une",
        "et",
        "a",
        "au",
        "aux",
        "en",
        "pour",
        "avec",
        "sur",
        "dans",
        "par",
        "chez",
        "d",
        "l",
        "quel",
        "quels",
        "quelle",
        "quelles",
        "que",
        "qui",
        "ou",
        "où",
        "est",
        "sont",
        "ont",
        "lieu",
        "faire",
        "propose",
        "proposes",
        "proposé",
        "proposés",
        "proposée",
        "proposées",
        "evenement",
        "evenements",
        "événement",
        "événements",
        "culturel",
        "culturels",
        "culturelle",
        "culturelles",
        "des",
        "ces",
        "cet",
        "cette",
        "aujourd",
        "hui",
        "demain",
    }

    EVENT_TYPE_TERMS = {
        "exposition": [
            "exposition",
            "expositions",
            "expo",
            "expos",
            "vernissage",
            "vernissages",
            "installation",
            "installations",
            "retrospective",
            "retrospectives",
            "rétrospective",
            "rétrospectives",
        ],
        "concert": [
            "concert",
            "concerts",
            "live",
            "showcase",
            "showcases",
            "dj set",
            "dj sets",
            "performance musicale",
            "performances musicales",
            "performance musicale live",
            "session live",
            "sessions live",
            "soiree concert",
            "soirée concert",
            "soirees concert",
            "soirées concert",
        ],
        "visite": [
            "visite",
            "visites",
            "visite guidee",
            "visites guidees",
            "visite guidée",
            "visites guidées",
            "parcours guide",
            "parcours guides",
            "parcours guidé",
            "parcours guidés",
            "parcours commente",
            "parcours commentes",
            "parcours commenté",
            "parcours commentés",
        ],
        "conference": [
            "conference",
            "conferences",
            "conférence",
            "conférences",
            "rencontre",
            "rencontres",
            "debat",
            "debats",
            "débat",
            "débats",
            "table ronde",
            "tables rondes",
            "discussion",
            "discussions",
            "entretien",
            "entretiens",
            "conversation",
            "conversations",
            "masterclass",
        ],
        "atelier": [
            "atelier",
            "ateliers",
            "initiation",
            "initiations",
            "workshop",
            "workshops",
            "stage",
            "stages",
            "laboratoire",
            "laboratoires",
            "atelier participatif",
            "ateliers participatifs",
        ],
        "conte": [
            "conte",
            "contes",
            "conteur",
            "conteurs",
            "conteuse",
            "conteuses",
            "lecture contee",
            "lectures contees",
            "lecture contée",
            "lectures contées",
        ],
        "projection": [
            "projection",
            "projections",
            "film",
            "films",
            "cinema",
            "cinéma",
            "seance",
            "seances",
            "séance",
            "séances",
            "cine debat",
            "ciné débat",
            "cine-debat",
            "ciné-débat",
            "documentaire",
            "documentaires",
        ],
        "festival": [
            "festival",
            "festivals",
            "micro festival",
            "micro festivals",
            "micro-festival",
            "micro-festivals",
            "biennale",
            "biennales",
        ],
        "marche": [
            "marche",
            "marches",
            "marché",
            "marchés",
            "braderie",
            "braderies",
            "brocante",
            "brocantes",
            "vide grenier",
            "vide greniers",
            "vide-grenier",
            "vide-greniers",
            "pop up market",
            "pop up markets",
            "pop-up market",
            "pop-up markets",
        ],
        "spectacle": [
            "spectacle",
            "spectacles",
            "theatre",
            "théâtre",
            "representation",
            "representations",
            "représentation",
            "représentations",
            "performance",
            "performances",
            "one man show",
            "one man shows",
            "one-man-show",
            "one-man-shows",
        ],
        "lecture": [
            "lecture",
            "lectures",
            "lecture publique",
            "lectures publiques",
            "lecture musicale",
            "lectures musicales",
            "lecture performee",
            "lectures performees",
            "lecture performée",
            "lectures performées",
        ],
    }

    MUSIC_GENRE_TERMS = {
        "rock": ["rock", "garage", "punk", "noise", "metal", "grunge", "hardcore"],
        "jazz": ["jazz", "swing", "bebop"],
        "blues": ["blues"],
        "electro": ["electro", "électro", "techno", "house"],
        "folk": ["folk"],
        "rap": ["rap", "hip hop", "hip-hop"],
        "classique": ["classique", "baroque", "symphonique", "orchestre"],
    }

    AUDIENCE_TERMS = {
        "enfant": ["enfant", "enfants", "jeune public"],
        "famille": ["famille", "familial", "familiale", "tout public"],
    }

    FREE_MARKERS = [
        "gratuit",
        "gratuite",
        "gratuits",
        "gratuites",
        "entree libre",
        "entrée libre",
        "acces libre",
        "accès libre",
        "libre participation",
        "sans frais",
        "free",
    ]

    PAID_MARKERS = [
        "payant",
        "payante",
        "payants",
        "payantes",
        "billetterie",
        "tarif",
        "tarifs",
        "prix",
        "reservation obligatoire",
        "réservation obligatoire",
        "sur reservation",
        "sur réservation",
        "€",
        "euro",
        "euros",
    ]

    CULTURAL_EVENT_TYPES = {
        "exposition",
        "concert",
        "festival",
        "projection",
        "conference",
        "visite",
        "atelier",
        "spectacle",
        "conte",
        "lecture",
    }

    MUSICAL_EVENT_TYPES = {
        "concert",
        "festival",
        "spectacle",
    }

    CULTURAL_TERMS = [
        "culture",
        "culturel",
        "culturelle",
        "culturels",
        "culturelles",
        "exposition",
        "expo",
        "vernissage",
        "concert",
        "festival",
        "projection",
        "cinema",
        "cinéma",
        "conference",
        "conférence",
        "debat",
        "débat",
        "rencontre",
        "visite",
        "atelier",
        "spectacle",
        "theatre",
        "théâtre",
        "conte",
        "lecture",
        "patrimoine",
        "musee",
        "musée",
        "galerie",
        "art",
        "photo",
        "photographie",
        "architecture",
        "que faire",
        "sortie culturelle",
        "sorties culturelles",
    ]

    WEAK_CULTURAL_TERMS = [
        "que faire",
        "sortie",
        "sorties",
        "culture",
        "culturel",
        "culturelle",
        "culturels",
        "culturelles",
    ]

    # ------------------------------------------------------------------
    # Utilitaires texte de base
    # ------------------------------------------------------------------

    def safe(self, value: object) -> str:
        """
        Convertit une valeur potentiellement nulle en chaîne.
        """
        return "" if value is None else str(value)

    def clean_text(self, value: object) -> str:
        """
        Nettoie un texte sans en modifier le sens métier.

        Le nettoyage applique :
        - suppression des retours ligne
        - suppression des tabulations
        - réduction des espaces multiples
        """
        text = self.safe(value)
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_text(self, value: object) -> str:
        """
        Extrait un texte propre depuis une valeur arbitraire.

        La valeur peut être :
        - une chaîne simple
        - un dictionnaire multilingue
        - toute autre valeur convertible en texte
        """
        if isinstance(value, dict):
            for key in ("fr", "en"):
                if key in value and value[key]:
                    return self.clean_text(value[key])

            for candidate in value.values():
                text = self.clean_text(candidate)
                if text:
                    return text

            return ""

        return self.clean_text(value)

    def normalize_text(self, value: object) -> str:
        """
        Normalise un texte pour les comparaisons lexicales simples.

        Transformations appliquées :
        - minuscules
        - suppression des accents
        - suppression de la ponctuation non utile
        - compactage des espaces
        """
        text = self.extract_text(value).lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(char for char in text if not unicodedata.combining(char))
        text = re.sub(r"[^\w\s-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def tokenize_text(self, value: object) -> list[str]:
        """
        Découpe un texte normalisé en tokens simples.
        """
        text = self.normalize_text(value)
        return re.findall(r"\b[a-z0-9]{3,}\b", text)

    def join_texts(self, *values: object) -> str:
        """
        Concatène proprement plusieurs morceaux de texte.
        """
        return self.clean_text(
            " ".join(self.extract_text(value) for value in values if value)
        )

    # ------------------------------------------------------------------
    # Matching lexical simple
    # ------------------------------------------------------------------

    def contains_term(self, text: str, term: str) -> bool:
        """
        Vérifie la présence d'un terme complet dans un texte normalisé.
        """
        normalized_text = self.normalize_text(text)
        normalized_term = self.normalize_text(term)

        if not normalized_term:
            return False

        return re.search(rf"\b{re.escape(normalized_term)}\b", normalized_text) is not None

    def contains_any_term(self, text: str, terms: list[str]) -> bool:
        """
        Vérifie si au moins un terme complet est présent dans un texte normalisé.
        """
        normalized_text = self.normalize_text(text)
        normalized_terms = [
            self.normalize_text(term)
            for term in terms
            if self.normalize_text(term)
        ]

        for term in normalized_terms:
            if self.contains_term(normalized_text, term):
                return True

        return False

    def count_matching_terms(self, text: str, terms: list[str]) -> int:
        """
        Compte combien de termes distincts sont présents dans un texte.
        """
        normalized_text = self.normalize_text(text)
        normalized_terms = {
            self.normalize_text(term)
            for term in terms
            if self.normalize_text(term)
        }
        return sum(
            1 for term in normalized_terms
            if self.contains_term(normalized_text, term)
        )

    # ------------------------------------------------------------------
    # Extraction de mots-clés
    # ------------------------------------------------------------------

    def extract_keywords(
        self,
        text: object,
        *,
        min_len: int = 3,
        remove_stopwords: bool = True,
    ) -> list[str]:
        """
        Extrait des mots-clés simples et uniques depuis un texte.
        """
        tokens = self.tokenize_text(text)

        keywords = {
            token
            for token in tokens
            if len(token) >= min_len
            and (not remove_stopwords or token not in self.STOPWORDS)
        }

        return sorted(keywords)

    def extract_title_keywords(self, title: str) -> list[str]:
        """
        Extrait des mots-clés simples depuis un titre.
        """
        return self.extract_keywords(title, min_len=3, remove_stopwords=True)

    # ------------------------------------------------------------------
    # Détection de signaux métier
    # ------------------------------------------------------------------

    def extract_event_type(self, text: object) -> str | None:
        """
        Détecte un type d'événement explicite dans un texte.

        Cette fonction est utile surtout pour analyser les questions
        utilisateur. Elle reste volontairement prudente.
        """
        text_norm = self.normalize_text(text)

        for canonical_type, variants in self.EVENT_TYPE_TERMS.items():
            if self.contains_any_term(text_norm, variants):
                return canonical_type

        return None

    def extract_music_genre(self, text: object) -> str | None:
        """
        Détecte un genre musical explicite dans un texte.

        Cette fonction est utile surtout pour analyser les questions
        utilisateur. Elle reste volontairement prudente.
        """
        text_norm = self.normalize_text(text)

        for canonical_genre, variants in self.MUSIC_GENRE_TERMS.items():
            if self.contains_any_term(text_norm, variants):
                return canonical_genre

        return None

    def extract_audience_terms(self, text: object) -> list[str]:
        """
        Extrait des signaux simples de public cible.
        """
        text_norm = self.normalize_text(text)
        terms: set[str] = set()

        for canonical_audience, variants in self.AUDIENCE_TERMS.items():
            if self.contains_any_term(text_norm, variants):
                terms.add(canonical_audience)
                terms.update(
                    self.normalize_text(variant)
                    for variant in variants
                    if self.contains_term(text_norm, variant)
                )

        return sorted(terms)

    def extract_price_info(self, *values: object) -> tuple[str, bool | None]:
        """
        Déduit une information simple de tarification à partir de plusieurs textes.

        Retourne :
        - ("gratuit", True)
        - ("payant", False)
        - ("inconnu", None)
        """
        text = self.join_texts(*values)
        text_norm = self.normalize_text(text)

        has_free = any(marker in text_norm for marker in self.FREE_MARKERS)
        has_paid = any(marker in text_norm for marker in self.PAID_MARKERS)

        if has_free and not has_paid:
            return "gratuit", True

        if has_paid and not has_free:
            return "payant", False

        if has_free and has_paid:
            return "inconnu", None

        return "inconnu", None

    def is_cultural_query(self, text: object) -> bool:
        """
        Détecte si une question semble viser explicitement un besoin culturel.

        Logique prudente :
        - vrai si la question contient un type culturel explicite
        - vrai si elle contient un vocabulaire culturel fort
        - vrai si elle contient "que faire" + au moins un signal culturel
        - faux sinon
        """
        text_norm = self.normalize_text(text)

        if self.extract_event_type(text_norm) in self.CULTURAL_EVENT_TYPES:
            return True

        strong_terms = [
            "exposition",
            "expo",
            "vernissage",
            "concert",
            "festival",
            "projection",
            "cinema",
            "cinéma",
            "conference",
            "conférence",
            "debat",
            "débat",
            "visite",
            "atelier",
            "spectacle",
            "theatre",
            "théâtre",
            "conte",
            "lecture",
            "musee",
            "musée",
            "galerie",
            "patrimoine",
            "architecture",
            "photo",
            "photographie",
        ]

        if self.contains_any_term(text_norm, strong_terms):
            return True

        if self.contains_term(text_norm, "que faire") and self.contains_any_term(
            text_norm,
            ["montpellier", "paris", "sete", "toulouse", "lyon", "marseille", "bordeaux", "lille", "nantes"],
        ):
            return True

        if self.contains_any_term(text_norm, ["evenement culturel", "evenements culturels"]):
            return True

        return False

    # ------------------------------------------------------------------
    # Dérivation de termes métier pour les documents
    # ------------------------------------------------------------------

    def derive_event_terms(self, title: str, description: str, event_type: str) -> list[str]:
        """
        Déduit quelques termes métier utiles pour enrichir le texte indexé.

        Cette fonction peut être plus généreuse que l'inférence canonique,
        car son but est surtout d'aider la recherche et le matching léger.
        """
        title_norm = self.normalize_text(title)
        desc_norm = self.normalize_text(description)
        event_type_norm = self.normalize_text(event_type)

        terms: set[str] = set()

        if (
            self.contains_any_term(title_norm, self.EVENT_TYPE_TERMS["exposition"])
            or self.contains_any_term(event_type_norm, self.EVENT_TYPE_TERMS["exposition"])
            or (
                self.contains_any_term(desc_norm, self.EVENT_TYPE_TERMS["exposition"])
                and self.contains_any_term(
                    desc_norm,
                    ["art", "artiste", "galerie", "photo", "photographie"],
                )
            )
        ):
            terms.update(["exposition", "expo", "vernissage"])

        if (
            self.contains_any_term(title_norm, self.EVENT_TYPE_TERMS["concert"])
            or self.contains_any_term(event_type_norm, self.EVENT_TYPE_TERMS["concert"])
            or (
                self.contains_any_term(
                    desc_norm,
                    ["concert", "concerts", "musique", "musical", "live", "dj set"],
                )
                and self.contains_any_term(
                    desc_norm,
                    ["scene", "scène", "artiste", "groupe", "chanteur", "musicien", "duo"],
                )
            )
        ):
            terms.update(["concert", "musique", "live"])

        if (
            self.contains_any_term(title_norm, self.EVENT_TYPE_TERMS["atelier"])
            or self.contains_any_term(event_type_norm, self.EVENT_TYPE_TERMS["atelier"])
        ):
            terms.update(["atelier", "initiation", "workshop"])

        if (
            self.contains_any_term(title_norm, self.EVENT_TYPE_TERMS["conte"])
            or self.contains_any_term(event_type_norm, self.EVENT_TYPE_TERMS["conte"])
        ):
            terms.update(["conte", "contes", "conteur"])

        if (
            self.contains_any_term(title_norm, self.EVENT_TYPE_TERMS["conference"])
            or self.contains_any_term(event_type_norm, self.EVENT_TYPE_TERMS["conference"])
        ):
            terms.update(["conference", "conférence", "rencontre"])

        if (
            self.contains_any_term(title_norm, self.EVENT_TYPE_TERMS["visite"])
            or self.contains_any_term(event_type_norm, self.EVENT_TYPE_TERMS["visite"])
        ):
            terms.update(["visite", "visites"])

        if (
            self.contains_any_term(title_norm, self.EVENT_TYPE_TERMS["projection"])
            or self.contains_any_term(event_type_norm, self.EVENT_TYPE_TERMS["projection"])
        ):
            terms.update(["projection", "film", "cinema"])

        if (
            self.contains_any_term(title_norm, self.EVENT_TYPE_TERMS["festival"])
            or self.contains_any_term(event_type_norm, self.EVENT_TYPE_TERMS["festival"])
        ):
            terms.update(["festival"])

        if (
            self.contains_any_term(title_norm, self.EVENT_TYPE_TERMS["marche"])
            or self.contains_any_term(event_type_norm, self.EVENT_TYPE_TERMS["marche"])
        ):
            terms.update(["marche", "marché", "braderie"])

        if (
            self.contains_any_term(title_norm, self.EVENT_TYPE_TERMS["spectacle"])
            or self.contains_any_term(event_type_norm, self.EVENT_TYPE_TERMS["spectacle"])
        ):
            terms.update(["spectacle", "theatre", "théâtre"])

        if (
            self.contains_any_term(title_norm, self.EVENT_TYPE_TERMS["lecture"])
            or self.contains_any_term(event_type_norm, self.EVENT_TYPE_TERMS["lecture"])
        ):
            terms.update(["lecture"])

        if (
            self.contains_any_term(title_norm, ["photo", "photographie"])
            or self.contains_any_term(desc_norm, ["photo", "photographie"])
        ):
            terms.update(["photo", "photographie"])

        if (
            self.contains_any_term(title_norm, ["vinyl"])
            or self.contains_any_term(desc_norm, ["vinyl"])
        ):
            terms.update(["vinyl"])

        return sorted(terms)

    def derive_music_terms(self, title: str, description: str, event_type: str) -> list[str]:
        """
        Déduit quelques termes de genre musical utiles pour le ranking.

        Version prudente :
        - on ne déduit un genre que si le signal est explicite
        - on évite de transformer des indices faibles en faux genres
        """
        title_norm = self.normalize_text(title)
        desc_norm = self.normalize_text(description)
        event_type_norm = self.normalize_text(event_type)

        support_text = f"{title_norm} {desc_norm} {event_type_norm}"
        terms: set[str] = set()

        for canonical_genre, variants in self.MUSIC_GENRE_TERMS.items():
            if self.contains_any_term(support_text, variants):
                terms.add(canonical_genre)
                terms.update(
                    self.normalize_text(variant)
                    for variant in variants
                    if self.contains_term(support_text, variant)
                )

        return sorted(terms)

    # ------------------------------------------------------------------
    # Inférence canonique prudente pour les documents
    # ------------------------------------------------------------------

    def infer_canonical_event_type(
        self,
        title: str,
        description: str,
        event_type: str,
    ) -> str:
        """
        Déduit un type d'événement canonique simple.

        Logique prudente :
        1. priorité au titre
        2. puis au champ event_type brut
        3. puis à quelques signaux forts dans la description
        4. sinon on retourne vide ou le type brut normalisé si vraiment exploitable

        On préfère renvoyer une chaîne vide plutôt qu'un faux type.
        """
        title_norm = self.normalize_text(title)
        desc_norm = self.normalize_text(description)
        event_type_norm = self.normalize_text(event_type)

        strong_rules = [
            ("exposition", self.EVENT_TYPE_TERMS["exposition"]),
            ("concert", self.EVENT_TYPE_TERMS["concert"]),
            ("visite", self.EVENT_TYPE_TERMS["visite"]),
            ("conference", self.EVENT_TYPE_TERMS["conference"]),
            ("atelier", self.EVENT_TYPE_TERMS["atelier"]),
            ("conte", self.EVENT_TYPE_TERMS["conte"]),
            ("projection", self.EVENT_TYPE_TERMS["projection"]),
            ("festival", self.EVENT_TYPE_TERMS["festival"]),
            ("marche", self.EVENT_TYPE_TERMS["marche"]),
            ("spectacle", self.EVENT_TYPE_TERMS["spectacle"]),
            ("lecture", self.EVENT_TYPE_TERMS["lecture"]),
        ]

        for canonical_type, variants in strong_rules:
            if self.contains_any_term(title_norm, variants):
                return canonical_type

        for canonical_type, variants in strong_rules:
            if self.contains_any_term(event_type_norm, variants):
                return canonical_type

        if (
            self.contains_any_term(desc_norm, self.EVENT_TYPE_TERMS["exposition"])
            and self.contains_any_term(
                desc_norm,
                ["art", "artiste", "galerie", "photo", "photographie"],
            )
        ):
            return "exposition"

        if (
            self.contains_any_term(desc_norm, ["concert", "concerts", "musique", "live", "dj set"])
            and self.contains_any_term(
                desc_norm,
                ["scene", "scène", "groupe", "musicien", "chanteur", "duo"],
            )
        ):
            return "concert"

        if self.contains_any_term(desc_norm, self.EVENT_TYPE_TERMS["atelier"]):
            return "atelier"

        if self.contains_any_term(desc_norm, self.EVENT_TYPE_TERMS["conference"]):
            return "conference"

        if self.contains_any_term(desc_norm, self.EVENT_TYPE_TERMS["conte"]):
            return "conte"

        if self.contains_any_term(desc_norm, self.EVENT_TYPE_TERMS["projection"]):
            return "projection"

        if self.contains_any_term(desc_norm, self.EVENT_TYPE_TERMS["festival"]):
            return "festival"

        if self.contains_any_term(desc_norm, self.EVENT_TYPE_TERMS["marche"]):
            return "marche"

        if self.contains_any_term(desc_norm, self.EVENT_TYPE_TERMS["spectacle"]):
            return "spectacle"

        if self.contains_any_term(desc_norm, self.EVENT_TYPE_TERMS["lecture"]):
            return "lecture"

        if event_type_norm in {
            "exposition",
            "concert",
            "visite",
            "conference",
            "atelier",
            "conte",
            "projection",
            "festival",
            "marche",
            "spectacle",
            "lecture",
        }:
            return event_type_norm

        return ""

    def infer_canonical_music_genre(
        self,
        title: str,
        description: str,
        event_type: str,
    ) -> str:
        """
        Déduit un genre musical canonique simple lorsqu'il est identifiable.

        Version très prudente :
        - priorité au titre
        - puis à la description
        - aucun genre n'est inféré sur des indices faibles
        - on préfère retourner vide plutôt qu'un faux genre
        """
        title_norm = self.normalize_text(title)
        desc_norm = self.normalize_text(description)
        _ = self.normalize_text(event_type)

        for canonical_genre, variants in self.MUSIC_GENRE_TERMS.items():
            if self.contains_any_term(title_norm, variants):
                return canonical_genre

        for canonical_genre, variants in self.MUSIC_GENRE_TERMS.items():
            if self.contains_any_term(desc_norm, variants):
                return canonical_genre

        return ""

    # ------------------------------------------------------------------
    # Helpers plus structurés
    # ------------------------------------------------------------------

    def extract_question_signals(self, question: str) -> dict[str, Any]:
        """
        Extrait quelques signaux lexicaux simples depuis une question.

        Ce helper sert de base commune à filter_service et retrieval_service.

        Signaux renvoyés :
        - texte normalisé
        - mots-clés
        - type d'événement explicite
        - genre musical explicite
        - filtre tarifaire simple
        - signaux de public
        - indicateur de requête culturelle large
        """
        question_norm = self.normalize_text(question)
        price_info, _ = self.extract_price_info(question_norm)

        return {
            "question_norm": question_norm,
            "keywords": self.extract_keywords(question_norm),
            "event_type": self.extract_event_type(question_norm),
            "music_genre": self.extract_music_genre(question_norm),
            "price_filter": None if price_info == "inconnu" else price_info,
            "audience_terms": self.extract_audience_terms(question_norm),
            "is_cultural_query": self.is_cultural_query(question_norm),
        }

    def build_document_lexical_profile(
        self,
        *,
        title: str,
        description: str,
        long_description: str,
        event_type: str,
    ) -> dict[str, Any]:
        """
        Construit un petit profil lexical de document réutilisable.

        Ce helper peut être utilisé par document_service pour limiter
        la duplication locale.
        """
        full_description = self.join_texts(description, long_description)

        keywords_title = self.extract_title_keywords(title)
        derived_event_terms = self.derive_event_terms(
            title=title,
            description=full_description,
            event_type=event_type,
        )
        derived_music_terms = self.derive_music_terms(
            title=title,
            description=full_description,
            event_type=event_type,
        )
        audience_terms = self.extract_audience_terms(
            self.join_texts(title, full_description)
        )
        canonical_event_type = self.infer_canonical_event_type(
            title=title,
            description=full_description,
            event_type=event_type,
        )
        music_genre = self.infer_canonical_music_genre(
            title=title,
            description=full_description,
            event_type=event_type,
        )

        return {
            "keywords_title": keywords_title,
            "derived_event_terms": derived_event_terms,
            "derived_music_terms": derived_music_terms,
            "audience_terms": audience_terms,
            "canonical_event_type": canonical_event_type,
            "music_genre": music_genre,
        }