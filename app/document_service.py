"""
Services de chargement, de préparation et d'exploitation des documents OpenAgenda.

Ce module centralise la logique documentaire du pipeline RAG.
Il permet de :

- rechercher les agendas liés à une zone géographique
- récupérer les événements depuis l'API OpenAgenda
- normaliser les données dans un format tabulaire stable
- construire un texte adapté à l'embedding
- convertir les événements en objets `Document` LangChain
- extraire des filtres simples depuis une question utilisateur
- filtrer et scorer des documents pour améliorer le retrieval

Les documents produits par ce module servent ensuite à construire
l'index vectoriel du système RAG et à améliorer la pertinence
des résultats renvoyés au moment de la recherche.
"""
from __future__ import annotations

import calendar
import json
import re
import unicodedata
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from langchain_core.documents import Document

from app.config import OPENAGENDA_API_KEY


# Racine du projet
BASE_DIR = Path(__file__).resolve().parents[1]

# Répertoire principal des données
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Fichiers par défaut
DEFAULT_JSON_PATH = RAW_DIR / "openagenda_events.json"
DEFAULT_CSV_PATH = PROCESSED_DIR / "openagenda_events.csv"

# Poids utilisés dans le scoring métier des documents
SCORE_WEIGHTS = {
    "city_match": 3.0,
    "location_match": 3.0,
    "event_type_match": 2.0,
    "keyword_match": 1.0,
    "date_filter_bonus": 1.0,
    "title_overlap_bonus": 0.5,
}


def get_default_date_window(days_back: int = 365) -> tuple[str, str]:
    """
    Calcule la fenêtre temporelle par défaut utilisée pour récupérer
    les événements OpenAgenda.

    La période couvre par défaut l'année précédente jusqu'à aujourd'hui.

    Parameters
    ----------
    days_back : int, default=365
        Nombre de jours à remonter dans le passé.

    Returns
    -------
    tuple[str, str]
        Couple `(date_from, date_to)` au format ISO YYYY-MM-DD.
    """
    today = date.today()
    date_from = (today - timedelta(days=days_back)).isoformat()
    date_to = today.isoformat()
    return date_from, date_to


def _safe(value: object) -> str:
    """
    Convertit une valeur potentiellement nulle en chaîne de caractères.

    Cette fonction utilitaire évite d'obtenir des valeurs `None`
    dans les champs texte utilisés pour construire les documents
    ou les métadonnées.

    Parameters
    ----------
    value : object
        Valeur à convertir.

    Returns
    -------
    str
        Chaîne vide si la valeur est nulle, sinon représentation texte.
    """
    return "" if value is None else str(value)


def _clean_text_field(value: object) -> str:
    """
    Nettoie un champ texte simple sans casser la structure globale.

    Parameters
    ----------
    value : object
        Valeur à nettoyer.

    Returns
    -------
    str
        Texte nettoyé.
    """
    text = _safe(value).replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _build_multiline_text(parts: list[tuple[str, str]]) -> str:
    """
    Construit un texte multiligne à partir d'une liste de couples
    `(libellé, valeur)`.

    Les lignes vides sont ignorées afin d'éviter les documents bruités.

    Parameters
    ----------
    parts : list[tuple[str, str]]
        Liste ordonnée des champs à intégrer.

    Returns
    -------
    str
        Texte consolidé prêt pour l'embedding.
    """
    lines = []
    for label, value in parts:
        cleaned_value = _clean_text_field(value)
        if cleaned_value:
            lines.append(f"{label} : {cleaned_value}")
    return "\n".join(lines).strip()


def _parse_iso_date(value: str) -> date | None:
    """
    Convertit une chaîne ISO YYYY-MM-DD en objet date.

    Parameters
    ----------
    value : str
        Date à convertir.

    Returns
    -------
    date | None
        Date convertie ou `None` si le format est invalide.
    """
    value = _safe(value).strip()
    if not value:
        return None

    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def normalize_text(text: str) -> str:
    """
    Normalise un texte pour faciliter les comparaisons lexicales.

    La normalisation applique :
    - passage en minuscules
    - suppression des accents
    - nettoyage des espaces multiples

    Parameters
    ----------
    text : str
        Texte à normaliser.

    Returns
    -------
    str
        Texte normalisé.
    """
    text = _safe(text).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"\s+", " ", text)
    return text


def save_events_to_json(events: list[dict], output_path: str | Path) -> None:
    """
    Sauvegarde la liste brute des événements au format JSON.

    Parameters
    ----------
    events : list[dict]
        Liste brute des événements récupérés depuis l'API.
    output_path : str | Path
        Chemin du fichier JSON de sortie.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)


def save_events_to_csv(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Sauvegarde les événements normalisés au format CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des événements normalisés.
    output_path : str | Path
        Chemin du fichier CSV de sortie.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False, encoding="utf-8")


def load_events_from_json(input_path: str | Path) -> list[dict]:
    """
    Charge une liste d'événements depuis un fichier JSON.

    Parameters
    ----------
    input_path : str | Path
        Chemin du fichier JSON à lire.

    Returns
    -------
    list[dict]
        Liste brute des événements.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier JSON introuvable : {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_events_from_csv(input_path: str | Path) -> pd.DataFrame:
    """
    Charge les événements normalisés depuis un fichier CSV.

    Parameters
    ----------
    input_path : str | Path
        Chemin du fichier CSV à lire.

    Returns
    -------
    pd.DataFrame
        DataFrame des événements.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier CSV introuvable : {path}")

    return pd.read_csv(path).fillna("")


def build_event_document(event: dict) -> Document:
    """
    Construit un objet `Document` LangChain à partir d'un événement.

    Cette fonction transforme un dictionnaire d'événement OpenAgenda
    en document indexable par le pipeline RAG.

    Le contenu principal (`page_content`) est basé en priorité sur
    la colonne `text_for_embedding` si elle existe. Sinon, il est
    reconstruit à partir des principaux champs descriptifs.

    Les métadonnées sont conservées pour permettre :
    - la traçabilité des événements
    - l'affichage des informations dans l'API
    - le filtrage métier au moment du retrieval
    - le reranking documentaire

    Parameters
    ----------
    event : dict
        Dictionnaire contenant les informations d'un événement.

    Returns
    -------
    Document
        Document LangChain contenant :
        - un texte exploitable pour la recherche sémantique
        - des métadonnées descriptives
    """
    event_uid = _safe(event.get("event_uid"))
    title = _clean_text_field(event.get("title"))
    description = _clean_text_field(event.get("description"))
    location_name = _clean_text_field(event.get("location_name"))
    city = _clean_text_field(event.get("city"))
    region = _clean_text_field(event.get("region"))
    first_date = _clean_text_field(event.get("first_date"))
    last_date = _clean_text_field(event.get("last_date"))
    event_type = _clean_text_field(event.get("event_type"))
    source_url = _clean_text_field(event.get("source_url"))
    agenda_uid = _clean_text_field(event.get("agenda_uid"))
    text_for_embedding = _safe(event.get("text_for_embedding")).strip()

    page_content = text_for_embedding if text_for_embedding else _build_multiline_text(
        [
            ("Titre", title),
            ("Description", description),
            ("Lieu", location_name),
            ("Ville", city),
            ("Région", region),
            ("Date de début", first_date),
            ("Date de fin", last_date),
            ("Type d'événement", event_type),
        ]
    )

    metadata = {
        "doc_id": f"openagenda_{event_uid}",
        "chunk_id": f"openagenda_{event_uid}_0",
        "source": "openagenda",
        "event_uid": event_uid,
        "agenda_uid": agenda_uid,
        "title": title,
        "city": city,
        "region": region,
        "location_name": location_name,
        "first_date": first_date,
        "last_date": last_date,
        "event_type": event_type,
        "url": source_url,
        "lang": "fr",
    }

    return Document(page_content=page_content, metadata=metadata)


def search_agendas_for_zone(
    zone: str,
    size: int = 30,
    official: bool = True
) -> list[dict]:
    """
    Recherche les agendas OpenAgenda correspondant à une zone géographique.

    Cette fonction interroge l'endpoint des agendas OpenAgenda afin
    d'identifier les agendas liés à une ville, une région ou une zone donnée.

    Parameters
    ----------
    zone : str
        Zone géographique recherchée, par exemple une ville ou une région.
    size : int, default=30
        Nombre maximum d'agendas à récupérer.
    official : bool, default=True
        Si True, limite la recherche aux agendas officiels.

    Returns
    -------
    list[dict]
        Liste des agendas retournés par l'API.

    Raises
    ------
    ValueError
        Si la clé API OpenAgenda est absente ou si la réponse est inattendue.
    RuntimeError
        Si l'appel HTTP échoue.
    """
    if not OPENAGENDA_API_KEY:
        raise ValueError("OPENAGENDA_API_KEY manquante dans le fichier .env")

    url = "https://api.openagenda.com/v2/agendas"
    headers = {"key": OPENAGENDA_API_KEY}
    params = {
        "search": zone,
        "size": size,
    }

    if official:
        params["official"] = 1

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Erreur réseau lors de la recherche des agendas pour la zone '{zone}'."
        ) from exc

    if "agendas" not in payload:
        raise ValueError(f"Réponse inattendue : {list(payload.keys())}")

    return payload["agendas"]


def fetch_openagenda_events(
    agenda_uid: str,
    zone: str,
    scope: str = "city",
    limit: int = 300,
    max_pages: int = 50,
    days_back: int = 365,
) -> list[dict]:
    """
    Récupère les événements d'un agenda OpenAgenda avec pagination.

    Cette fonction interroge l'API OpenAgenda pour extraire les événements
    d'un agenda sur une période glissante allant de l'année précédente
    jusqu'à la date actuelle.

    Parameters
    ----------
    agenda_uid : str
        Identifiant unique de l'agenda OpenAgenda.
    zone : str
        Zone géographique utilisée dans la recherche.
    scope : str, default="city"
        Portée géographique du filtre, par exemple `city`.
    limit : int, default=300
        Nombre maximum d'événements récupérés par requête.
    max_pages : int, default=50
        Nombre maximum de pages à interroger.
    days_back : int, default=365
        Fenêtre temporelle glissante en jours.

    Returns
    -------
    list[dict]
        Liste complète des événements récupérés.

    Raises
    ------
    ValueError
        Si la clé API est absente ou si la réponse API est inattendue.
    RuntimeError
        Si l'appel HTTP échoue.
    """
    if not OPENAGENDA_API_KEY:
        raise ValueError("OPENAGENDA_API_KEY manquante dans le fichier .env")

    date_from, date_to = get_default_date_window(days_back=days_back)

    all_events: list[dict] = []
    offset = 0

    for _ in range(max_pages):
        url = f"https://openagenda.com/agendas/{agenda_uid}/events.json"

        params = {
            "key": OPENAGENDA_API_KEY,
            "limit": limit,
            "offset": offset,
            "oaq[passed]": 1,
            "oaq[from]": date_from,
            "oaq[to]": date_to,
            "oaq[what]": zone,
            "oaq[scope]": scope,
            "oaq[order]": "latest",
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Erreur réseau lors de la récupération des événements "
                f"de l'agenda '{agenda_uid}' pour la zone '{zone}'."
            ) from exc

        if "events" not in payload:
            raise ValueError(f"Réponse inattendue de l'API : {list(payload.keys())}")

        events = payload["events"]

        if not events:
            break

        all_events.extend(events)

        total = payload.get("total", 0)
        offset += limit

        if offset >= total:
            break

    return all_events


def _is_event_content_rich_enough(event: dict[str, Any]) -> bool:
    """
    Vérifie qu'un événement contient un minimum d'information utile
    pour être conservé dans le corpus.

    Parameters
    ----------
    event : dict[str, Any]
        Événement OpenAgenda brut ou normalisé.

    Returns
    -------
    bool
        `True` si l'événement est suffisamment informatif.
    """
    title = _clean_text_field(event.get("title") or event.get("title.fr"))
    description = _clean_text_field(event.get("description") or event.get("description.fr"))
    location_name = _clean_text_field(event.get("location_name") or event.get("location.name"))

    return any([title, description, location_name])


def fetch_and_save_events(
    zone: str = "Montpellier",
    scope: str = "city",
    json_path: str | Path = DEFAULT_JSON_PATH,
    csv_path: str | Path = DEFAULT_CSV_PATH
) -> pd.DataFrame:
    """
    Récupère les événements OpenAgenda, puis les sauvegarde en JSON et CSV.

    La recherche des agendas s'effectue à partir de la zone fournie.
    Le paramètre `scope` est utilisé ensuite au moment de récupérer
    les événements de chaque agenda.

    Parameters
    ----------
    zone : str, default="Montpellier"
        Zone géographique ciblée.
    scope : str, default="city"
        Portée géographique utilisée pour filtrer les événements.
    json_path : str | Path, default=DEFAULT_JSON_PATH
        Chemin du fichier JSON de sauvegarde.
    csv_path : str | Path, default=DEFAULT_CSV_PATH
        Chemin du fichier CSV de sauvegarde.

    Returns
    -------
    pd.DataFrame
        DataFrame normalisé des événements.
    """
    agendas = search_agendas_for_zone(zone)

    if not agendas:
        return pd.DataFrame()

    all_events: list[dict] = []

    for agenda in agendas:
        agenda_uid = _safe(agenda.get("uid"))
        if not agenda_uid:
            continue

        try:
            events = fetch_openagenda_events(
                agenda_uid=agenda_uid,
                zone=zone,
                scope=scope,
            )
            valid_events = [event for event in events if _is_event_content_rich_enough(event)]
            all_events.extend(valid_events)
        except RuntimeError:
            continue

    if not all_events:
        return pd.DataFrame()

    save_events_to_json(all_events, json_path)

    df_events = normalize_events(all_events)

    if df_events.empty:
        return df_events

    df_events = df_events.drop_duplicates(subset=["event_uid"]).reset_index(drop=True)
    df_events = build_indexable_text(df_events)

    save_events_to_csv(df_events, csv_path)

    return df_events


def normalize_events(events: list[dict]) -> pd.DataFrame:
    """
    Normalise les événements OpenAgenda dans un schéma tabulaire stable.

    Cette fonction transforme la structure JSON brute de l'API
    en DataFrame pandas homogène avec les colonnes attendues
    par le pipeline RAG.

    Elle gère :
    - l'aplatissement du JSON
    - le renommage des colonnes utiles
    - l'ajout des colonnes manquantes
    - un nettoyage simple des champs texte

    Parameters
    ----------
    events : list[dict]
        Liste brute des événements récupérés depuis l'API.

    Returns
    -------
    pd.DataFrame
        DataFrame normalisé contenant les colonnes standard du pipeline.
    """
    schema_cols = [
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

    if not events:
        return pd.DataFrame(columns=schema_cols)

    df = pd.json_normalize(events)

    rename_map = {
        "uid": "event_uid",
        "agenda.uid": "agenda_uid",
        "title.fr": "title",
        "description.fr": "description",
        "canonicalUrl": "source_url",
        "firstDate": "first_date",
        "lastDate": "last_date",
        "type": "event_type",
        "eventType": "event_type",
        "location.name": "location_name",
        "location.city": "city",
        "location.region": "region",
    }

    existing_cols = [c for c in rename_map if c in df.columns]
    df = df[existing_cols].copy()
    df = df.rename(columns={c: rename_map[c] for c in existing_cols})

    for col in schema_cols:
        if col not in df.columns:
            df[col] = ""

    for col in schema_cols:
        df[col] = df[col].fillna("").astype(str).map(_clean_text_field)

    return df[schema_cols].copy()


def build_indexable_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit le texte consolidé destiné à l'embedding.

    Cette fonction crée une colonne `text_for_embedding` en concaténant
    les principaux champs descriptifs de l'événement.

    L'objectif est de produire un texte plus riche et cohérent pour :
    - la vectorisation
    - la recherche sémantique
    - le rappel contextuel du système RAG
    - le retrieval sur des contraintes métier comme la date, le lieu ou le type

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame d'événements normalisés.

    Returns
    -------
    pd.DataFrame
        Copie du DataFrame enrichie avec la colonne `text_for_embedding`.
    """
    df = df.copy()

    text_columns = [
        "title",
        "description",
        "location_name",
        "city",
        "region",
        "first_date",
        "last_date",
        "event_type",
    ]

    for col in text_columns:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").map(_clean_text_field)

    df["text_for_embedding"] = df.apply(
        lambda row: _build_multiline_text(
            [
                ("Titre", row["title"]),
                ("Description", row["description"]),
                ("Lieu", row["location_name"]),
                ("Ville", row["city"]),
                ("Région", row["region"]),
                ("Date de début", row["first_date"]),
                ("Date de fin", row["last_date"]),
                ("Type d'événement", row["event_type"]),
            ]
        ),
        axis=1,
    )

    return df


def get_known_metadata_values(documents: list[Document]) -> dict[str, set[str]]:
    """
    Extrait les principales valeurs de métadonnées connues du corpus.

    Cette fonction parcourt les documents disponibles afin de récupérer
    les villes, lieux et types d'événements présents dans le corpus.
    Ces valeurs servent ensuite à détecter automatiquement certains
    filtres mentionnés dans une question utilisateur.

    Parameters
    ----------
    documents : list[Document]
        Documents du corpus.

    Returns
    -------
    dict[str, set[str]]
        Dictionnaire contenant :
        - `cities`
        - `locations`
        - `event_types`
    """
    values = {
        "cities": set(),
        "locations": set(),
        "event_types": set(),
    }

    for doc in documents:
        md = doc.metadata or {}

        city = md.get("city")
        location = md.get("location_name")
        event_type = md.get("event_type")

        if city:
            values["cities"].add(normalize_text(city))
        if location:
            values["locations"].add(normalize_text(location))
        if event_type:
            values["event_types"].add(normalize_text(event_type))

    return values


def _resolve_relative_date_filters(question: str) -> tuple[str | None, str | None]:
    """
    Interprète quelques expressions temporelles relatives fréquentes.

    Parameters
    ----------
    question : str
        Question utilisateur normalisée.

    Returns
    -------
    tuple[str | None, str | None]
        Couple `(date_start, date_end)` au format ISO.
    """
    q = normalize_text(question)
    today = date.today()

    if "aujourd hui" in q or "aujourdhui" in q or "aujourd'hui" in q:
        iso = today.isoformat()
        return iso, iso

    if "demain" in q:
        d = (today + timedelta(days=1)).isoformat()
        return d, d

    if "ce soir" in q:
        iso = today.isoformat()
        return iso, iso

    if "ce week end" in q or "weekend" in q or "week end" in q or "week-end" in q:
        days_until_saturday = (5 - today.weekday()) % 7
        saturday = today + timedelta(days=days_until_saturday)
        sunday = saturday + timedelta(days=1)
        return saturday.isoformat(), sunday.isoformat()

    if "semaine prochaine" in q:
        days_until_next_monday = (7 - today.weekday()) % 7
        if days_until_next_monday == 0:
            days_until_next_monday = 7
        monday = today + timedelta(days=days_until_next_monday)
        sunday = monday + timedelta(days=6)
        return monday.isoformat(), sunday.isoformat()

    return None, None


def parse_date_filters(question: str) -> tuple[str | None, str | None]:
    """
    Extrait une plage de dates simple à partir d'une question.

    La fonction gère notamment :
    - un mois et une année, par exemple "septembre 2025"
    - une date précise, par exemple "20 septembre 2025"
    - un intervalle simple, par exemple "20 au 21 septembre 2025"
    - un format numérique simple, par exemple "20/09/2025"
    - quelques expressions relatives fréquentes

    Parameters
    ----------
    question : str
        Question utilisateur.

    Returns
    -------
    tuple[str | None, str | None]
        Couple `(date_start, date_end)` au format YYYY-MM-DD.
        Si aucune date n'est détectée, les deux valeurs sont `None`.
    """
    q = normalize_text(question)

    relative_start, relative_end = _resolve_relative_date_filters(q)
    if relative_start or relative_end:
        return relative_start, relative_end

    month_map = {
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

    range_pattern = (
        r"(\d{1,2})\s+(?:au|et)\s+(\d{1,2})\s+"
        r"(janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre)"
        r"\s+(\d{4})"
    )
    match = re.search(range_pattern, q)
    if match:
        day1, day2, month_name, year = match.groups()
        month_num = month_map[month_name]
        return (
            date(int(year), month_num, int(day1)).isoformat(),
            date(int(year), month_num, int(day2)).isoformat(),
        )

    numeric_exact_pattern = r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"
    match = re.search(numeric_exact_pattern, q)
    if match:
        day, month_num, year = match.groups()
        exact_date = date(int(year), int(month_num), int(day)).isoformat()
        return exact_date, exact_date

    exact_pattern = (
        r"(\d{1,2})\s+"
        r"(janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre)"
        r"\s+(\d{4})"
    )
    match = re.search(exact_pattern, q)
    if match:
        day, month_name, year = match.groups()
        month_num = month_map[month_name]
        exact_date = date(int(year), month_num, int(day)).isoformat()
        return exact_date, exact_date

    for month_name, month_num in month_map.items():
        pattern = rf"{month_name}\s+(\d{{4}})"
        match = re.search(pattern, q)
        if match:
            year = int(match.group(1))
            _, last_day = calendar.monthrange(year, month_num)
            return (
                date(year, month_num, 1).isoformat(),
                date(year, month_num, last_day).isoformat(),
            )

    return None, None


def _match_known_values(
    question: str,
    known_values: set[str],
    min_similarity: float = 0.88,
) -> list[str]:
    """
    Tente d'associer des valeurs connues du corpus à la question utilisateur.

    La détection combine :
    - recherche par sous-chaîne
    - comparaison approximative légère sur les segments de texte

    Parameters
    ----------
    question : str
        Question normalisée.
    known_values : set[str]
        Valeurs connues du corpus.
    min_similarity : float, default=0.88
        Seuil minimum de similarité pour retenir une correspondance.

    Returns
    -------
    list[str]
        Liste des valeurs détectées dans la question.
    """
    matches = set()

    for value in known_values:
        if not value:
            continue

        if value in question:
            matches.add(value)
            continue

        question_tokens = question.split()
        value_tokens = value.split()
        window_size = max(1, len(value_tokens))

        for idx in range(len(question_tokens) - window_size + 1):
            candidate = " ".join(question_tokens[idx: idx + window_size])
            similarity = SequenceMatcher(None, candidate, value).ratio()
            if similarity >= min_similarity:
                matches.add(value)
                break

    return sorted(matches)


def extract_filters_from_question(
    question: str,
    documents: list[Document],
) -> dict[str, Any]:
    """
    Extrait des filtres métier à partir d'une question utilisateur.

    Cette fonction identifie de manière heuristique :
    - les villes présentes dans la question
    - les lieux mentionnés
    - les types d'événements
    - une éventuelle plage de dates
    - des mots-clés libres utiles pour le reranking

    L'extraction s'appuie à la fois sur :
    - les métadonnées connues du corpus
    - des règles simples de normalisation et de détection
    - une détection approximative légère pour tolérer certaines variantes

    Parameters
    ----------
    question : str
        Question utilisateur.
    documents : list[Document]
        Documents du corpus ou documents candidats.

    Returns
    -------
    dict[str, Any]
        Dictionnaire de filtres métier.
    """
    q = normalize_text(question)
    known = get_known_metadata_values(documents)
    date_start, date_end = parse_date_filters(q)

    filters = {
        "cities": _match_known_values(q, known["cities"]),
        "locations": _match_known_values(q, known["locations"]),
        "event_types": _match_known_values(q, known["event_types"]),
        "date_start": date_start,
        "date_end": date_end,
        "keywords": [],
    }

    stopwords = {
        "quel", "quels", "quelle", "quelles",
        "y", "a", "t", "il", "des", "de", "du", "le", "la", "les",
        "au", "aux", "dans", "en", "sur", "pour", "avec",
        "ont", "lieu", "evenement", "evenements",
        "proposes", "propose", "culturels", "culturel",
        "septembre", "octobre", "novembre", "decembre",
        "janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout",
        "week", "end", "weekend", "soir", "demain", "aujourd", "hui",
        "semaine", "prochaine", "cette", "ce",
    }

    tokens = re.findall(r"\b[a-z]{3,}\b", q)
    filters["keywords"] = [
        token
        for token in tokens
        if token not in stopwords
        and token not in filters["cities"]
        and token not in filters["locations"]
        and token not in filters["event_types"]
    ]

    return filters


def doc_matches_filters(doc: Document, filters: dict[str, Any]) -> bool:
    """
    Vérifie si un document respecte les filtres extraits de la question.

    Le filtrage porte principalement sur :
    - la ville
    - le lieu
    - le type d'événement
    - la compatibilité temporelle

    Parameters
    ----------
    doc : Document
        Document candidat.
    filters : dict[str, Any]
        Filtres extraits depuis la question.

    Returns
    -------
    bool
        `True` si le document reste compatible avec les filtres,
        sinon `False`.
    """
    md = doc.metadata or {}

    city = normalize_text(md.get("city", ""))
    location_name = normalize_text(md.get("location_name", ""))
    event_type = normalize_text(md.get("event_type", ""))

    first_date_obj = _parse_iso_date(_safe(md.get("first_date", "")).strip())
    last_date_obj = _parse_iso_date(_safe(md.get("last_date", "")).strip())

    if filters["cities"] and city not in filters["cities"]:
        return False

    if filters["locations"]:
        if not any(loc in location_name for loc in filters["locations"]):
            return False

    if filters["event_types"]:
        if not any(evt in event_type for evt in filters["event_types"]):
            return False

    date_start = _parse_iso_date(filters.get("date_start"))
    date_end = _parse_iso_date(filters.get("date_end"))

    if date_start and last_date_obj and last_date_obj < date_start:
        return False

    if date_end and first_date_obj and first_date_obj > date_end:
        return False

    return True


def score_document(question: str, doc: Document, filters: dict[str, Any]) -> float:
    """
    Attribue un score métier à un document pour le reranking final.

    Le score combine plusieurs signaux simples :
    - correspondance de ville
    - correspondance de lieu
    - correspondance de type
    - présence de mots-clés dans le contenu ou les métadonnées
    - bonus léger si une contrainte temporelle est détectée
    - léger bonus si le titre recoupe la formulation de la question

    Le score reste volontairement heuristique et interprétable afin
    de pouvoir être expliqué facilement lors d'une soutenance ou
    ajusté rapidement selon les résultats observés.

    Parameters
    ----------
    question : str
        Question utilisateur.
    doc : Document
        Document candidat.
    filters : dict[str, Any]
        Filtres extraits depuis la question.

    Returns
    -------
    float
        Score métier utilisé pour trier les documents.
    """
    md = doc.metadata or {}

    city = normalize_text(md.get("city", ""))
    location_name = normalize_text(md.get("location_name", ""))
    event_type = normalize_text(md.get("event_type", ""))
    title = normalize_text(md.get("title", ""))

    full_text = normalize_text(
        f"{doc.page_content} "
        f"{md.get('title', '')} "
        f"{md.get('location_name', '')} "
        f"{md.get('city', '')} "
        f"{md.get('event_type', '')}"
    )

    score = 0.0

    if filters["cities"] and city in filters["cities"]:
        score += SCORE_WEIGHTS["city_match"]

    if filters["locations"] and any(loc in location_name for loc in filters["locations"]):
        score += SCORE_WEIGHTS["location_match"]

    if filters["event_types"] and any(evt in event_type for evt in filters["event_types"]):
        score += SCORE_WEIGHTS["event_type_match"]

    keyword_hits = sum(1 for keyword in filters["keywords"] if keyword in full_text)
    score += keyword_hits * SCORE_WEIGHTS["keyword_match"]

    if filters.get("date_start") or filters.get("date_end"):
        score += SCORE_WEIGHTS["date_filter_bonus"]

    normalized_question = normalize_text(question)
    question_tokens = set(normalized_question.split())
    title_tokens = set(title.split())

    if question_tokens & title_tokens:
        score += SCORE_WEIGHTS["title_overlap_bonus"]

    return score


def _load_events_dataframe_from_api(
    zone: str,
    scope: str,
    json_path: str | Path,
    csv_path: str | Path,
) -> pd.DataFrame:
    """
    Charge les événements depuis l'API OpenAgenda puis retourne
    le DataFrame normalisé.

    Parameters
    ----------
    zone : str
        Zone géographique ciblée.
    scope : str
        Portée géographique utilisée pour filtrer les événements.
    json_path : str | Path
        Chemin du fichier JSON local.
    csv_path : str | Path
        Chemin du fichier CSV local.

    Returns
    -------
    pd.DataFrame
        DataFrame normalisé des événements.
    """
    return fetch_and_save_events(
        zone=zone,
        scope=scope,
        json_path=json_path,
        csv_path=csv_path,
    )


def _load_events_dataframe_from_json(
    json_path: str | Path,
) -> pd.DataFrame:
    """
    Charge les événements depuis un fichier JSON local puis les normalise.

    Parameters
    ----------
    json_path : str | Path
        Chemin du fichier JSON local.

    Returns
    -------
    pd.DataFrame
        DataFrame normalisé enrichi pour l'embedding.
    """
    events = load_events_from_json(json_path)
    filtered_events = [event for event in events if _is_event_content_rich_enough(event)]
    df_events = normalize_events(filtered_events)
    return build_indexable_text(df_events)


def _load_events_dataframe_from_csv(
    csv_path: str | Path,
) -> pd.DataFrame:
    """
    Charge les événements depuis un fichier CSV local.

    Si la colonne `text_for_embedding` est absente, elle est reconstruite.

    Parameters
    ----------
    csv_path : str | Path
        Chemin du fichier CSV local.

    Returns
    -------
    pd.DataFrame
        DataFrame prêt à être converti en documents.
    """
    df_events = load_events_from_csv(csv_path)

    if "text_for_embedding" not in df_events.columns:
        df_events = build_indexable_text(df_events)

    return df_events


def load_documents(
    zone: str = "Montpellier",
    scope: str = "city",
    source: str = "api",
    json_path: str | Path = DEFAULT_JSON_PATH,
    csv_path: str | Path = DEFAULT_CSV_PATH
) -> list[Document]:
    """
    Charge et prépare les documents OpenAgenda pour le pipeline RAG.

    Parameters
    ----------
    zone : str, default="Montpellier"
        Zone géographique ciblée.
    scope : str, default="city"
        Portée géographique utilisée pour filtrer les événements.
    source : str, default="api"
        Source utilisée pour charger les événements :
        - "api" : récupération depuis l'API OpenAgenda
        - "json" : lecture depuis un fichier JSON local
        - "csv" : lecture depuis un fichier CSV local
    json_path : str | Path, default=DEFAULT_JSON_PATH
        Chemin du fichier JSON local.
    csv_path : str | Path, default=DEFAULT_CSV_PATH
        Chemin du fichier CSV local.

    Returns
    -------
    list[Document]
        Liste des documents LangChain prêts à être indexés.
    """
    if source == "api":
        df_events = _load_events_dataframe_from_api(
            zone=zone,
            scope=scope,
            json_path=json_path,
            csv_path=csv_path,
        )
    elif source == "json":
        df_events = _load_events_dataframe_from_json(json_path=json_path)
    elif source == "csv":
        df_events = _load_events_dataframe_from_csv(csv_path=csv_path)
    else:
        raise ValueError("source doit valoir 'api', 'json' ou 'csv'")

    if df_events.empty:
        return []

    if "event_uid" in df_events.columns:
        df_events = df_events.drop_duplicates(subset=["event_uid"]).reset_index(drop=True)

    documents = [
        build_event_document(row.to_dict())
        for _, row in df_events.iterrows()
        if _is_event_content_rich_enough(row.to_dict())
    ]

    return documents