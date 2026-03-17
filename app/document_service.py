"""
Services de chargement et de préparation des documents OpenAgenda.

Ce module gère l'ensemble de la préparation documentaire du pipeline RAG.
Il permet de :

- rechercher les agendas liés à une zone géographique
- récupérer les événements depuis l'API OpenAgenda
- normaliser les données dans un format tabulaire stable
- construire un texte adapté à l'embedding
- convertir les événements en objets `Document` LangChain

Les documents produits par ce module servent ensuite à construire
l'index vectoriel du système RAG.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from langchain_core.documents import Document

from app.config import OPENAGENDA_API_KEY

# Fenêtre temporelle utilisée pour récupérer les événements :
# de l'année précédente jusqu'à aujourd'hui.
today = date.today()
date_from = (today - timedelta(days=365)).isoformat()
date_to = today.isoformat()

# Racine du projet
BASE_DIR = Path(__file__).resolve().parents[1]

# Répertoire principal des données
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Fichiers par défaut
DEFAULT_JSON_PATH = RAW_DIR / "openagenda_events.json"
DEFAULT_CSV_PATH = PROCESSED_DIR / "openagenda_events.csv"

def _safe(value: object) -> str:
    """
    Convertit une valeur potentiellement nulle en chaîne de caractères.

    Cette fonction utilitaire évite d'obtenir des valeurs `None`
    dans les champs texte utilisés pour construire les documents
    ou les métadonnées.

    Parameters
    ----------
    value : Any
        Valeur à convertir.

    Returns
    -------
    str
        Chaîne vide si la valeur est nulle, sinon représentation texte.
    """
    return "" if value is None else str(value)

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
    - un éventuel filtrage ou enrichissement futur

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
    # Récupération sécurisée des champs utiles
    event_uid = _safe(event.get("event_uid"))
    title = _safe(event.get("title"))
    description = _safe(event.get("description"))
    location_name = _safe(event.get("location_name"))
    city = _safe(event.get("city"))
    region = _safe(event.get("region"))
    first_date = _safe(event.get("first_date"))
    last_date = _safe(event.get("last_date"))
    event_type = _safe(event.get("event_type"))
    source_url = _safe(event.get("source_url"))
    agenda_uid = _safe(event.get("agenda_uid"))
    text_for_embedding = _safe(event.get("text_for_embedding"))

    # Si un texte consolidé existe déjà, on l'utilise.
    # Sinon, on reconstruit un contenu simple à partir des champs utiles.
    page_content = text_for_embedding if text_for_embedding else "\n".join(
        part for part in [
            f"Titre : {title}",
            f"Description : {description}",
            f"Lieu : {location_name}",
            f"Ville : {city}",
            f"Région : {region}",
            f"Date de début : {first_date}",
            f"Date de fin : {last_date}",
            f"Type d'événement : {event_type}"
        ]
        if part and part.split(" : ", 1)[-1].strip()
    )

    # Métadonnées associées au document
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
        "lang": "fr"
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
    requests.HTTPError
        Si l'appel HTTP échoue.
    """
    # Vérification de la présence de la clé API
    if not OPENAGENDA_API_KEY:
        raise ValueError("OPENAGENDA_API_KEY manquante dans le fichier .env")

    url = "https://api.openagenda.com/v2/agendas"
    headers = {"key": OPENAGENDA_API_KEY}
    params = {
        "search": zone,
        "size" : size #limite voulu pour economie
    }

    # Filtre optionnel pour ne récupérer que les agendas officiels
    if official:
        params["official"] = 1

    # Appel API
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    # Validation minimale de la structure de réponse
    if "agendas" not in payload:
        raise ValueError(f"Réponse inattendue : {list(payload.keys())}")

    return payload["agendas"]

def fetch_openagenda_events(
    agenda_uid: str,
    zone: str,
    scope: str = "city",
    limit: int = 300,
    max_pages: int = 50
) -> list[dict]:
    """
    Récupère les événements d'un agenda OpenAgenda avec pagination.

    Cette fonction interroge l'API OpenAgenda pour extraire les événements
    d'un agenda sur une période glissante allant d'aujourd'hui à 365 jours.

    La pagination permet de récupérer un volume important d'événements
    sans dépendre d'une seule réponse API.

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
        
    Returns
    -------
    list[dict]
        Liste complète des événements récupérés.

    Raises
    ------
    ValueError
        Si la clé API est absente ou si la réponse API est inattendue.
    requests.HTTPError
        Si l'appel HTTP échoue.
    """
    # Vérification de la clé API
    if not OPENAGENDA_API_KEY:
        raise ValueError("OPENAGENDA_API_KEY manquante dans le fichier .env")

    all_events = []
    offset = 0

    # Boucle de pagination
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
            "oaq[order]": "latest"
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        # Vérifie que la réponse contient bien une liste d'événements
        if "events" not in payload:
            raise ValueError(f"Réponse inattendue de l'API : {list(payload.keys())}")

        events = payload["events"]

        # Si plus aucun événement, on arrête la pagination
        if not events:
            break

        all_events.extend(events)

        total = payload.get("total", 0)
        offset += limit

        # Arrêt si tous les événements ont été récupérés
        if offset >= total:
            break

    return all_events

def fetch_and_save_events(
    zone: str = "Montpellier",
    scope: str = "city",
    json_path: str = "data/raw/openagenda_events.json",
    csv_path: str = "data/processed/openagenda_events.csv"
) -> pd.DataFrame:
    """
    Récupère les événements OpenAgenda, puis les sauvegarde en JSON et CSV.

    Parameters
    ----------
    zone : str, default="Montpellier"
        Zone géographique ciblée.
    scope : str, default="city"
        Portée géographique utilisée pour filtrer les événements.
    json_path : str, default="data/raw/openagenda_events.json"
        Chemin du fichier JSON de sauvegarde.
    csv_path : str, default="data/processed/openagenda_events.csv"
        Chemin du fichier CSV de sauvegarde.

    Returns
    -------
    pd.DataFrame
        DataFrame normalisé des événements.
    """
    agendas = search_agendas_for_zone(zone)

    if not agendas:
        return pd.DataFrame()

    all_events = []

    for agenda in agendas:
        events = fetch_openagenda_events(
            agenda_uid=str(agenda["uid"]),
            zone=zone,
            scope=scope,
        )
        all_events.extend(events)

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
    # Colonnes attendues dans le schéma final
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
        "source_url"
    ]

    # Si aucun événement n'est présent, on retourne un DataFrame vide mais structuré
    if not events:
        return pd.DataFrame(columns=schema_cols)

    # Aplatissement du JSON en tableau
    df = pd.json_normalize(events)

    # Correspondance entre noms OpenAgenda et noms internes du projet
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
        "location.region": "region"
    }

    # On conserve uniquement les colonnes présentes dans le JSON
    existing_cols = [c for c in rename_map if c in df.columns]
    df = df[existing_cols].copy()
    df = df.rename(columns={c: rename_map[c] for c in existing_cols})

    # Ajout des colonnes manquantes pour garantir un schéma stable
    for col in schema_cols:
        if col not in df.columns:
            df[col] = ""

    # Nettoyage simple des valeurs
    for col in schema_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Nettoyage complémentaire sur les colonnes textuelles principales
    df["title"] = df["title"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["description"] = df["description"].str.replace(r"\s+", " ", regex=True).str.strip()

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

    # Construction d'un texte unique regroupant les champs clés
    df["text_for_embedding"] = (
        df["title"].fillna("") + "\n"
        + df["description"].fillna("") + "\n"
        + df["location_name"].fillna("") + "\n"
        + df["city"].fillna("") + "\n"
        + df["region"].fillna("") + "\n"
        + df["first_date"].fillna("").astype(str) + "\n"
        + df["last_date"].fillna("").astype(str) + "\n"
        + df["event_type"].fillna("")
    )

    return df

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

    Returns
    -------
    list[Document]
        Liste des documents LangChain prêts à être indexés.
    """
    if source == "api":
        df_events = fetch_and_save_events(
            zone=zone,
            scope=scope,
            json_path=json_path,
            csv_path=csv_path
        )

    elif source == "json":
        events = load_events_from_json(json_path)
        df_events = normalize_events(events)
        df_events = build_indexable_text(df_events)

    elif source == "csv":
        df_events = load_events_from_csv(csv_path)

        if "text_for_embedding" not in df_events.columns:
            df_events = build_indexable_text(df_events)

    else:
        raise ValueError("source doit valoir 'api', 'json' ou 'csv'")

    if df_events.empty:
        return []

    if "event_uid" in df_events.columns:
        df_events = df_events.drop_duplicates(
            subset=["event_uid"]
        ).reset_index(drop=True)

    documents = [
        build_event_document(row.to_dict())
        for _, row in df_events.iterrows()
    ]

    return documents