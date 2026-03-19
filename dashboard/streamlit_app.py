"""
Interface Streamlit du projet RAG OpenAgenda.

Ce module implémente le tableau de bord utilisateur du projet.
Il permet d'interagir simplement avec l'API FastAPI exposant le
pipeline RAG appliqué aux événements culturels OpenAgenda.

Fonctionnalités principales
---------------------------
- vérifier la disponibilité de l'API via l'endpoint `/health`
- poser une question en langage naturel au système RAG
- afficher la réponse générée de manière lisible
- visualiser les documents retournés par le moteur de recherche
- consulter un historique local des échanges de la session
- déclencher une reconstruction de la base documentaire via `/rebuild`

L'interface repose sur Streamlit pour la partie front-end et sur
des appels HTTP pour communiquer avec l'API.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st


# Ajout du dossier racine du projet au PYTHONPATH afin de permettre
# l'import des modules applicatifs depuis le dashboard Streamlit.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import API_KEY, API_URL, DEFAULT_SCOPE, DEFAULT_ZONE


st.set_page_config(
    page_title="Dashboard RAG OpenAgenda",
    layout="wide",
)

st.markdown(
    """
    <style>
    .box {
        color: #111111;
        background: #f8fdfc;
        border-left: 5px solid #48C9B0;
        padding: 16px 18px;
        border-radius: 10px;
        margin-bottom: 20px;
        line-height: 1.7;
    }

    .card {
        color: #111111;
        background: #ffffff;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
    }

    .history {
        color: #111111;
        background: #f8f9fa;
        border-left: 4px solid #48C9B0;
        padding: 12px 14px;
        border-radius: 8px;
        margin-bottom: 12px;
        line-height: 1.6;
    }

    .card h4 {
        color: #48C9B0;
        margin: 0 0 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_headers() -> dict[str, str]:
    """
    Construit les en-têtes HTTP utilisés pour les appels à l'API.

    Returns
    -------
    dict[str, str]
        Dictionnaire contenant les en-têtes HTTP à envoyer.
    """
    headers = {}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    return headers


def get_error_message(response: requests.Response) -> str:
    """
    Extrait un message d'erreur lisible depuis une réponse HTTP.

    Parameters
    ----------
    response : requests.Response
        Réponse HTTP renvoyée par l'API.

    Returns
    -------
    str
        Message d'erreur lisible pour l'utilisateur.
    """
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload.get("detail", str(payload))
    except Exception:
        pass

    return response.text or f"Erreur HTTP {response.status_code}"


def call_health() -> requests.Response:
    """
    Interroge l'endpoint `/health` de l'API.

    Returns
    -------
    requests.Response
        Réponse HTTP renvoyée par l'API.
    """
    return requests.get(
        f"{API_URL}/health",
        timeout=30,
    )


def call_rebuild(zone: str, scope: str) -> requests.Response:
    """
    Déclenche la reconstruction de la base documentaire.

    Parameters
    ----------
    zone : str
        Zone géographique ciblée.
    scope : str
        Niveau géographique utilisé pour la recherche.

    Returns
    -------
    requests.Response
        Réponse HTTP renvoyée par l'API.
    """
    return requests.post(
        f"{API_URL}/rebuild",
        json={"zone": zone, "scope": scope},
        headers=get_headers(),
        timeout=180,
    )


def call_ask(question: str) -> requests.Response:
    """
    Envoie une question utilisateur à l'endpoint `/ask`.

    Parameters
    ----------
    question : str
        Question formulée en langage naturel par l'utilisateur.

    Returns
    -------
    requests.Response
        Réponse HTTP renvoyée par l'API.
    """
    return requests.post(
        f"{API_URL}/ask",
        json={"question": question},
        headers=get_headers(),
        timeout=180,
    )


def format_date_range(first_date: str, last_date: str) -> str:
    """
    Formate une plage de dates pour l'affichage.

    Parameters
    ----------
    first_date : str
        Date de début de l'événement.
    last_date : str
        Date de fin de l'événement.

    Returns
    -------
    str
        Texte lisible représentant la date ou la période.
    """
    if first_date and last_date and first_date != last_date:
        return f"Du {first_date} au {last_date}"
    if first_date:
        return first_date
    if last_date:
        return last_date
    return "Non précisée"


def render_answer(answer: str) -> None:
    """
    Affiche la réponse générée par le système RAG.

    Parameters
    ----------
    answer : str
        Réponse textuelle produite par le système.
    """
    st.markdown("### Recommandation")
    formatted_answer = answer.replace("\n", "<br>")

    st.markdown(
        f'<div class="box">{formatted_answer}</div>',
        unsafe_allow_html=True,
    )


def render_documents(documents: list[dict]) -> None:
    """
    Affiche les documents récupérés sous forme de cartes.

    Parameters
    ----------
    documents : list[dict]
        Liste des documents retournés par l'API.
    """
    if not documents:
        return

    st.markdown("### Événements proposés")

    for i, doc in enumerate(documents, start=1):
        title = doc.get("title", "Événement sans titre")
        location_name = doc.get("location_name", "")
        city = doc.get("city", "")
        region = doc.get("region", "")
        first_date = doc.get("first_date", "")
        last_date = doc.get("last_date", "")
        event_type = doc.get("event_type", "")
        url = doc.get("url", "")

        date_text = format_date_range(first_date, last_date)

        st.markdown(
            f"""
            <div class="card">
                <h4>{i}. {title}</h4>
                <p><strong>Lieu :</strong> {location_name or "Non précisé"}</p>
                <p><strong>Ville :</strong> {city or "Non précisée"}</p>
                <p><strong>Région :</strong> {region or "Non précisée"}</p>
                <p><strong>Date :</strong> {date_text}</p>
                <p><strong>Type :</strong> {event_type or "Non précisé"}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if url:
            st.link_button(f"Voir la fiche de l’événement {i}", url)


def render_history_item(item: dict, index: int) -> None:
    """
    Affiche un élément de l'historique local.

    Parameters
    ----------
    item : dict
        Dictionnaire représentant un échange précédent.
    index : int
        Position de l'élément dans l'affichage.
    """
    formatted_history_answer = item["answer"].replace("\n", "<br>")

    with st.expander(f"{index}. {item['question']}"):
        st.markdown("**Réponse :**")
        st.markdown(
            f'<div class="history">{formatted_history_answer}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(f"**Documents utilisés :** {item['n_docs']}")

        docs = item.get("documents", [])
        if docs:
            st.markdown("**Sources récupérées :**")
            for doc in docs:
                st.markdown(
                    f"- **{doc.get('title', '')}** "
                    f"({doc.get('city', '')}, {doc.get('first_date', '')})"
                )


def render_metrics(answer: str, n_docs: int) -> None:
    """
    Affiche quelques indicateurs simples sur la réponse produite.

    Parameters
    ----------
    answer : str
        Réponse produite par le système.
    n_docs : int
        Nombre de documents utilisés lors de la génération.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Documents utilisés", n_docs)

    with col2:
        st.metric("Longueur de la réponse", len(answer))


if "history" not in st.session_state:
    st.session_state.history = []

if "health_data" not in st.session_state:
    st.session_state.health_data = None


if not API_KEY:
    st.warning("API_KEY manquante")


st.title("Assistant culturel OpenAgenda")
st.caption("Trouvez des événements culturels via langage naturel")


st.sidebar.header("Contrôles API")
st.sidebar.write(f"**URL API :** `{API_URL}`")

if st.sidebar.button("Vérifier /health", use_container_width=True):
    try:
        response = call_health()

        if response.ok:
            st.session_state.health_data = response.json()
            st.sidebar.success("API OK")
        else:
            st.sidebar.error(get_error_message(response))

    except Exception as exc:
        st.sidebar.error(str(exc))


tab_chat, tab_admin = st.tabs(["Chat", "Admin"])


with tab_chat:
    st.subheader("Question")

    question = st.text_area(
        "Votre question",
        value="Je cherche une exposition à Montpellier",
        height=120,
    )

    col_action, col_clear = st.columns([3, 1])

    with col_action:
        search_clicked = st.button(
            "Rechercher",
            type="primary",
            use_container_width=True,
        )

    with col_clear:
        if st.button("Vider l'historique", use_container_width=True):
            st.session_state.history = []
            st.success("Historique vidé.")

    if search_clicked:
        if question.strip():
            try:
                with st.spinner("Recherche..."):
                    response = call_ask(question)

                if response.ok:
                    data = response.json()
                    st.session_state.history.insert(0, data)

                    render_answer(data["answer"])
                    render_metrics(data["answer"], data["n_docs"])
                    render_documents(data["documents"])

                else:
                    st.error(get_error_message(response))

            except Exception as exc:
                st.error(str(exc))
        else:
            st.warning("Veuillez saisir une question.")

    st.divider()
    st.subheader("Historique")

    for i, item in enumerate(st.session_state.history[:5], 1):
        render_history_item(item, i)


with tab_admin:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reconstruction documentaire")

        zone = st.text_input("Zone", DEFAULT_ZONE)
        allowed_scopes = ["city", "region", "country"]
        scope = st.selectbox(
            "Scope",
            allowed_scopes,
            index=allowed_scopes.index(DEFAULT_SCOPE)
            if DEFAULT_SCOPE in allowed_scopes
            else 0,
        )

        if st.button("Rebuild", use_container_width=True):
            try:
                with st.spinner("Rebuild..."):
                    response = call_rebuild(zone, scope)

                if response.ok:
                    data = response.json()
                    st.success(data.get("message", "Reconstruction terminée."))
                    st.info(f"Documents indexés : {data.get('n_docs_indexed', 0)}")
                else:
                    st.error(get_error_message(response))

            except Exception as exc:
                st.error(str(exc))

    with col2:
        st.subheader("État de l'API")

        if st.button("Refresh health", use_container_width=True):
            try:
                response = call_health()

                if response.ok:
                    st.session_state.health_data = response.json()
                    st.success("API OK")
                else:
                    st.error(get_error_message(response))

            except Exception as exc:
                st.error(str(exc))

        if st.session_state.health_data:
            st.json(st.session_state.health_data)
        else:
            st.caption("Aucun état API récupéré pour le moment.")