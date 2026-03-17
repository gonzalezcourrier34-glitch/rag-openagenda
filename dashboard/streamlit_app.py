"""
Interface Streamlit du projet RAG OpenAgenda.

Ce module fournit un tableau de bord permettant de :
- vérifier l'état de l'API FastAPI
- poser une question au système RAG
- afficher une recommandation plus lisible
- consulter un historique local des échanges
- reconstruire la base documentaire via l'endpoint `/rebuild`

L'application communique avec l'API par requêtes HTTP.
"""

import requests
import streamlit as st

from app.config import API_KEY, API_URL, DEFAULT_SCOPE, DEFAULT_ZONE


st.set_page_config(
    page_title="Dashboard RAG OpenAgenda",
    layout="wide",
)


def get_headers() -> dict[str, str]:
    """
    Construit les en-têtes HTTP pour les appels vers l'API.

    Returns
    -------
    dict[str, str]
        Dictionnaire des headers HTTP.
    """
    headers = {}

    if API_KEY:
        headers["x-api-key"] = API_KEY

    return headers


def call_health() -> requests.Response:
    """
    Appelle l'endpoint `/health` de l'API.

    Returns
    -------
    requests.Response
        Réponse HTTP retournée par l'API.
    """
    return requests.get(
        f"{API_URL}/health",
        headers=get_headers(),
        timeout=30,
    )


def call_rebuild(zone: str, scope: str) -> requests.Response:
    """
    Appelle l'endpoint `/rebuild` pour reconstruire l'index documentaire.

    Parameters
    ----------
    zone : str
        Zone géographique utilisée pour récupérer les événements.
    scope : str
        Type de zone utilisé dans la recherche.

    Returns
    -------
    requests.Response
        Réponse HTTP retournée par l'API.
    """
    return requests.post(
        f"{API_URL}/rebuild",
        json={
            "zone": zone,
            "scope": scope,
        },
        headers=get_headers(),
        timeout=180,
    )


def call_ask(question: str) -> requests.Response:
    """
    Appelle l'endpoint `/ask` avec une question utilisateur.

    Parameters
    ----------
    question : str
        Question posée au système RAG.

    Returns
    -------
    requests.Response
        Réponse HTTP retournée par l'API.
    """
    return requests.post(
        f"{API_URL}/ask",
        json={"question": question},
        headers=get_headers(),
        timeout=180,
    )


def format_date_range(first_date: str, last_date: str) -> str:
    """
    Construit un texte lisible pour la plage de dates.

    Parameters
    ----------
    first_date : str
        Date de début.
    last_date : str
        Date de fin.

    Returns
    -------
    str
        Texte formaté de la date.
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
    Affiche la réponse du chatbot dans un bloc visuel.

    Parameters
    ----------
    answer : str
        Réponse générée par le système RAG.
    """
    st.markdown("### Recommandation")
    formatted_answer = answer.replace("\n", "<br>")

    st.markdown(
        f"""
<div style="
    background-color:#f8fdfc;
    border-left:5px solid #48C9B0;
    padding:16px 18px;
    border-radius:10px;
    margin-bottom:20px;
    color:#111111;
    line-height:1.7;
">
    {formatted_answer}
</div>
""",
        unsafe_allow_html=True,
    )

def render_documents(documents: list[dict]) -> None:
    """
    Affiche les documents récupérés sous forme de cartes lisibles.

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
<div style="
    border:1px solid #E5E7EB;
    border-radius:12px;
    padding:16px;
    margin-bottom:14px;
    background-color:#ffffff;
    box-shadow:0 1px 4px rgba(0, 0, 0, 0.05);
">
    <h4 style="margin:0 0 10px 0; color:#48C9B0;">{i}. {title}</h4>
    <p style="margin:4px 0;"><strong>Lieu :</strong> {location_name or "Non précisé"}</p>
    <p style="margin:4px 0;"><strong>Ville :</strong> {city or "Non précisée"}</p>
    <p style="margin:4px 0;"><strong>Région :</strong> {region or "Non précisée"}</p>
    <p style="margin:4px 0;"><strong>Date :</strong> {date_text}</p>
    <p style="margin:4px 0;"><strong>Type :</strong> {event_type or "Non précisé"}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        if url:
            st.link_button(f"Voir la fiche de l’événement {i}", url)

def render_history_item(item: dict, index: int) -> None:
    """
    Affiche une entrée d'historique.

    Parameters
    ----------
    item : dict
        Élément de l'historique.
    index : int
        Position d'affichage.
    """
    formatted_history_answer = item["answer"].replace("\n", "<br>")

    with st.expander(f"{index}. {item['question']}"):
        st.markdown("**Réponse :**")
        st.markdown(
            f"""
<div style="
    background-color:#f8f9fa;
    border-left:4px solid #48C9B0;
    padding:12px 14px;
    border-radius:8px;
    margin-bottom:12px;
    color:#111111;
    line-height:1.6;
">
    {formatted_history_answer}
</div>
""",
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
    Affiche quelques indicateurs simples.

    Parameters
    ----------
    answer : str
        Réponse générée.
    n_docs : int
        Nombre de documents utilisés.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Documents utilisés", n_docs)

    with col2:
        st.metric("Longueur de la réponse", len(answer))


# Initialisation de l'état de session
if "history" not in st.session_state:
    st.session_state.history = []

if "health_data" not in st.session_state:
    st.session_state.health_data = None


if not API_KEY:
    st.warning(
        "La variable d'environnement API_KEY est absente. "
        "Les appels sécurisés à l'API risquent d'échouer."
    )


st.title("Assistant culturel OpenAgenda")
st.caption("Trouvez des événements culturels à partir d’une question en langage naturel.")


# Sidebar
st.sidebar.header("Contrôles API")
st.sidebar.write(f"**URL API :** `{API_URL}`")

if st.sidebar.button("Vérifier /health", use_container_width=True):
    try:
        response = call_health()

        if response.ok:
            st.session_state.health_data = response.json()
            st.sidebar.success("API disponible")
        else:
            st.sidebar.error(f"Erreur {response.status_code}")
            st.sidebar.text(response.text)

    except Exception as exc:
        st.sidebar.error(f"Impossible de joindre l'API : {exc}")

st.sidebar.divider()
st.sidebar.subheader("Exemples de questions")
st.sidebar.markdown(
    """
- Je cherche une exposition d’architecture à Montpellier
- Y a-t-il une activité culturelle pour enfants ?
- Je veux une visite autour du patrimoine à Montpellier
- Que faire ce mois-ci autour de l’architecture ?
"""
)


tab_chat, tab_admin = st.tabs(["Chat", "Administration"])


with tab_chat:
    st.subheader("Poser une question")

    question = st.text_area(
        "Votre question",
        value="Je cherche une exposition d'architecture à Montpellier",
        height=120,
        placeholder="Exemple : Je cherche une activité culturelle pour enfants à Montpellier",
    )

    if st.button("Rechercher", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Merci de saisir une question.")
        else:
            try:
                with st.spinner("Recherche des événements et génération de la réponse..."):
                    response = call_ask(question)

                if response.ok:
                    data = response.json()

                    history_item = {
                        "question": data.get("question", question),
                        "answer": data.get("answer", ""),
                        "n_docs": data.get("n_docs", 0),
                        "documents": data.get("documents", []),
                    }

                    st.session_state.history.insert(0, history_item)

                    answer = data.get("answer", "")
                    n_docs = data.get("n_docs", 0)
                    documents = data.get("documents", [])

                    render_answer(answer)
                    render_metrics(answer=answer, n_docs=n_docs)
                    render_documents(documents)

                else:
                    st.error(f"Erreur API : {response.status_code}")
                    st.text(response.text)

            except Exception as exc:
                st.error(f"Erreur lors de l'appel à /ask : {exc}")

    st.divider()
    st.subheader("Historique local")

    if not st.session_state.history:
        st.info("Aucune question posée pour le moment.")
    else:
        for i, item in enumerate(st.session_state.history[:5], start=1):
            render_history_item(item, i)


with tab_admin:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Paramètres de rebuild")

        zone_input = st.text_input("Zone géographique", value=DEFAULT_ZONE)

        scope_options = ["city", "region", "country"]
        default_scope_index = (
            scope_options.index(DEFAULT_SCOPE)
            if DEFAULT_SCOPE in scope_options
            else 0
        )

        scope_input = st.selectbox(
            "Type de zone",
            options=scope_options,
            index=default_scope_index,
        )

        st.caption(f"Zone active : {zone_input} | Scope actif : {scope_input}")

        if st.button("Rebuild /rebuild", use_container_width=True):
            try:
                with st.spinner("Reconstruction de la base vectorielle..."):
                    response = call_rebuild(zone_input, scope_input)

                if response.ok:
                    st.success("Rebuild terminé")
                    st.json(response.json())
                else:
                    st.error(f"Erreur {response.status_code}")
                    st.text(response.text)

            except Exception as exc:
                st.error(f"Erreur rebuild : {exc}")

    with col2:
        st.subheader("État rapide")

        if st.button("Actualiser l’état", use_container_width=True):
            try:
                response = call_health()

                if response.ok:
                    st.session_state.health_data = response.json()
                    st.success("API en ligne")
                else:
                    st.error(f"Erreur {response.status_code}")

            except Exception as exc:
                st.error(str(exc))

        if st.session_state.health_data:
            st.json(st.session_state.health_data)
        else:
            st.info("Clique sur 'Actualiser l’état' ou 'Vérifier /health'.")

    st.divider()

    st.markdown("### Endpoints")
    st.code("/health\n/rebuild\n/ask")

    st.markdown("### Architecture")
    st.markdown(
        """
- **Streamlit** : interface utilisateur
- **FastAPI** : API REST
- **FAISS** : recherche vectorielle
- **Mistral** : génération de réponse
"""
    )

    st.markdown("### Objectif")
    st.write(
        "Permettre à un utilisateur métier de poser une question en langage naturel "
        "et d’obtenir une recommandation d’événements basée sur le système RAG."
    )