"""
Service de mémoire locale du système RAG.

Ce module implémente une mémoire conversationnelle simple persistée
dans un fichier JSON situé dans le dossier `data/memory/`.

Son rôle est de conserver une trace légère de certains échanges
entre l'utilisateur et le système.

Cette mémoire permet notamment de :

- retrouver une question déjà posée à l'identique
- réinjecter une ancienne réponse comme contexte léger
- interpréter certaines formulations de suivi comme
  "je prends le 2" ou "choix 1"

La mémoire ne remplace pas le contexte documentaire principal du
pipeline RAG. Elle agit uniquement comme un mécanisme d'assistance
conversationnelle pour rendre les échanges plus fluides.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


class MemoryService:
    """
    Service de mémoire locale persistée pour le système RAG.

    Cette classe centralise les opérations liées à la mémoire
    conversationnelle : chargement, sauvegarde, ajout d'entrées,
    recherche d'une question identique et interprétation de choix
    utilisateur à partir des documents précédemment proposés.

    La mémoire est stockée dans un fichier JSON local.

    Parameters
    ----------
    memory_file : str, default="data/memory/rag_memory.json"
        Chemin du fichier JSON utilisé pour persister la mémoire.
    max_entries : int, default=500
        Nombre maximum d'entrées conservées en mémoire.
    """

    def __init__(
        self,
        memory_file: str = "data/memory/rag_memory.json",
        max_entries: int = 500,
    ) -> None:
        self.memory_file = Path(memory_file)
        self.max_entries = max_entries

    def _normalize(self, text: str) -> str:
        """
        Normalise un texte afin de faciliter les comparaisons.

        La normalisation applique plusieurs traitements simples :
        - mise en minuscules
        - suppression des espaces superflus
        - retrait des caractères spéciaux non nécessaires

        Parameters
        ----------
        text : str
            Texte à normaliser.

        Returns
        -------
        str
            Texte normalisé, prêt à être comparé.
        """
        text = (text or "").lower().strip()
        text = re.sub(r"[^\w\sàâçéèêëîïôûùüÿñæœ-]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def load_memory(self) -> list[dict[str, Any]]:
        """
        Charge les entrées mémoire depuis le fichier JSON.

        Si le fichier n'existe pas ou si son contenu est invalide,
        une liste vide est retournée.

        Returns
        -------
        list[dict[str, Any]]
            Liste des entrées mémoire disponibles.
        """
        if not self.memory_file.exists():
            return []

        try:
            with self.memory_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            return data if isinstance(data, list) else []

        except Exception:
            return []

    def save_memory(self, entries: list[dict[str, Any]]) -> None:
        """
        Sauvegarde les entrées mémoire dans le fichier JSON.

        Le dossier parent est créé automatiquement si nécessaire.

        Parameters
        ----------
        entries : list[dict[str, Any]]
            Liste des entrées à persister.
        """
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)

        with self.memory_file.open("w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        """
        Réinitialise complètement la mémoire persistée.

        Cette méthode remplace le contenu du fichier mémoire
        par une liste vide.
        """
        self.save_memory([])

    def get_last_entry(self) -> dict[str, Any] | None:
        """
        Retourne la dernière entrée mémoire disponible.

        Returns
        -------
        dict[str, Any] | None
            Dernière entrée enregistrée, ou None si la mémoire est vide.
        """
        entries = self.load_memory()
        return entries[-1] if entries else None

    def find_exact_question(self, question: str) -> dict[str, Any] | None:
        """
        Recherche une question déjà posée à l'identique.

        La comparaison est effectuée sur une version normalisée
        du texte afin de limiter les variations de casse,
        de ponctuation et d'espacement.

        Parameters
        ----------
        question : str
            Question à rechercher dans la mémoire.

        Returns
        -------
        dict[str, Any] | None
            Entrée mémoire correspondante si elle existe, sinon None.
        """
        normalized_question = self._normalize(question)

        for entry in reversed(self.load_memory()):
            if self._normalize(entry.get("question", "")) == normalized_question:
                return entry

        return None

    def add_entry(
        self,
        question: str,
        answer: str,
        documents: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Ajoute une nouvelle entrée dans la mémoire persistée.

        Une entrée mémoire contient :
        - la question posée
        - la réponse générée
        - les documents associés
        - un horodatage simple de création

        La mémoire est tronquée automatiquement pour conserver
        uniquement les `max_entries` entrées les plus récentes.

        Parameters
        ----------
        question : str
            Question utilisateur.
        answer : str
            Réponse générée par le système.
        documents : list[dict[str, Any]] | None, default=None
            Documents associés à la réponse.

        Returns
        -------
        dict[str, Any]
            Entrée nouvellement ajoutée.
        """
        entries = self.load_memory()

        entry = {
            "question": (question or "").strip(),
            "answer": (answer or "").strip(),
            "documents": documents or [],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        entries.append(entry)
        entries = entries[-self.max_entries:]

        self.save_memory(entries)
        return entry

    def build_memory_context(
        self,
        question: str,
        max_chars: int = 600,
    ) -> str:
        """
        Construit un contexte mémoire léger à partir d'une question.

        Si une question identique a déjà été posée, un rappel est généré
        contenant la question passée et un extrait de la réponse.

        Cette mémoire n'a pas vocation à remplacer le contexte documentaire.
        Elle sert uniquement de rappel conversationnel court.

        Parameters
        ----------
        question : str
            Question actuelle de l'utilisateur.
        max_chars : int, default=600
            Nombre maximum de caractères pour l'extrait de réponse.

        Returns
        -------
        str
            Texte de contexte mémoire ou chaîne vide.
        """
        entry = self.find_exact_question(question)

        if not entry:
            return ""

        answer = entry.get("answer", "").strip()
        answer_preview = answer[:max_chars].strip()

        if len(answer) > max_chars:
            answer_preview += "..."

        lines = [
            "Souvenir pertinent",
            f"Question passée : {entry.get('question', '').strip()}",
            f"Réponse passée : {answer_preview}",
        ]

        created_at = entry.get("created_at", "").strip()
        if created_at:
            lines.insert(1, f"Date du souvenir : {created_at}")

        return "\n".join(line for line in lines if line.strip())

    def extract_choice_number(self, question: str) -> int | None:
        """
        Extrait un numéro de choix depuis une formulation utilisateur.

        Exemples pris en charge :
        - "choix 2"
        - "numéro 1"
        - "je prends le 3"
        - "je veux le 2"
        - "le 1"

        Parameters
        ----------
        question : str
            Texte utilisateur.

        Returns
        -------
        int | None
            Numéro extrait ou None si aucun choix n'est détecté.
        """
        normalized = self._normalize(question)

        patterns = [
            r"\b(?:choix|numero|numéro)\s+(\d+)\b",
            r"\b(?:je veux le|je prends le|je choisis le)\s+(\d+)\b",
            r"\ble\s+(\d+)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                return int(match.group(1))

        return None

    def build_choice_answer(self, question: str) -> dict[str, Any] | None:
        """
        Construit une réponse ciblée à partir d'un choix utilisateur.

        Cette méthode permet de récupérer un document précis parmi
        ceux proposés précédemment, en fonction du numéro sélectionné
        dans le dernier échange enregistré en mémoire.

        Parameters
        ----------
        question : str
            Question contenant un numéro de choix.

        Returns
        -------
        dict[str, Any] | None
            Réponse structurée ou None si aucun choix valide n'est détecté.
        """
        choice_number = self.extract_choice_number(question)
        if choice_number is None:
            return None

        last_entry = self.get_last_entry()
        if not last_entry:
            return None

        documents = last_entry.get("documents", [])
        if not documents:
            return None

        if choice_number < 1 or choice_number > len(documents):
            return None

        selected_doc = documents[choice_number - 1]

        title = selected_doc.get("title", "")
        location_name = selected_doc.get("location_name", "")
        city = selected_doc.get("city", "")
        first_date = selected_doc.get("first_date", "")
        last_date = selected_doc.get("last_date", "")
        event_type = selected_doc.get("event_type", "")
        url = selected_doc.get("url", "")

        date_text = first_date
        if first_date and last_date and first_date != last_date:
            date_text = f"du {first_date} au {last_date}"
        elif last_date and not first_date:
            date_text = last_date

        lines = [
            "Voici l'événement correspondant à votre choix :",
            "",
            f"Titre : {title}",
            f"Lieu : {location_name}",
            f"Ville : {city}",
            f"Date : {date_text}",
        ]

        if event_type:
            lines.append(f"Type d'événement : {event_type}")

        if url:
            lines.append(f"Lien : {url}")

        return {
            "question": (question or "").strip(),
            "answer": "\n".join(line for line in lines if line.strip()),
            "n_docs": 1,
            "documents": [selected_doc],
        }