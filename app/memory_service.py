from __future__ import annotations

"""
Service de mémoire locale du système RAG.

Ce module gère une mémoire simple persistée dans un fichier JSON
afin de conserver certains échanges précédents entre l'utilisateur
et le système.

Cette mémoire permet notamment :

- de retrouver une question déjà posée à l'identique
- de réinjecter une ancienne réponse comme contexte léger
- d'interpréter des formulations de suivi comme
  "je prends le 2" ou "choix 1"

La mémoire ne remplace jamais le contexte documentaire principal.
Elle sert uniquement d'aide conversationnelle pour rendre
le système plus fluide d'un échange à l'autre.
"""

import json
import re
from pathlib import Path
from typing import Any


class MemoryService:
    """
    Service de mémoire locale persistée pour le système RAG.
    """

    def __init__(
        self,
        memory_file: str = "rag_memory.json",
        max_entries: int = 500,
    ) -> None:
        self.memory_file = Path(memory_file)
        self.max_entries = max_entries

    def _normalize(self, text: str) -> str:
        """
        Normalise un texte pour faciliter les comparaisons.
        """
        text = (text or "").lower().strip()
        text = re.sub(r"[^\w\sàâçéèêëîïôûùüÿñæœ-]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def load_memory(self) -> list[dict[str, Any]]:
        """
        Charge les entrées mémoire depuis le fichier JSON.
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
        """
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)

        with self.memory_file.open("w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        """
        Vide entièrement la mémoire persistée.
        """
        self.save_memory([])

    def get_last_entry(self) -> dict[str, Any] | None:
        """
        Retourne la dernière entrée mémoire disponible.
        """
        entries = self.load_memory()
        return entries[-1] if entries else None

    def find_exact_question(self, question: str) -> dict[str, Any] | None:
        """
        Recherche une question déjà posée à l'identique.
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
        Ajoute une nouvelle entrée en mémoire.
        """
        entries = self.load_memory()

        entry = {
            "question": question.strip(),
            "answer": answer.strip(),
            "documents": documents or [],
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
        Construit un petit contexte mémoire à partir d'une question.
        """
        entry = self.find_exact_question(question)

        if not entry:
            return ""

        answer_preview = entry.get("answer", "")[:max_chars]

        return "\n".join(
            [
                "Souvenir",
                f"Question passée : {entry.get('question', '')}",
                f"Réponse passée : {answer_preview}",
            ]
        )

    def extract_choice_number(self, question: str) -> int | None:
        """
        Extrait un numéro de choix depuis une formulation utilisateur.
        """
        normalized = self._normalize(question)

        match = re.search(r"\b(?:choix|num[eé]ro)\s+(\d+)\b", normalized)
        if match:
            return int(match.group(1))

        match = re.search(r"\b(?:je veux le|je prends le|le)\s+(\d+)\b", normalized)
        if match:
            return int(match.group(1))

        return None

    def build_choice_answer(self, question: str) -> dict[str, Any] | None:
        """
        Construit une réponse ciblée à partir d'un choix utilisateur.
        """
        choice_number = self.extract_choice_number(question)
        if choice_number is None:
            return None

        last_entry = self.get_last_entry()
        if not last_entry:
            return None

        documents = last_entry.get("documents", [])
        if not documents or choice_number < 1 or choice_number > len(documents):
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
            "question": question.strip(),
            "answer": "\n".join(line for line in lines if line.strip()),
            "n_docs": 1,
            "documents": [selected_doc],
        }