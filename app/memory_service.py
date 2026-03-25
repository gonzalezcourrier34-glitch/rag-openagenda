"""
Service de mémoire locale minimal pour le système RAG.

Ce module permet uniquement de :
- charger la mémoire depuis un fichier JSON
- sauvegarder la mémoire
- ajouter une entrée
- lire les dernières entrées
- vider la mémoire

Il ne contient volontairement aucune logique de ranking,
d'interprétation conversationnelle ou de génération de réponse.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_FILE = BASE_DIR / "data" / "memory" / "rag_memory.json"


class MemoryService:
    """
    Service minimal de mémoire locale persistée.

    Parameters
    ----------
    memory_file : str | Path, default=DEFAULT_MEMORY_FILE
        Chemin du fichier JSON utilisé pour la mémoire.
    max_entries : int, default=100
        Nombre maximum d'entrées conservées.
    """

    def __init__(
        self,
        memory_file: str | Path = DEFAULT_MEMORY_FILE,
        max_entries: int = 100,
    ) -> None:
        self.memory_file = Path(memory_file)
        self.max_entries = max_entries

    def load_memory(self) -> list[dict[str, Any]]:
        """
        Charge les entrées mémoire depuis le fichier JSON.

        Returns
        -------
        list[dict[str, Any]]
            Liste des entrées mémoire.
        """
        if not self.memory_file.exists():
            return []

        try:
            with self.memory_file.open("r", encoding="utf-8") as file:
                data = json.load(file)

            return data if isinstance(data, list) else []

        except (json.JSONDecodeError, OSError):
            return []

    def save_memory(self, entries: list[dict[str, Any]]) -> None:
        """
        Sauvegarde les entrées mémoire dans le fichier JSON.

        Parameters
        ----------
        entries : list[dict[str, Any]]
            Entrées à sauvegarder.
        """
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)

        with self.memory_file.open("w", encoding="utf-8") as file:
            json.dump(entries, file, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        """
        Vide complètement la mémoire.
        """
        self.save_memory([])

    def add_entry(
        self,
        question: str,
        answer: str,
        documents: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Ajoute une entrée mémoire.

        Parameters
        ----------
        question : str
            Question utilisateur.
        answer : str
            Réponse générée.
        documents : list[dict[str, Any]] | None, default=None
            Documents associés à la réponse.

        Returns
        -------
        dict[str, Any]
            Entrée ajoutée.
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

    def get_recent_entries(self, limit: int = 5) -> list[dict[str, Any]]:
        """
        Retourne les dernières entrées mémoire.

        Parameters
        ----------
        limit : int, default=5
            Nombre maximum d'entrées à retourner.

        Returns
        -------
        list[dict[str, Any]]
            Dernières entrées mémoire.
        """
        entries = self.load_memory()
        return entries[-max(1, limit):]