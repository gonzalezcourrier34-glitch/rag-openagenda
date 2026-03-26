from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MemoryService:
    """
    Service de gestion de la mémoire conversationnelle courte.

    Cette mémoire est :
    - persistée dans un fichier JSON local
    - vidée au démarrage de l'API
    - limitée à une fenêtre courte par session
    - utilisée uniquement pour maintenir la cohérence
      d'une conversation en cours

    Le service ne sert pas de base de connaissance.
    Il conserve seulement les derniers échanges utiles
    à l'interprétation des questions utilisateur.
    """

    def __init__(
        self,
        memory_dir: str | Path = "data/memory",
        memory_file: str = "conversations.json",
        max_turns: int = 3,
    ) -> None:
        """
        Initialise le service mémoire.

        Parameters
        ----------
        memory_dir : str | Path, default="data/memory"
            Dossier de stockage local.
        memory_file : str, default="conversations.json"
            Nom du fichier JSON de mémoire.
        max_turns : int, default=3
            Nombre maximal de tours conservés par session.
            Un tour = 1 message utilisateur + 1 message assistant.
        """
        self.memory_dir = Path(memory_dir)
        self.memory_file = self.memory_dir / memory_file
        self.max_turns = max_turns
        self.max_messages = max_turns * 2

        self.memory_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Gestion fichier
    # ------------------------------------------------------------------

    def reset_memory(self) -> None:
        """
        Réinitialise complètement la mémoire au démarrage de l'API.
        """
        self._write_data({})

    def _read_data(self) -> dict[str, list[dict[str, str]]]:
        """
        Lit le contenu mémoire depuis le fichier JSON.
        """
        if not self.memory_file.exists():
            return {}

        try:
            with self.memory_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

        if not isinstance(data, dict):
            return {}

        return data

    def _write_data(self, data: dict[str, Any]) -> None:
        """
        Écrit le contenu mémoire dans le fichier JSON.
        """
        with self.memory_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Gestion session
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """
        Retourne l'historique court d'une session.
        """
        if not session_id or not session_id.strip():
            return []

        data = self._read_data()
        history = data.get(session_id, [])

        if not isinstance(history, list):
            return []

        clean_history: list[dict[str, str]] = []
        for item in history:
            if not isinstance(item, dict):
                continue

            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()

            if role in {"user", "assistant"} and content:
                clean_history.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

        return clean_history[-self.max_messages :]

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """
        Ajoute un message à la mémoire courte d'une session.
        """
        session_id = (session_id or "").strip()
        role = (role or "").strip().lower()
        content = (content or "").strip()

        if not session_id or role not in {"user", "assistant"} or not content:
            return

        data = self._read_data()

        if session_id not in data or not isinstance(data[session_id], list):
            data[session_id] = []

        data[session_id].append(
            {
                "role": role,
                "content": content,
            }
        )

        data[session_id] = data[session_id][-self.max_messages :]
        self._write_data(data)

    def clear_session(self, session_id: str) -> None:
        """
        Supprime l'historique d'une session donnée.
        """
        session_id = (session_id or "").strip()
        if not session_id:
            return

        data = self._read_data()
        data.pop(session_id, None)
        self._write_data(data)

    def format_history_for_prompt(self, session_id: str) -> str:
        """
        Formate l'historique d'une session pour l'injection dans un prompt.
        """
        history = self.get_history(session_id)
        if not history:
            return ""

        lines: list[str] = []
        for msg in history:
            speaker = "Utilisateur" if msg["role"] == "user" else "Assistant"
            lines.append(f"{speaker} : {msg['content']}")

        return "\n".join(lines)