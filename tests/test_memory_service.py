from __future__ import annotations

import json

from app.memory_service import MemoryService


def test_get_history_returns_empty_list_when_file_not_exists(tmp_path):
    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
    )

    assert service.get_history("session_1") == []


def test_read_data_invalid_json_returns_empty_dict(tmp_path):
    memory_file = tmp_path / "memory.json"
    memory_file.write_text("{invalid json", encoding="utf-8")

    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
    )

    assert service._read_data() == {}


def test_read_data_valid_json_but_not_dict_returns_empty_dict(tmp_path):
    memory_file = tmp_path / "memory.json"
    memory_file.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
    )

    assert service._read_data() == {}


def test_reset_memory_creates_empty_dict_file(tmp_path):
    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
    )

    service.reset_memory()

    assert service.memory_file.exists()
    data = json.loads(service.memory_file.read_text(encoding="utf-8"))
    assert data == {}


def test_append_message_and_get_history(tmp_path):
    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
        max_turns=3,
    )

    service.append_message("session_1", "user", "Bonjour")
    service.append_message("session_1", "assistant", "Salut")

    history = service.get_history("session_1")

    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Bonjour"}
    assert history[1] == {"role": "assistant", "content": "Salut"}


def test_append_message_ignores_invalid_session_id(tmp_path):
    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
    )

    service.append_message("", "user", "Bonjour")
    service.append_message("   ", "assistant", "Salut")

    assert service.get_history("session_1") == []


def test_append_message_ignores_invalid_role(tmp_path):
    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
    )

    service.append_message("session_1", "system", "Bonjour")
    service.append_message("session_1", "", "Salut")

    assert service.get_history("session_1") == []


def test_append_message_ignores_empty_content(tmp_path):
    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
    )

    service.append_message("session_1", "user", "")
    service.append_message("session_1", "assistant", "   ")

    assert service.get_history("session_1") == []


def test_append_message_normalizes_role_and_content(tmp_path):
    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
    )

    service.append_message("session_1", " USER ", "  Bonjour  ")

    history = service.get_history("session_1")

    assert history == [{"role": "user", "content": "Bonjour"}]


def test_histories_are_isolated_by_session(tmp_path):
    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
    )

    service.append_message("session_a", "user", "Question A")
    service.append_message("session_b", "user", "Question B")

    history_a = service.get_history("session_a")
    history_b = service.get_history("session_b")

    assert history_a == [{"role": "user", "content": "Question A"}]
    assert history_b == [{"role": "user", "content": "Question B"}]


def test_get_history_respects_max_turns(tmp_path):
    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
        max_turns=2,
    )

    service.append_message("session_1", "user", "Q1")
    service.append_message("session_1", "assistant", "R1")
    service.append_message("session_1", "user", "Q2")
    service.append_message("session_1", "assistant", "R2")
    service.append_message("session_1", "user", "Q3")
    service.append_message("session_1", "assistant", "R3")

    history = service.get_history("session_1")

    assert len(history) == 4
    assert history == [
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "R2"},
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": "R3"},
    ]


def test_get_history_filters_invalid_items_from_file(tmp_path):
    memory_file = tmp_path / "memory.json"
    payload = {
        "session_1": [
            {"role": "user", "content": "Bonjour"},
            {"role": "assistant", "content": "Salut"},
            {"role": "system", "content": "Ignore-moi"},
            {"role": "user", "content": ""},
            "not a dict",
            {"role": "", "content": "xxx"},
        ]
    }
    memory_file.write_text(json.dumps(payload), encoding="utf-8")

    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
    )

    history = service.get_history("session_1")

    assert history == [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Salut"},
    ]


def test_get_history_returns_empty_when_session_content_is_not_a_list(tmp_path):
    memory_file = tmp_path / "memory.json"
    payload = {"session_1": {"role": "user", "content": "Bonjour"}}
    memory_file.write_text(json.dumps(payload), encoding="utf-8")

    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
    )

    assert service.get_history("session_1") == []


def test_clear_session_removes_only_target_session(tmp_path):
    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
    )

    service.append_message("session_1", "user", "Bonjour")
    service.append_message("session_2", "user", "Salut")

    service.clear_session("session_1")

    assert service.get_history("session_1") == []
    assert service.get_history("session_2") == [
        {"role": "user", "content": "Salut"}
    ]


def test_clear_session_with_empty_session_id_does_nothing(tmp_path):
    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
    )

    service.append_message("session_1", "user", "Bonjour")
    service.clear_session("")

    assert service.get_history("session_1") == [
        {"role": "user", "content": "Bonjour"}
    ]


def test_format_history_for_prompt_returns_empty_string_when_no_history(tmp_path):
    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
    )

    assert service.format_history_for_prompt("session_1") == ""


def test_format_history_for_prompt_formats_messages_correctly(tmp_path):
    service = MemoryService(
        memory_dir=tmp_path,
        memory_file="memory.json",
    )

    service.append_message("session_1", "user", "Je veux une expo")
    service.append_message("session_1", "assistant", "Voici des expositions")

    formatted = service.format_history_for_prompt("session_1")

    assert formatted == (
        "Utilisateur : Je veux une expo\n"
        "Assistant : Voici des expositions"
    )