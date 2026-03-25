from pathlib import Path

from app.memory_service import MemoryService


def test_load_memory_file_not_exists(tmp_path: Path):
    memory_file = tmp_path / "missing_memory.json"
    memory = MemoryService(memory_file=str(memory_file))

    result = memory.load_memory()

    assert result == []


def test_load_memory_invalid_json_returns_empty_list(tmp_path: Path):
    memory_file = tmp_path / "memory.json"
    memory_file.write_text("{invalid json", encoding="utf-8")

    memory = MemoryService(memory_file=str(memory_file))

    result = memory.load_memory()

    assert result == []


def test_load_memory_valid_json_but_not_list_returns_empty_list(tmp_path: Path):
    memory_file = tmp_path / "memory.json"
    memory_file.write_text('{"question": "Q"}', encoding="utf-8")

    memory = MemoryService(memory_file=str(memory_file))

    result = memory.load_memory()

    assert result == []


def test_save_and_load_memory(tmp_path: Path):
    memory_file = tmp_path / "memory.json"
    memory = MemoryService(memory_file=str(memory_file))

    entries = [
        {
            "question": "Question 1",
            "answer": "Réponse 1",
            "documents": [],
        }
    ]

    memory.save_memory(entries)
    loaded = memory.load_memory()

    assert memory_file.exists()
    assert loaded == entries


def test_save_memory_creates_parent_directory(tmp_path: Path):
    memory_file = tmp_path / "nested" / "memory" / "memory.json"
    memory = MemoryService(memory_file=str(memory_file))

    memory.save_memory(
        [
            {
                "question": "Q1",
                "answer": "R1",
                "documents": [],
            }
        ]
    )

    assert memory_file.exists()
    assert memory_file.parent.exists()


def test_clear_memory(tmp_path: Path):
    memory_file = tmp_path / "memory.json"
    memory = MemoryService(memory_file=str(memory_file))

    memory.save_memory(
        [{"question": "Q", "answer": "R", "documents": []}]
    )
    memory.clear()

    assert memory.load_memory() == []


def test_add_entry(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    entry = memory.add_entry(
        question="  Ma question  ",
        answer="  Ma réponse  ",
        documents=[{"title": "Doc 1"}],
    )

    assert entry["question"] == "Ma question"
    assert entry["answer"] == "Ma réponse"
    assert entry["documents"] == [{"title": "Doc 1"}]
    assert "created_at" in entry
    assert isinstance(entry["created_at"], str)
    assert entry["created_at"]

    loaded = memory.load_memory()
    assert len(loaded) == 1
    assert loaded[0]["question"] == "Ma question"
    assert loaded[0]["answer"] == "Ma réponse"
    assert loaded[0]["documents"] == [{"title": "Doc 1"}]
    assert "created_at" in loaded[0]


def test_add_entry_with_none_values(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    entry = memory.add_entry(
        question=None,
        answer=None,
        documents=None,
    )

    assert entry["question"] == ""
    assert entry["answer"] == ""
    assert entry["documents"] == []

    loaded = memory.load_memory()
    assert len(loaded) == 1
    assert loaded[0]["question"] == ""
    assert loaded[0]["answer"] == ""
    assert loaded[0]["documents"] == []


def test_add_entry_respects_max_entries(tmp_path: Path):
    memory = MemoryService(
        memory_file=str(tmp_path / "memory.json"),
        max_entries=2,
    )

    memory.add_entry("Q1", "R1")
    memory.add_entry("Q2", "R2")
    memory.add_entry("Q3", "R3")

    loaded = memory.load_memory()

    assert len(loaded) == 2
    assert loaded[0]["question"] == "Q2"
    assert loaded[1]["question"] == "Q3"


def test_get_recent_entries_empty(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    result = memory.get_recent_entries()

    assert result == []


def test_get_recent_entries_default_limit(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    memory.add_entry("Q1", "R1")
    memory.add_entry("Q2", "R2")
    memory.add_entry("Q3", "R3")
    memory.add_entry("Q4", "R4")
    memory.add_entry("Q5", "R5")
    memory.add_entry("Q6", "R6")

    result = memory.get_recent_entries()

    assert len(result) == 5
    assert result[0]["question"] == "Q2"
    assert result[-1]["question"] == "Q6"


def test_get_recent_entries_custom_limit(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    memory.add_entry("Q1", "R1")
    memory.add_entry("Q2", "R2")
    memory.add_entry("Q3", "R3")

    result = memory.get_recent_entries(limit=2)

    assert len(result) == 2
    assert result[0]["question"] == "Q2"
    assert result[1]["question"] == "Q3"


def test_get_recent_entries_limit_greater_than_entries(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    memory.add_entry("Q1", "R1")
    memory.add_entry("Q2", "R2")

    result = memory.get_recent_entries(limit=10)

    assert len(result) == 2
    assert result[0]["question"] == "Q1"
    assert result[1]["question"] == "Q2"


def test_get_recent_entries_limit_zero_returns_at_least_one_when_entries_exist(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    memory.add_entry("Q1", "R1")
    memory.add_entry("Q2", "R2")

    result = memory.get_recent_entries(limit=0)

    assert len(result) == 1
    assert result[0]["question"] == "Q2"


def test_get_recent_entries_negative_limit_returns_at_least_one_when_entries_exist(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    memory.add_entry("Q1", "R1")
    memory.add_entry("Q2", "R2")
    memory.add_entry("Q3", "R3")

    result = memory.get_recent_entries(limit=-5)

    assert len(result) == 1
    assert result[0]["question"] == "Q3"