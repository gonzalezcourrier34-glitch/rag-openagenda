from pathlib import Path

from app.memory_service import MemoryService


def test_normalize_basic():
    memory = MemoryService()

    result = memory._normalize("  Bonjour,   Montpellier !!! ")

    assert result == "bonjour montpellier"


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


def test_clear_memory(tmp_path: Path):
    memory_file = tmp_path / "memory.json"
    memory = MemoryService(memory_file=str(memory_file))

    memory.save_memory(
        [{"question": "Q", "answer": "R", "documents": []}]
    )
    memory.clear()

    assert memory.load_memory() == []


def test_get_last_entry_empty(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    assert memory.get_last_entry() is None


def test_get_last_entry_returns_last(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    memory.save_memory(
        [
            {"question": "Q1", "answer": "R1", "documents": []},
            {"question": "Q2", "answer": "R2", "documents": []},
        ]
    )

    last = memory.get_last_entry()

    assert last is not None
    assert last["question"] == "Q2"
    assert last["answer"] == "R2"


def test_find_exact_question_found_despite_case_and_punctuation(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    memory.save_memory(
        [
            {
                "question": "Je cherche une exposition à Montpellier !",
                "answer": "Réponse",
                "documents": [],
            }
        ]
    )

    result = memory.find_exact_question("je cherche une exposition à montpellier")

    assert result is not None
    assert result["answer"] == "Réponse"


def test_find_exact_question_not_found(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    memory.save_memory(
        [
            {
                "question": "Question A",
                "answer": "Réponse A",
                "documents": [],
            }
        ]
    )

    result = memory.find_exact_question("Question B")

    assert result is None


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
    assert "created_at" in loaded[0]


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


def test_build_memory_context_found(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    memory.add_entry(
        question="Je cherche une exposition",
        answer="Voici une réponse assez longue",
    )

    context = memory.build_memory_context("je cherche une exposition")

    assert "Souvenir pertinent" in context
    assert "Question passée : Je cherche une exposition" in context
    assert "Réponse passée : Voici une réponse assez longue" in context
    assert "Date du souvenir :" in context


def test_build_memory_context_not_found(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    memory.add_entry(
        question="Question A",
        answer="Réponse A",
    )

    context = memory.build_memory_context("Question B")

    assert context == ""


def test_build_memory_context_max_chars(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    memory.add_entry(
        question="Question test",
        answer="abcdefghijklmnopqrstuvwxyz",
    )

    context = memory.build_memory_context("Question test", max_chars=5)

    assert "Réponse passée : abcde..." in context


def test_extract_choice_number_choix():
    memory = MemoryService()

    assert memory.extract_choice_number("choix 2") == 2


def test_extract_choice_number_numero():
    memory = MemoryService()

    assert memory.extract_choice_number("numéro 4") == 4


def test_extract_choice_number_je_prends():
    memory = MemoryService()

    assert memory.extract_choice_number("je prends le 3") == 3


def test_extract_choice_number_je_veux():
    memory = MemoryService()

    assert memory.extract_choice_number("je veux le 1") == 1


def test_extract_choice_number_le():
    memory = MemoryService()

    assert memory.extract_choice_number("le 1") == 1


def test_extract_choice_number_does_not_confuse_date():
    memory = MemoryService()

    assert memory.extract_choice_number("le 3 mars") is None


def test_extract_choice_number_not_found():
    memory = MemoryService()

    assert memory.extract_choice_number("je veux une exposition") is None


def test_build_choice_answer_no_choice_number(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    result = memory.build_choice_answer("je veux une exposition")

    assert result is None


def test_build_choice_answer_no_last_entry(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    result = memory.build_choice_answer("choix 1")

    assert result is None


def test_build_choice_answer_no_documents_in_last_entry(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    memory.add_entry(
        question="Question",
        answer="Réponse",
        documents=[],
    )

    result = memory.build_choice_answer("choix 1")

    assert result is None


def test_build_choice_answer_invalid_choice_out_of_range(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    memory.add_entry(
        question="Question",
        answer="Réponse",
        documents=[
            {
                "title": "Doc 1",
                "location_name": "Lieu 1",
                "city": "Montpellier",
                "first_date": "2026-03-01",
                "last_date": "2026-03-01",
                "event_type": "Exposition",
                "url": "http://doc1.com",
            }
        ],
    )

    result = memory.build_choice_answer("choix 2")

    assert result is None


def test_build_choice_answer_success_same_dates(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    selected_doc = {
        "title": "Expo Archi",
        "location_name": "Musée Fabre",
        "city": "Montpellier",
        "first_date": "2026-03-01",
        "last_date": "2026-03-01",
        "event_type": "Exposition",
        "url": "http://expo.com",
    }

    memory.add_entry(
        question="Question",
        answer="Réponse",
        documents=[selected_doc],
    )

    result = memory.build_choice_answer("choix 1")

    assert result is not None
    assert result["question"] == "choix 1"
    assert result["n_docs"] == 1
    assert result["documents"] == [selected_doc]
    assert "Voici l'événement correspondant à votre choix :" in result["answer"]
    assert "Titre : Expo Archi" in result["answer"]
    assert "Lieu : Musée Fabre" in result["answer"]
    assert "Ville : Montpellier" in result["answer"]
    assert "Date : 2026-03-01" in result["answer"]
    assert "Type d'événement : Exposition" in result["answer"]
    assert "Lien : http://expo.com" in result["answer"]


def test_build_choice_answer_success_date_range(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    selected_doc = {
        "title": "Festival",
        "location_name": "Parc",
        "city": "Sète",
        "first_date": "2026-04-01",
        "last_date": "2026-04-05",
        "event_type": "Festival",
        "url": "http://festival.com",
    }

    memory.add_entry(
        question="Question",
        answer="Réponse",
        documents=[selected_doc],
    )

    result = memory.build_choice_answer("je prends le 1")

    assert result is not None
    assert "Date : du 2026-04-01 au 2026-04-05" in result["answer"]


def test_build_choice_answer_with_last_date_only(tmp_path: Path):
    memory = MemoryService(memory_file=str(tmp_path / "memory.json"))

    selected_doc = {
        "title": "Concert",
        "location_name": "Salle Y",
        "city": "Sète",
        "first_date": "",
        "last_date": "2026-05-10",
        "event_type": "",
        "url": "",
    }

    memory.add_entry(
        question="Question",
        answer="Réponse",
        documents=[selected_doc],
    )

    result = memory.build_choice_answer("le 1")

    assert result is not None
    assert "Date : 2026-05-10" in result["answer"]
    assert "Type d'événement" not in result["answer"]
    assert "Lien :" not in result["answer"]