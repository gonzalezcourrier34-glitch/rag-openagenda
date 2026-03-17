import requests

BASE_URL = "http://127.0.0.1:8000"


def test_health():
    response = requests.get(f"{BASE_URL}/health", timeout=30)
    print("HEALTH")
    print(response.status_code)
    print(response.json())
    print("-" * 60)


def test_rebuild():
    response = requests.post(f"{BASE_URL}/rebuild", timeout=120)
    print("REBUILD")
    print(response.status_code)
    print(response.json())
    print("-" * 60)


def test_ask():
    payload = {
        "question": "Je cherche une exposition d'architecture à Montpellier"
    }
    response = requests.post(f"{BASE_URL}/ask", json=payload, timeout=120)
    print("ASK")
    print(response.status_code)
    print(response.json())
    print("-" * 60)


if __name__ == "__main__":
    test_health()
    test_rebuild()
    test_ask()