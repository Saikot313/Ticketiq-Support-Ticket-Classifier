import pytest
from fastapi.testclient import TestClient

from ticketclf import api


class FakeClassifier:
    def predict(self, texts, top_k=None):
        return [{
            "intent": "lost_or_stolen_card", "confidence": 0.93, "priority": "High",
            "needs_human_review": False,
            "top_k": [{"intent": "lost_or_stolen_card", "confidence": 0.93},
                      {"intent": "compromised_card", "confidence": 0.04}],
        } for _ in texts]


client = TestClient(api.app)


@pytest.fixture(autouse=True)
def reset_state():
    api.state.clear()
    yield
    api.state.clear()


def test_health_without_model():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is False


def test_predict_returns_503_without_model():
    assert client.post("/predict", json={"text": "my card is lost"}).status_code == 503


def test_predict_ok():
    api.state["clf"] = FakeClassifier()
    r = client.post("/predict", json={"text": "my card is lost"})
    assert r.status_code == 200
    body = r.json()
    assert body["priority"] == "High" and body["needs_human_review"] is False


def test_blank_text_rejected():
    api.state["clf"] = FakeClassifier()
    assert client.post("/predict", json={"text": "     "}).status_code == 422


def test_batch():
    api.state["clf"] = FakeClassifier()
    r = client.post("/predict/batch", json={"texts": ["card lost", "top up failed"]})
    assert r.status_code == 200 and r.json()["count"] == 2