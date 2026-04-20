from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict(monkeypatch):
    def fake_predict(payload):
        return [
            {
                "prediction": 1,
                "probability": 0.8123,
            }
        ]

    # В app.main функция predict импортирована напрямую:
    # from src.predict import predict
    # Поэтому патчить нужно именно app.main.predict
    monkeypatch.setattr("app.main.predict", fake_predict)

    payload = {
        "preg": 2,
        "plas": 130,
        "pres": 70,
        "skin": 25,
        "insu": 120,
        "mass": 28.5,
        "pedi": 0.35,
        "age": 33,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "probability" in data

    assert data["prediction"] in [0, 1]
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_validation_error():
    # Специально передаём невалидное значение age
    payload = {
        "preg": 2,
        "plas": 130,
        "pres": 70,
        "skin": 25,
        "insu": 120,
        "mass": 28.5,
        "pedi": 0.35,
        "age": -1,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422