import json

import allure
import pytest


@allure.epic("Diabetes Prediction ML Project")
@allure.feature("FastAPI service")
@allure.story("Healthcheck endpoint")
@allure.title("Check that /health endpoint returns service status")
@allure.severity(allure.severity_level.CRITICAL)
@pytest.mark.api
@pytest.mark.smoke
def test_healthcheck(client):
    with allure.step("Send GET request to /health"):
        response = client.get("/health")

    with allure.step("Check response status code"):
        assert response.status_code == 200

    with allure.step("Check response body"):
        response_body = response.json()

        allure.attach(
            json.dumps(response_body, indent=2),
            name="Healthcheck response body",
            attachment_type=allure.attachment_type.JSON,
        )

        assert response_body["status"] == "ok"


@allure.epic("Diabetes Prediction ML Project")
@allure.feature("FastAPI service")
@allure.story("Diabetes prediction endpoint")
@allure.title("Check that /predict returns valid prediction and probability")
@allure.severity(allure.severity_level.CRITICAL)
@pytest.mark.api
@pytest.mark.ml
@pytest.mark.regression
def test_predict(client, monkeypatch):
    with allure.step("Mock ML prediction function"):
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

    with allure.step("Prepare valid prediction request payload"):
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

        allure.attach(
            json.dumps(payload, indent=2),
            name="Prediction request payload",
            attachment_type=allure.attachment_type.JSON,
        )

    with allure.step("Send POST request to /predict"):
        response = client.post("/predict", json=payload)

    with allure.step("Check successful response status code"):
        assert response.status_code == 200

    with allure.step("Check prediction response structure and values"):
        data = response.json()

        allure.attach(
            json.dumps(data, indent=2),
            name="Prediction response body",
            attachment_type=allure.attachment_type.JSON,
        )

        assert "prediction" in data
        assert "probability" in data

        assert data["prediction"] in [0, 1]
        assert isinstance(data["probability"], float)
        assert 0.0 <= data["probability"] <= 1.0


@allure.epic("Diabetes Prediction ML Project")
@allure.feature("FastAPI service")
@allure.story("Input validation")
@allure.title("Check that /predict rejects invalid input data")
@allure.severity(allure.severity_level.NORMAL)
@pytest.mark.api
@pytest.mark.regression
def test_predict_validation_error(client):
    with allure.step("Prepare invalid prediction request payload with negative age"):
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

        allure.attach(
            json.dumps(payload, indent=2),
            name="Invalid prediction request payload",
            attachment_type=allure.attachment_type.JSON,
        )

    with allure.step("Send POST request to /predict"):
        response = client.post("/predict", json=payload)

    with allure.step("Check that API returns validation error"):
        assert response.status_code == 422

    with allure.step("Check validation error response body"):
        data = response.json()

        allure.attach(
            json.dumps(data, indent=2),
            name="Validation error response body",
            attachment_type=allure.attachment_type.JSON,
        )

        assert "detail" in data