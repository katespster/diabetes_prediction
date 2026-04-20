from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocess import payload_to_dataframe, prepare_inference_data


MODEL_PATH = Path("models/model.pkl")


def load_model(model_path: str | Path = MODEL_PATH):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train the model first."
        )

    return joblib.load(model_path)


def get_model_feature_names(model) -> list[str]:
    """
    После fit у sklearn Pipeline обычно хранит feature_names_in_.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    raise AttributeError(
        "Model does not contain feature_names_in_. "
        "Make sure it was trained on a pandas DataFrame."
    )


def validate_and_align_features(
    df: pd.DataFrame,
    expected_columns: list[str],
) -> pd.DataFrame:
    """
    1. Проверяет неожиданные колонки
    2. Добавляет отсутствующие ожидаемые колонки как NaN
    3. Выставляет правильный порядок колонок
    """
    unexpected_columns = sorted(set(df.columns) - set(expected_columns))
    if unexpected_columns:
        raise ValueError(
            f"Unexpected columns in input: {unexpected_columns}. "
            f"Expected only: {expected_columns}"
        )

    aligned_df = df.reindex(columns=expected_columns)
    return aligned_df


def make_inference_frame(
    payload: dict[str, Any] | list[dict[str, Any]],
    expected_columns: list[str],
) -> pd.DataFrame:
    raw_df = payload_to_dataframe(payload)
    prepared_df = prepare_inference_data(raw_df, verbose=False)
    aligned_df = validate_and_align_features(prepared_df, expected_columns)
    return aligned_df


def predict(
    payload: dict[str, Any] | list[dict[str, Any]],
    model_path: str | Path = MODEL_PATH,
) -> list[dict[str, Any]]:
    """
    Выполняет предсказание для одного объекта или списка объектов.
    Возвращает список словарей с prediction и probability.
    """
    model = load_model(model_path)
    expected_columns = get_model_feature_names(model)

    X = make_inference_frame(payload, expected_columns=expected_columns)

    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = [None] * len(predictions)

    results: list[dict[str, Any]] = []
    for pred, prob in zip(predictions, probabilities):
        item = {"prediction": int(pred)}
        if prob is not None:
            item["probability"] = float(prob)
        results.append(item)

    return results


if __name__ == "__main__":
    example_payload = {
        "preg": 2,
        "plas": 130,
        "pres": 70,
        "skin": 25,
        "insu": 120,
        "mass": 28.5,
        "pedi": 0.35,
        "age": 33,
    }

    result = predict(example_payload)
    print(result)