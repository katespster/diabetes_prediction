from src.preprocess import normalize_target, payload_to_dataframe
import pandas as pd


def test_normalize_target_string_labels():
    y = pd.Series(["tested_negative", "tested_positive", "tested_negative"])
    result = normalize_target(y)

    assert result.tolist() == [0, 1, 0]


def test_payload_to_dataframe_from_dict():
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

    df = payload_to_dataframe(payload)

    assert df.shape == (1, 8)
    assert "plas" in df.columns
    assert df.iloc[0]["age"] == 33