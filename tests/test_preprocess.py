import pandas as pd
import pytest

from src.preprocess import (
    detect_target_column,
    get_zero_as_missing_columns,
    normalize_target,
    payload_to_dataframe,
    replace_invalid_zeros_with_nan,
    validate_and_align_features,
)


def test_detect_target_column():
    df = pd.DataFrame(
        {
            "preg": [1, 2],
            "plas": [100, 120],
            "class": ["tested_negative", "tested_positive"],
        }
    )

    target_col = detect_target_column(df)

    assert target_col == "class"


def test_get_zero_as_missing_columns():
    df = pd.DataFrame(
        {
            "preg": [0, 1],
            "plas": [100, 0],
            "pres": [70, 0],
            "skin": [20, 0],
            "insu": [80, 0],
            "mass": [25.0, 0.0],
            "pedi": [0.1, 0.2],
            "age": [30, 40],
        }
    )

    cols = get_zero_as_missing_columns(df)

    assert cols == ["plas", "pres", "skin", "insu", "mass"]


def test_replace_invalid_zeros_with_nan():
    df = pd.DataFrame(
        {
            "preg": [0, 1],
            "plas": [100, 0],
            "pres": [70, 0],
            "skin": [20, 0],
            "insu": [80, 0],
            "mass": [25.0, 0.0],
            "pedi": [0.1, 0.2],
            "age": [30, 40],
            "class": ["tested_negative", "tested_positive"],
        }
    )

    result = replace_invalid_zeros_with_nan(df, target_col="class", verbose=False)

    assert pd.isna(result.loc[1, "plas"])
    assert pd.isna(result.loc[1, "pres"])
    assert pd.isna(result.loc[1, "skin"])
    assert pd.isna(result.loc[1, "insu"])
    assert pd.isna(result.loc[1, "mass"])

    assert result.loc[0, "preg"] == 0
    assert result.loc[1, "age"] == 40


def test_normalize_target_string_labels():
    y = pd.Series(["tested_negative", "tested_positive", "tested_negative"])
    result = normalize_target(y)

    assert result.tolist() == [0, 1, 0]


def test_normalize_target_numeric_labels():
    y = pd.Series([0, 1, 0, 1])
    result = normalize_target(y)

    assert result.tolist() == [0, 1, 0, 1]


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


def test_payload_to_dataframe_from_list():
    payload = [
        {
            "preg": 2,
            "plas": 130,
            "pres": 70,
            "skin": 25,
            "insu": 120,
            "mass": 28.5,
            "pedi": 0.35,
            "age": 33,
        },
        {
            "preg": 1,
            "plas": 110,
            "pres": 68,
            "skin": 22,
            "insu": 100,
            "mass": 26.1,
            "pedi": 0.28,
            "age": 29,
        },
    ]

    df = payload_to_dataframe(payload)

    assert df.shape == (2, 8)
    assert df.iloc[1]["age"] == 29


def test_payload_to_dataframe_empty_list_raises():
    with pytest.raises(ValueError, match="Payload list is empty"):
        payload_to_dataframe([])


def test_validate_and_align_features_reorders_columns():
    df = pd.DataFrame(
        {
            "plas": [130],
            "preg": [2],
            "age": [33],
        }
    )

    expected_columns = ["preg", "plas", "age"]

    result = validate_and_align_features(df, expected_columns)

    assert result.columns.tolist() == ["preg", "plas", "age"]


def test_validate_and_align_features_adds_missing_columns_as_nan():
    df = pd.DataFrame(
        {
            "preg": [2],
            "plas": [130],
        }
    )

    expected_columns = ["preg", "plas", "age"]

    result = validate_and_align_features(df, expected_columns)

    assert result.columns.tolist() == ["preg", "plas", "age"]
    assert pd.isna(result.loc[0, "age"])


def test_validate_and_align_features_unexpected_columns_raises():
    df = pd.DataFrame(
        {
            "preg": [2],
            "plas": [130],
            "unexpected": [999],
        }
    )

    expected_columns = ["preg", "plas"]

    with pytest.raises(ValueError, match="Unexpected columns in input"):
        validate_and_align_features(df, expected_columns)