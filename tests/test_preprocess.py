import json

import allure
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


def attach_json(data, name):
    allure.attach(
        json.dumps(data, indent=2, default=str),
        name=name,
        attachment_type=allure.attachment_type.JSON,
    )


@allure.epic("Diabetes Prediction ML Project")
@allure.feature("Data preprocessing")
@allure.story("Target column detection")
@allure.title("Detect target column in diabetes dataset")
@allure.severity(allure.severity_level.CRITICAL)
@pytest.mark.preprocess
@pytest.mark.regression
def test_detect_target_column():
    with allure.step("Prepare dataframe with diabetes features and target column"):
        df = pd.DataFrame(
            {
                "preg": [1, 2],
                "plas": [100, 120],
                "class": ["tested_negative", "tested_positive"],
            }
        )
        attach_json(df.to_dict(orient="records"), "Input dataframe")

    with allure.step("Detect target column"):
        target_col = detect_target_column(df)

    with allure.step("Check that detected target column is 'class'"):
        assert target_col == "class"


@allure.epic("Diabetes Prediction ML Project")
@allure.feature("Data preprocessing")
@allure.story("Invalid zero columns detection")
@allure.title("Detect columns where zero values should be treated as missing")
@allure.severity(allure.severity_level.NORMAL)
@pytest.mark.preprocess
@pytest.mark.regression
def test_get_zero_as_missing_columns():
    with allure.step("Prepare dataframe with zero values in medical feature columns"):
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
        attach_json(df.to_dict(orient="records"), "Input dataframe")

    with allure.step("Get columns where zero means missing value"):
        cols = get_zero_as_missing_columns(df)

    with allure.step("Check that only medical measurement columns are selected"):
        assert cols == ["plas", "pres", "skin", "insu", "mass"]


@allure.epic("Diabetes Prediction ML Project")
@allure.feature("Data preprocessing")
@allure.story("Replace invalid zero values")
@allure.title("Replace invalid zero values with NaN")
@allure.severity(allure.severity_level.CRITICAL)
@pytest.mark.preprocess
@pytest.mark.regression
def test_replace_invalid_zeros_with_nan():
    with allure.step("Prepare sample dataframe with invalid zero values"):
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
        attach_json(df.to_dict(orient="records"), "Input dataframe")

    with allure.step("Run zero-to-NaN replacement"):
        result = replace_invalid_zeros_with_nan(
            df,
            target_col="class",
            verbose=False,
        )
        attach_json(result.to_dict(orient="records"), "Processed dataframe")

    with allure.step("Check that invalid zero values were replaced with NaN"):
        assert pd.isna(result.loc[1, "plas"])
        assert pd.isna(result.loc[1, "pres"])
        assert pd.isna(result.loc[1, "skin"])
        assert pd.isna(result.loc[1, "insu"])
        assert pd.isna(result.loc[1, "mass"])

    with allure.step("Check that valid zero values and non-missing values were preserved"):
        assert result.loc[0, "preg"] == 0
        assert result.loc[1, "age"] == 40


@allure.epic("Diabetes Prediction ML Project")
@allure.feature("Data preprocessing")
@allure.story("Target normalization")
@allure.title("Normalize string target labels to binary values")
@allure.severity(allure.severity_level.CRITICAL)
@pytest.mark.preprocess
@pytest.mark.regression
def test_normalize_target_string_labels():
    with allure.step("Prepare target series with string labels"):
        y = pd.Series(["tested_negative", "tested_positive", "tested_negative"])
        attach_json(y.tolist(), "Input target labels")

    with allure.step("Normalize target labels"):
        result = normalize_target(y)
        attach_json(result.tolist(), "Normalized target labels")

    with allure.step("Check that negative class is 0 and positive class is 1"):
        assert result.tolist() == [0, 1, 0]


@allure.epic("Diabetes Prediction ML Project")
@allure.feature("Data preprocessing")
@allure.story("Target normalization")
@allure.title("Keep numeric target labels unchanged")
@allure.severity(allure.severity_level.NORMAL)
@pytest.mark.preprocess
@pytest.mark.regression
def test_normalize_target_numeric_labels():
    with allure.step("Prepare target series with numeric labels"):
        y = pd.Series([0, 1, 0, 1])
        attach_json(y.tolist(), "Input target labels")

    with allure.step("Normalize numeric target labels"):
        result = normalize_target(y)
        attach_json(result.tolist(), "Normalized target labels")

    with allure.step("Check that numeric labels are preserved"):
        assert result.tolist() == [0, 1, 0, 1]


@allure.epic("Diabetes Prediction ML Project")
@allure.feature("Payload transformation")
@allure.story("Convert API payload to dataframe")
@allure.title("Convert single prediction payload from dict to dataframe")
@allure.severity(allure.severity_level.CRITICAL)
@pytest.mark.preprocess
@pytest.mark.regression
def test_payload_to_dataframe_from_dict():
    with allure.step("Prepare single prediction payload as dictionary"):
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
        attach_json(payload, "Input payload")

    with allure.step("Convert payload to dataframe"):
        df = payload_to_dataframe(payload)
        attach_json(df.to_dict(orient="records"), "Output dataframe")

    with allure.step("Check dataframe shape and values"):
        assert df.shape == (1, 8)
        assert "plas" in df.columns
        assert df.iloc[0]["age"] == 33


@allure.epic("Diabetes Prediction ML Project")
@allure.feature("Payload transformation")
@allure.story("Convert API payload to dataframe")
@allure.title("Convert batch prediction payload from list to dataframe")
@allure.severity(allure.severity_level.CRITICAL)
@pytest.mark.preprocess
@pytest.mark.regression
def test_payload_to_dataframe_from_list():
    with allure.step("Prepare batch prediction payload as list of dictionaries"):
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
        attach_json(payload, "Input batch payload")

    with allure.step("Convert batch payload to dataframe"):
        df = payload_to_dataframe(payload)
        attach_json(df.to_dict(orient="records"), "Output dataframe")

    with allure.step("Check dataframe shape and second row values"):
        assert df.shape == (2, 8)
        assert df.iloc[1]["age"] == 29


@allure.epic("Diabetes Prediction ML Project")
@allure.feature("Payload transformation")
@allure.story("Payload validation")
@allure.title("Raise validation error for empty payload list")
@allure.severity(allure.severity_level.NORMAL)
@pytest.mark.preprocess
@pytest.mark.regression
def test_payload_to_dataframe_empty_list_raises():
    with allure.step("Prepare empty payload list"):
        payload = []
        attach_json(payload, "Empty input payload")

    with allure.step("Check that empty payload list raises ValueError"):
        with pytest.raises(ValueError, match="Payload list is empty"):
            payload_to_dataframe(payload)


@allure.epic("Diabetes Prediction ML Project")
@allure.feature("Feature validation")
@allure.story("Align input features")
@allure.title("Reorder input dataframe columns according to expected feature order")
@allure.severity(allure.severity_level.CRITICAL)
@pytest.mark.preprocess
@pytest.mark.regression
def test_validate_and_align_features_reorders_columns():
    with allure.step("Prepare dataframe with columns in non-standard order"):
        df = pd.DataFrame(
            {
                "plas": [130],
                "preg": [2],
                "age": [33],
            }
        )
        expected_columns = ["preg", "plas", "age"]

        attach_json(df.to_dict(orient="records"), "Input dataframe")
        attach_json(expected_columns, "Expected columns")

    with allure.step("Validate and align dataframe features"):
        result = validate_and_align_features(df, expected_columns)
        attach_json(result.to_dict(orient="records"), "Aligned dataframe")

    with allure.step("Check that columns were reordered correctly"):
        assert result.columns.tolist() == ["preg", "plas", "age"]


@allure.epic("Diabetes Prediction ML Project")
@allure.feature("Feature validation")
@allure.story("Handle missing features")
@allure.title("Add missing expected columns as NaN")
@allure.severity(allure.severity_level.CRITICAL)
@pytest.mark.preprocess
@pytest.mark.regression
def test_validate_and_align_features_adds_missing_columns_as_nan():
    with allure.step("Prepare dataframe with missing expected column"):
        df = pd.DataFrame(
            {
                "preg": [2],
                "plas": [130],
            }
        )
        expected_columns = ["preg", "plas", "age"]

        attach_json(df.to_dict(orient="records"), "Input dataframe")
        attach_json(expected_columns, "Expected columns")

    with allure.step("Validate and align dataframe features"):
        result = validate_and_align_features(df, expected_columns)
        attach_json(result.to_dict(orient="records"), "Aligned dataframe")

    with allure.step("Check that missing column was added"):
        assert result.columns.tolist() == ["preg", "plas", "age"]

    with allure.step("Check that missing feature value is NaN"):
        assert pd.isna(result.loc[0, "age"])


@allure.epic("Diabetes Prediction ML Project")
@allure.feature("Feature validation")
@allure.story("Reject unexpected features")
@allure.title("Raise validation error for unexpected input columns")
@allure.severity(allure.severity_level.CRITICAL)
@pytest.mark.preprocess
@pytest.mark.regression
def test_validate_and_align_features_unexpected_columns_raises():
    with allure.step("Prepare dataframe with unexpected input column"):
        df = pd.DataFrame(
            {
                "preg": [2],
                "plas": [130],
                "unexpected": [999],
            }
        )
        expected_columns = ["preg", "plas"]

        attach_json(df.to_dict(orient="records"), "Input dataframe")
        attach_json(expected_columns, "Expected columns")

    with allure.step("Check that unexpected column raises ValueError"):
        with pytest.raises(ValueError, match="Unexpected columns in input"):
            validate_and_align_features(df, expected_columns)