from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TARGET_CANDIDATES = ["class", "Class", "Outcome", "outcome", "target"]


def detect_target_column(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find target column. Available columns: {df.columns.tolist()}"
    )


def get_zero_as_missing_columns(df: pd.DataFrame) -> list[str]:
    """
    Возвращает список колонок, где 0 следует трактовать как пропуск.
    Поддерживает как короткие названия OpenML/Pima, так и более длинные варианты.
    """
    candidate_groups = [
        ["Glucose", "glucose", "plas"],
        ["BloodPressure", "bloodpressure", "pres"],
        ["SkinThickness", "skinthickness", "skin"],
        ["Insulin", "insulin", "insu"],
        ["BMI", "bmi", "mass"],
    ]

    result: list[str] = []
    columns_lower_map = {col.lower(): col for col in df.columns}

    for aliases in candidate_groups:
        for alias in aliases:
            if alias.lower() in columns_lower_map:
                result.append(columns_lower_map[alias.lower()])
                break

    return result


def print_basic_info(df: pd.DataFrame) -> None:
    print("Dataset shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDtypes:")
    print(df.dtypes)

    print("\nMissing values before zero->NaN replacement:")
    print(df.isna().sum())


def replace_invalid_zeros_with_nan(
    df: pd.DataFrame,
    target_col: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Заменяет физиологически нереалистичные нули на NaN.
    target_col исключается из обработки.
    """
    df = df.copy()
    cols_to_replace = get_zero_as_missing_columns(df)

    if target_col is not None and target_col in cols_to_replace:
        raise ValueError("Target column must not be included in zero->NaN replacement.")

    if verbose:
        print("\nColumns where 0 will be treated as missing:")
        print(cols_to_replace)

        print("\nZero counts before replacement:")
        for col in cols_to_replace:
            zero_count = (df[col] == 0).sum()
            print(f"{col}: {zero_count}")

    for col in cols_to_replace:
        df[col] = df[col].replace(0, np.nan)

    if verbose:
        print("\nMissing values after zero->NaN replacement:")
        print(df.isna().sum())

    return df




def normalize_target(y: pd.Series) -> pd.Series:
    """
    Приводит target к 0/1.
    Поддерживает строковые значения:
    - tested_negative / tested_positive
    - negative / positive
    - no / yes
    """

    mapping = {
        "tested_negative": 0,
        "tested_positive": 1,
        "negative": 0,
        "positive": 1,
        "no": 0,
        "yes": 1,
        "0": 0,
        "1": 1,
    }

    # Если target строковый/категориальный — нормализуем через lowercase map
    if pd.api.types.is_string_dtype(y) or pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        lowered = y.astype(str).str.strip().str.lower()

        unique_values = sorted(lowered.dropna().unique().tolist())
        print("\nRaw target unique values:")
        print(unique_values)

        if set(unique_values).issubset(set(mapping.keys())):
            return lowered.map(mapping).astype(int)

        raise ValueError(
            f"Unsupported target labels: {unique_values}. "
            f"Expected labels compatible with: {list(mapping.keys())}"
        )

    # Если target уже числовой
    return y.astype(int)

def prepare_training_data(
    df: pd.DataFrame,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Полный preprocessing для train:
    1. находит target
    2. заменяет ложные нули на NaN
    3. отделяет X и y
    4. нормализует target к 0/1
    """
    if verbose:
        print_basic_info(df)

    target_col = detect_target_column(df)

    if verbose:
        print(f"\nDetected target column: {target_col}")

    df = replace_invalid_zeros_with_nan(df, target_col=target_col, verbose=verbose)

    X = df.drop(columns=[target_col])
    y = normalize_target(df[target_col])

    if verbose:
        print("\nTarget distribution:")
        print(y.value_counts(dropna=False))

    return X, y, target_col


def prepare_inference_data(
    df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Preprocessing для инференса:
    - target тут не нужен
    - ложные нули заменяются на NaN
    """
    return replace_invalid_zeros_with_nan(df, target_col=None, verbose=verbose)


def payload_to_dataframe(payload: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    """
    Преобразует один объект или список объектов в DataFrame.
    """
    if isinstance(payload, dict):
        return pd.DataFrame([payload])

    if isinstance(payload, list):
        if not payload:
            raise ValueError("Payload list is empty.")
        return pd.DataFrame(payload)

    raise TypeError("Payload must be a dict or a list of dicts.")