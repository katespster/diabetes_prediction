from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report


RAW_DATA_PATH = Path("data/raw/diabetes_raw.csv")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def detect_target_column(df: pd.DataFrame) -> str:
    candidates = ["class", "Class", "Outcome", "outcome", "target"]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find target column. Available columns: {df.columns.tolist()}"
    )


def get_zero_as_missing_columns(df: pd.DataFrame) -> list[str]:
    """
    Возвращает список колонок, где нули стоит считать пропусками.
    Поддерживает и длинные, и короткие названия колонок.
    """
    candidate_groups = [
        ["Glucose", "glucose", "plas"],
        ["BloodPressure", "bloodpressure", "pres"],
        ["SkinThickness", "skinthickness", "skin"],
        ["Insulin", "insulin", "insu"],
        ["BMI", "bmi", "mass"],
    ]

    result = []
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


def replace_invalid_zeros_with_nan(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    cols_to_replace = get_zero_as_missing_columns(df)

    print("\nColumns where 0 will be treated as missing:")
    print(cols_to_replace)

    print("\nZero counts before replacement:")
    for col in cols_to_replace:
        zero_count = (df[col] == 0).sum()
        print(f"{col}: {zero_count}")

    for col in cols_to_replace:
        df[col] = df[col].replace(0, np.nan)

    print("\nMissing values after zero->NaN replacement:")
    print(df.isna().sum())

    if target_col in cols_to_replace:
        raise ValueError("Target column must not be included in zero->NaN replacement.")

    return df


def normalize_target(y: pd.Series) -> pd.Series:
    """
    Приводит target к 0/1.

    Поддерживает варианты:
    - tested_negative / tested_positive
    - negative / positive
    - no / yes
    - 0 / 1 (как числа или строки)
    """
    y_non_null = y.dropna()

    if y_non_null.empty:
        raise ValueError("Target column is empty or contains only NaN values.")

    normalized_str = y.astype("string").str.strip().str.lower()

    print("\nRaw target unique values:")
    print(sorted(normalized_str.dropna().unique().tolist()))

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

    unique_values = set(normalized_str.dropna().unique().tolist())

    if unique_values.issubset(set(mapping.keys())):
        return normalized_str.map(mapping).astype(int)

    try:
        return pd.to_numeric(y, errors="raise").astype(int)
    except Exception as e:
        raise ValueError(
            f"Unsupported target values: {sorted(y_non_null.astype(str).unique().tolist())}"
        ) from e


def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_DATA_PATH} not found. Run `python src/download_data.py` first."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print_basic_info(df)

    target_col = detect_target_column(df)
    print(f"\nDetected target column: {target_col}")

    df = replace_invalid_zeros_with_nan(df, target_col=target_col)

    X = df.drop(columns=[target_col])
    y = normalize_target(df[target_col])

    print("\nTarget distribution:")
    print(y.value_counts(dropna=False))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    model_path = MODEL_DIR / "model.pkl"
    metrics_path = REPORTS_DIR / "metrics.json"

    joblib.dump(model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()