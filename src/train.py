from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocess import prepare_training_data


RAW_DATA_PATH = Path("data/raw/diabetes_raw.csv")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_model(model_type: str, random_state: int, rf_n_estimators: int = 200, rf_max_depth: int = 5):
    if model_type == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_type == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=rf_n_estimators,
                        max_depth=rf_max_depth,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def run_experiment(
    model_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
    rf_n_estimators: int = 200,
    rf_max_depth: int = 5,
) -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_DATA_PATH} not found. Run `python -m src.download_data` first."
        )

    df = pd.read_csv(RAW_DATA_PATH)
    X, y, target_col = prepare_training_data(df, verbose=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_model(
        model_type=model_type,
        random_state=random_state,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
    )

    if model_type == "logistic_regression":
        run_name = "logreg-baseline"
    else:
        run_name = f"rf-n{rf_n_estimators}-depth{rf_max_depth}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("project", "diabetes-risk-prediction")
        mlflow.set_tag("stage", "baseline")
        mlflow.set_tag("model_family", model_type)

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("target_column", target_col)
        mlflow.log_param("n_rows", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("feature_columns", ",".join(X.columns.tolist()))
        mlflow.log_param("imputer_strategy", "median")

        if model_type == "logistic_regression":
            mlflow.log_param("scaler", "standard")
            mlflow.log_param("max_iter", 1000)

        if model_type == "random_forest":
            mlflow.log_param("rf_n_estimators", rf_n_estimators)
            mlflow.log_param("rf_max_depth", rf_max_depth)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        precision = float(precision_score(y_test, y_pred))
        recall = float(recall_score(y_test, y_pred))

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        report_text = classification_report(y_test, y_pred)

        model_path = MODEL_DIR / f"{model_type}.pkl"
        metrics_path = REPORTS_DIR / f"{model_type}_metrics.json"
        report_path = REPORTS_DIR / f"{model_type}_classification_report.txt"

        metrics = {
            "model_type": model_type,
            "accuracy": accuracy,
            "f1": f1,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "target_column": target_col,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "feature_columns": X.columns.tolist(),
        }

        joblib.dump(model, model_path)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(report_path))
        mlflow.sklearn.log_model(model, name="model")

        print(
            f"{model_type}: "
            f"accuracy={accuracy:.4f}, "
            f"f1={f1:.4f}, "
            f"roc_auc={roc_auc:.4f}, "
            f"precision={precision:.4f}, "
            f"recall={recall:.4f}"
        )


def main():
    mlflow.set_experiment("diabetes-risk-prediction")

    # Baseline Logistic Regression
    run_experiment(
        model_type="logistic_regression",
        test_size=0.2,
        random_state=42,
    )

    # Random Forest experiments
    run_experiment(
        model_type="random_forest",
        test_size=0.2,
        random_state=42,
        rf_n_estimators=100,
        rf_max_depth=5,
    )

    run_experiment(
        model_type="random_forest",
        test_size=0.2,
        random_state=42,
        rf_n_estimators=200,
        rf_max_depth=5,
    )

    run_experiment(
        model_type="random_forest",
        test_size=0.2,
        random_state=42,
        rf_n_estimators=300,
        rf_max_depth=7,
    )


if __name__ == "__main__":
    main()