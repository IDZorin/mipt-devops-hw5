import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import joblib

import mlflow
import mlflow.sklearn

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

RANDOM_STATE = 42


def set_seeds() -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)


def prepare_output_dirs() -> None:
    """Create required output directories."""
    for dirname in ["reports", "models"]:
        os.makedirs(dirname, exist_ok=True)


def load_iris_dataset(test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load iris dataset and split into train and test partitions."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    feature_names = iris.feature_names

    X = df[feature_names]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int | None = 5,
) -> RandomForestClassifier:
    """Train RandomForestClassifier on the provided data."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    report_dir: str = "reports",
) -> tuple[float, str]:
    """Evaluate model on test data and save classification report as text."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred)

    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Accuracy on test set: {acc:.4f}")
    print("Classification report saved to:", report_path)
    return acc, report_path


def run_deepchecks_full_suite(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    report_dir: str = "reports",
) -> str:
    """Run Deepchecks full_suite on train/test data and model, save HTML report."""
    train_ds = Dataset(
        X_train,
        label=y_train,
        cat_features=[],
        features=X_train.columns.tolist(),
    )
    test_ds = Dataset(
        X_test,
        label=y_test,
        cat_features=[],
        features=X_test.columns.tolist(),
    )

    suite = full_suite()
    result = suite.run(
        train_dataset=train_ds,
        test_dataset=test_ds,
        model=model,
    )

    os.makedirs(report_dir, exist_ok=True)
    html_path = os.path.join(report_dir, "deepchecks_full_suite.html")
    result.save_as_html(html_path, connected=False, as_widget=False)
    print("Deepchecks report saved to:", html_path)
    return html_path


def run_evidently_drift_report(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    report_dir: str = "reports",
) -> str:
    """Run Evidently data drift and data quality report, save HTML."""
    train_df = X_train.copy()
    train_df["target"] = y_train.values

    test_df = X_test.copy()
    test_df["target"] = y_test.values

    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ]
    )
    report.run(
        reference_data=train_df,
        current_data=test_df,
    )

    os.makedirs(report_dir, exist_ok=True)
    html_path = os.path.join(report_dir, "evidently_data_drift.html")
    report.save_html(html_path)
    print("Evidently report saved to:", html_path)
    return html_path


def run_mlflow_pipeline() -> None:
    """Run full ML pipeline under MLflow tracking."""
    set_seeds()
    prepare_output_dirs()

    X_train, X_test, y_train, y_test = load_iris_dataset()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("hw5_iris_experiment")

    n_estimators = 100
    max_depth = 5

    with mlflow.start_run(run_name="rf_iris_baseline"):
        model = train_random_forest(
            X_train,
            y_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )

        accuracy, cls_report_path = evaluate_model(
            model,
            X_test,
            y_test,
        )

        deepchecks_path = run_deepchecks_full_suite(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        evidently_path = run_evidently_drift_report(
            X_train,
            y_train,
            X_test,
            y_test,
        )

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.log_artifact(cls_report_path, artifact_path="reports")
        mlflow.log_artifact(deepchecks_path, artifact_path="reports")
        mlflow.log_artifact(evidently_path, artifact_path="reports")

        os.makedirs("models", exist_ok=True)
        local_model_path = os.path.join("models", "rf_iris_model.pkl")

        joblib.dump(model, local_model_path)
        print("Local model saved to:", local_model_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        run_id = mlflow.active_run().info.run_id
        print("MLflow run finished. Run id:", run_id)


if __name__ == "__main__":
    run_mlflow_pipeline()
