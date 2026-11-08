"""Reusable mixed-type classification benchmark for uploaded CSV datasets."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV


MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbors",
    "Gaussian Naive Bayes",
    "Support Vector Machine",
]
MAX_ROWS = 15_000
MAX_FEATURES = 100
MAX_CLASSES = 50
MAX_CATEGORIES_PER_FEATURE = 1_000


def _preprocessor(*, ordinal_categories: bool = False) -> ColumnTransformer:
    """Build leakage-safe imputing and encoding for numeric/mixed CSV columns."""
    if ordinal_categories:
        categorical_encoder: Any = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    else:
        categorical_encoder = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", categorical_encoder),
                    ]
                ),
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        remainder="drop",
    )


def _make_models(training_labels: pd.Series) -> dict[str, Any]:
    """Create fresh estimators suitable for the current training partition."""
    smallest_class = int(training_labels.value_counts().min())
    calibration_folds = min(3, smallest_class)
    if calibration_folds < 2:
        raise ValueError(
            "There are too few training rows in at least one class for SVM "
            "probability calibration. Add more examples for each class."
        )

    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=min(5, len(training_labels))
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": CalibratedClassifierCV(
            SVC(kernel="linear", random_state=42, cache_size=256),
            method="sigmoid",
            cv=calibration_folds,
            ensemble=False,
        ),
    }


def compare_classifiers(
    dataset: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    include_probabilities: bool = False,
) -> (
    tuple[
        pd.DataFrame,
        dict[str, np.ndarray],
        pd.Series,
        list[str],
        pd.DataFrame,
    ]
    | tuple[
        pd.DataFrame,
        dict[str, np.ndarray],
        pd.Series,
        list[str],
        pd.DataFrame,
        dict[str, np.ndarray],
    ]
):
    """Compare five classifiers on one stratified holdout from any labeled CSV.

    Returns the comparison table, predictions by model, holdout labels, sorted
    class labels, and training-partition feature relevance scores. When
    ``include_probabilities`` is true, append the per-model holdout probabilities.
    """
    if target_column not in dataset.columns:
        raise ValueError(f"Choose a target column present in the CSV.")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("Test split must be between 10% and 40%.")

    data = dataset.copy()
    if data.empty or len(data) < 20:
        raise ValueError("Upload a dataset with at least 20 labeled rows.")
    if len(data) > MAX_ROWS:
        raise ValueError(
            f"Dataset has {len(data):,} rows; the interactive benchmark limit "
            f"is {MAX_ROWS:,} rows to keep comparisons responsive."
        )
    if data.columns.duplicated().any():
        raise ValueError("CSV column names must be unique.")
    if any(not isinstance(column, str) or not column.strip() for column in data.columns):
        raise ValueError("CSV columns must have non-empty text names.")
    if data[target_column].isna().any():
        raise ValueError(
            f"Target '{target_column}' contains missing values. Remove those rows first."
        )

    features = data.drop(columns=[target_column])
    numeric_columns = features.select_dtypes(include=np.number).columns
    features.loc[:, numeric_columns] = features.loc[:, numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    features = features.dropna(axis=1, how="all")
    if features.shape[1] == 0:
        raise ValueError("The CSV must have at least one usable input feature.")
    if features.shape[1] > MAX_FEATURES:
        raise ValueError(
            f"Dataset has {features.shape[1]} usable features; the interactive "
            f"benchmark limit is {MAX_FEATURES}."
        )
    categorical = features.select_dtypes(exclude=np.number)
    high_cardinality = [
        str(column)
        for column in categorical.columns
        if categorical[column].nunique(dropna=True) > MAX_CATEGORIES_PER_FEATURE
    ]
    if high_cardinality:
        raise ValueError(
            "Categorical features have too many distinct values for a responsive "
            "comparison (limit 1,000 per column): "
            + ", ".join(high_cardinality[:5])
        )

    labels = data[target_column].astype(str)
    class_counts = labels.value_counts()
    if len(class_counts) < 2:
        raise ValueError("The target must contain at least two classes.")
    if len(class_counts) > MAX_CLASSES:
        raise ValueError(
            f"Target has {len(class_counts)} classes; the interactive benchmark "
            f"limit is {MAX_CLASSES}."
        )
    if class_counts.min() < 3:
        raise ValueError(
            "Each target class needs at least 3 rows for a stratified split "
            "and calibrated SVM comparison."
        )

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError as error:
        raise ValueError(
            "Could not create a stratified split. Add more rows per class or "
            "use a larger test split."
        ) from error

    class_labels = sorted(labels.unique().tolist())
    encoded_labels, _ = pd.factorize(y_train, sort=True)
    relevance_frame = x_train.copy()
    discrete_features: list[bool] = []
    for column in relevance_frame.columns:
        if pd.api.types.is_numeric_dtype(relevance_frame[column]):
            numeric_values = pd.to_numeric(relevance_frame[column], errors="coerce")
            numeric_values = numeric_values.replace([np.inf, -np.inf], np.nan)
            relevance_frame[column] = numeric_values.fillna(
                numeric_values.median()
            ).fillna(0)
            discrete_features.append(False)
        else:
            category_codes, _ = pd.factorize(relevance_frame[column], sort=True)
            relevance_frame[column] = category_codes
            discrete_features.append(True)
    relevance_scores = mutual_info_classif(
        relevance_frame.to_numpy(dtype=float),
        encoded_labels,
        discrete_features=np.asarray(discrete_features, dtype=bool),
        n_neighbors=min(3, max(1, len(y_train) - 1)),
        random_state=random_state,
    )
    feature_relevance = (
        pd.DataFrame(
            {"Feature": relevance_frame.columns, "Importance": relevance_scores}
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    probabilities_by_model: dict[str, np.ndarray] = {}
    models = _make_models(y_train)

    for model_name in MODEL_NAMES:
        estimator = models[model_name]
        preprocessor = _preprocessor(
            ordinal_categories=(model_name == "Gaussian Naive Bayes")
        )
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )
        training_started = perf_counter()
        pipeline.fit(x_train, y_train)
        training_time = perf_counter() - training_started
        prediction_started = perf_counter()
        predictions = pipeline.predict(x_test)
        prediction_time = perf_counter() - prediction_started
        probabilities = pipeline.predict_proba(x_test)
        model_classes = list(pipeline.classes_)
        if len(model_classes) == 2:
            positive_class_index = model_classes.index(class_labels[-1])
            auc = float(
                roc_auc_score(
                    y_test,
                    probabilities[:, positive_class_index],
                )
            )
        else:
            auc = float(
                roc_auc_score(
                    y_test,
                    probabilities,
                    labels=model_classes,
                    multi_class="ovr",
                    average="weighted",
                )
            )

        rows.append(
            {
                "Model": model_name,
                "Accuracy": float(accuracy_score(y_test, predictions)),
                "AUC": auc,
                "Precision": float(
                    precision_score(
                        y_test, predictions, average="weighted", zero_division=0
                    )
                ),
                "Recall": float(
                    recall_score(
                        y_test, predictions, average="weighted", zero_division=0
                    )
                ),
                "F1": float(
                    f1_score(y_test, predictions, average="weighted", zero_division=0)
                ),
                "MCC": float(matthews_corrcoef(y_test, predictions)),
                "Training Time (s)": float(training_time),
                "Prediction Time (s)": float(prediction_time),
            }
        )
        predictions_by_model[model_name] = predictions
        probabilities_by_model[model_name] = probabilities

    results = pd.DataFrame(rows)
    benchmark_output = (
        results,
        predictions_by_model,
        y_test,
        class_labels,
        feature_relevance,
    )
    if include_probabilities:
        return (*benchmark_output, probabilities_by_model)
    return benchmark_output
