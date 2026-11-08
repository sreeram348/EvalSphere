"""Reusable mixed-type supervised and unsupervised benchmarks."""

from __future__ import annotations

import json
import re
from io import BytesIO, StringIO
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, overload
from zipfile import BadZipFile
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import sparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.calibration import CalibratedClassifierCV


MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbors",
    "Gaussian Naive Bayes",
    "Support Vector Machine",
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "Neural Network (MLP)",
]
SUPPORTED_EXTENSIONS = (
    ".csv",
    ".xlsx",
    ".json",
    ".parquet",
    ".html",
    ".htm",
    ".xml",
    ".md",
    ".markdown",
)
MAX_ROWS = 15_000
MAX_FEATURES = 100
MAX_CLASSES = 50
MAX_CATEGORIES_PER_FEATURE = 1_000
MAX_CLUSTER_EVALUATION_ROWS = 2_000


def dataset_sources(file_name: str, raw_data: bytes) -> list[str]:
    """Return selectable worksheets or HTML tables when an upload has several."""
    extension = Path(file_name).suffix.lower()
    if extension == ".xlsx":
        try:
            return list(pd.ExcelFile(BytesIO(raw_data)).sheet_names)
        except (ValueError, ImportError, BadZipFile) as error:
            raise ValueError(f"Could not inspect the Excel workbook: {error}") from error
    if extension in {".html", ".htm"}:
        try:
            tables = _read_html_tables(raw_data)
        except (ValueError, ImportError) as error:
            raise ValueError(f"Could not find readable tables in the HTML file: {error}") from error
        return [f"Table {index + 1}" for index in range(len(tables))]
    return []


def _select_source(sources: list[str], source: str | None, label: str) -> str:
    if source is not None:
        if source not in sources:
            raise ValueError(f"Selected {label} was not found in the uploaded file.")
        return source
    if not sources:
        raise ValueError(f"The uploaded file contains no readable {label}s.")
    return sources[0]


def _read_html_tables(raw_data: bytes) -> list[pd.DataFrame]:
    try:
        html = raw_data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("HTML tables must use UTF-8 encoding.") from error
    return pd.read_html(StringIO(html))


def _read_markdown_table(raw_data: bytes) -> pd.DataFrame:
    try:
        text = raw_data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Markdown tables must use UTF-8 encoding.") from error

    table_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("|") and "|" in line.strip()[1:]
    ]
    if len(table_lines) < 2:
        raise ValueError("The Markdown file must contain a pipe-delimited table.")

    def split_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    rows = [split_row(line) for line in table_lines]
    headers = rows[0]
    separator = re.compile(r":?-{3,}:?")
    if len(rows) > 1 and all(separator.fullmatch(cell.replace(" ", "")) for cell in rows[1]):
        rows.pop(1)
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Markdown table rows must have the same number of columns.")
    return pd.DataFrame(rows[1:], columns=headers)


def read_dataset(
    file_name: str,
    raw_data: bytes,
    *,
    source: str | None = None,
) -> pd.DataFrame:
    """Read a supported tabular file without executing uploaded content."""
    extension = Path(file_name).suffix.lower()
    try:
        if extension == ".csv":
            data = pd.read_csv(BytesIO(raw_data), encoding="utf-8-sig")
        elif extension == ".xlsx":
            sheets = dataset_sources(file_name, raw_data)
            sheet_name = _select_source(sheets, source, "worksheet")
            data = pd.read_excel(BytesIO(raw_data), sheet_name=sheet_name, engine="openpyxl")
        elif extension == ".json":
            try:
                data = pd.read_json(BytesIO(raw_data), orient="records")
            except ValueError:
                value = json.loads(raw_data.decode("utf-8-sig"))
                if isinstance(value, dict):
                    value = value.get("data", value.get("records", value))
                data = pd.json_normalize(value)
        elif extension == ".parquet":
            data = pd.read_parquet(BytesIO(raw_data))
        elif extension in {".html", ".htm"}:
            tables = _read_html_tables(raw_data)
            sources = [f"Table {index + 1}" for index in range(len(tables))]
            selected = _select_source(sources, source, "HTML table")
            data = tables[sources.index(selected)]
        elif extension == ".xml":
            try:
                ElementTree.fromstring(raw_data)
            except ElementTree.ParseError as error:
                raise ValueError(f"Could not read XML: {error}") from error
            data = pd.read_xml(BytesIO(raw_data))
        elif extension in {".md", ".markdown"}:
            data = _read_markdown_table(raw_data)
        else:
            supported = ", ".join(SUPPORTED_EXTENSIONS)
            raise ValueError(f"Unsupported file extension. Supported formats: {supported}.")
    except (pa.ArrowException, BadZipFile) as error:
        raise ValueError(f"Could not read {extension or 'the uploaded file'}: {error}") from error
    except ValueError:
        raise
    except (UnicodeDecodeError, ImportError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {extension or 'the uploaded file'}: {error}") from error

    if data.empty:
        raise ValueError("The uploaded file contains no data rows.")
    if len(data.columns) == 0:
        raise ValueError("The uploaded file must have column headers.")
    return data


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
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=1,
            random_state=42,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=1,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=12,
            random_state=42,
        ),
    }


def _classifier_param_grid(model_name: str) -> dict[str, list[Any]]:
    grids: dict[str, dict[str, list[Any]]] = {
        "Logistic Regression": {"classifier__C": [0.1, 1.0, 10.0]},
        "Decision Tree": {
            "classifier__max_depth": [None, 8, 16],
            "classifier__min_samples_leaf": [1, 2],
        },
        "K-Nearest Neighbors": {"classifier__n_neighbors": [1, 2]},
        "Gaussian Naive Bayes": {
            "classifier__var_smoothing": [1e-9, 1e-8, 1e-7]
        },
        "Support Vector Machine": {
            "classifier__estimator__C": [0.1, 1.0, 10.0]
        },
        "Random Forest": {
            "classifier__max_depth": [None, 12],
            "classifier__min_samples_leaf": [1, 2],
        },
        "Extra Trees": {
            "classifier__max_depth": [None, 12],
            "classifier__min_samples_leaf": [1, 2],
        },
        "Gradient Boosting": {
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [2, 3],
        },
        "Neural Network (MLP)": {
            "classifier__hidden_layer_sizes": [(32,), (64,)],
            "classifier__alpha": [0.0001, 0.001],
        },
    }
    return grids[model_name]


@overload
def compare_classifiers(
    dataset: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    include_probabilities: Literal[False] = False,
    tune_best: bool = False,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    pd.Series,
    list[str],
    pd.DataFrame,
]: ...


@overload
def compare_classifiers(
    dataset: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    include_probabilities: Literal[True],
    tune_best: bool = False,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    pd.Series,
    list[str],
    pd.DataFrame,
    dict[str, np.ndarray],
]: ...


def compare_classifiers(
    dataset: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    include_probabilities: bool = False,
    tune_best: bool = False,
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
    """Compare classifiers on one stratified holdout from a labeled table.

    Returns the comparison table, predictions by model, holdout labels, sorted
    class labels, and training-partition feature relevance scores. When
    ``include_probabilities`` is true, append holdout probabilities. If
    ``tune_best`` is true, tune the leading model using a bounded training-only
    cross-validation search before evaluating it on the holdout.
    """
    if target_column not in dataset.columns:
        raise ValueError("Choose a target column present in the dataset.")
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
        raise ValueError("Dataset column names must be unique.")
    if any(not isinstance(column, str) or not column.strip() for column in data.columns):
        raise ValueError("Dataset columns must have non-empty text names.")
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
        raise ValueError("The dataset must have at least one usable input feature.")
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
    results["Tuned parameters"] = ""
    if tune_best:
        winner_index = results["F1"].idxmax()
        winner_name = str(results.at[winner_index, "Model"])
        if int(y_train.value_counts().min()) < 3:
            results.loc[winner_index, "Tuned parameters"] = (
                "Skipped: at least 3 training rows per class are needed for CV tuning"
            )
        else:
            winner_pipeline = Pipeline(
                [
                    (
                        "preprocessor",
                        _preprocessor(
                            ordinal_categories=(winner_name == "Gaussian Naive Bayes")
                        ),
                    ),
                    ("classifier", _make_models(y_train)[winner_name]),
                ]
            )
            search = GridSearchCV(
                winner_pipeline,
                _classifier_param_grid(winner_name),
                scoring="f1_weighted",
                cv=min(3, int(y_train.value_counts().min())),
                n_jobs=1,
                refit=True,
            )
            tuning_started = perf_counter()
            search.fit(x_train, y_train)
            tuning_time = perf_counter() - tuning_started
            tuned_pipeline = search.best_estimator_
            tuned_predictions = tuned_pipeline.predict(x_test)
            tuned_probabilities = tuned_pipeline.predict_proba(x_test)
            tuned_classes = list(tuned_pipeline.classes_)
            if len(tuned_classes) == 2:
                tuned_auc = float(
                    roc_auc_score(
                        y_test,
                        tuned_probabilities[:, tuned_classes.index(class_labels[-1])],
                    )
                )
            else:
                tuned_auc = float(
                    roc_auc_score(
                        y_test,
                        tuned_probabilities,
                        labels=tuned_classes,
                        multi_class="ovr",
                        average="weighted",
                    )
                )
            results.loc[winner_index, "Accuracy"] = accuracy_score(
                y_test, tuned_predictions
            )
            results.loc[winner_index, "AUC"] = tuned_auc
            results.loc[winner_index, "Precision"] = precision_score(
                y_test, tuned_predictions, average="weighted", zero_division=0
            )
            results.loc[winner_index, "Recall"] = recall_score(
                y_test, tuned_predictions, average="weighted", zero_division=0
            )
            results.loc[winner_index, "F1"] = f1_score(
                y_test, tuned_predictions, average="weighted", zero_division=0
            )
            results.loc[winner_index, "MCC"] = matthews_corrcoef(
                y_test, tuned_predictions
            )
            results.loc[winner_index, "Training Time (s)"] = tuning_time
            results.loc[winner_index, "Tuned parameters"] = repr(search.best_params_)
            predictions_by_model[winner_name] = tuned_predictions
            probabilities_by_model[winner_name] = tuned_probabilities
    results = results.sort_values("F1", ascending=False).reset_index(drop=True)
    results.insert(0, "Rank", np.arange(1, len(results) + 1))
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


def compare_regressors(
    dataset: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    tune_best: bool = False,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.Series]:
    """Compare CPU-friendly regressors on a held-out portion of a labeled table."""
    if target_column not in dataset.columns:
        raise ValueError("Choose a target column present in the dataset.")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("Test split must be between 10% and 40%.")

    data = dataset.copy()
    if data.empty or len(data) < 20:
        raise ValueError("Upload a dataset with at least 20 labeled rows.")
    if len(data) > MAX_ROWS:
        raise ValueError(
            f"Dataset has {len(data):,} rows; the interactive benchmark limit "
            f"is {MAX_ROWS:,} rows."
        )
    if data.columns.duplicated().any():
        raise ValueError("Dataset column names must be unique.")

    target = pd.to_numeric(data[target_column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    if target.isna().any():
        raise ValueError(
            f"Regression target '{target_column}' must contain only finite numeric values."
        )
    features = data.drop(columns=[target_column])
    numeric_columns = features.select_dtypes(include=np.number).columns
    features.loc[:, numeric_columns] = features.loc[:, numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    features = features.dropna(axis=1, how="all")
    if features.shape[1] == 0:
        raise ValueError("The dataset must have at least one usable input feature.")
    if features.shape[1] > MAX_FEATURES:
        raise ValueError(
            f"Dataset has {features.shape[1]} usable features; the interactive "
            f"benchmark limit is {MAX_FEATURES}."
        )
    high_cardinality = [
        str(column)
        for column in features.select_dtypes(exclude=np.number).columns
        if features[column].nunique(dropna=True) > MAX_CATEGORIES_PER_FEATURE
    ]
    if high_cardinality:
        raise ValueError(
            "Categorical features have too many distinct values for a responsive "
            "(limit 1,000 per column): "
            + ", ".join(high_cardinality[:5])
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    models: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "K-Nearest Neighbors": KNeighborsRegressor(
            n_neighbors=min(5, len(y_train))
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, n_jobs=1, random_state=random_state
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100, n_jobs=1, random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        "Neural Network (MLP)": MLPRegressor(
            hidden_layer_sizes=(32,),
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=12,
            tol=1e-2,
            alpha=0.01,
            random_state=random_state,
        ),
    }
    parameter_grids: dict[str, dict[str, list[Any]]] = {
        "Linear Regression": {"regressor__fit_intercept": [True, False]},
        "Ridge Regression": {"regressor__alpha": [0.1, 1.0, 10.0]},
        "Decision Tree": {
            "regressor__max_depth": [None, 8, 16],
            "regressor__min_samples_leaf": [1, 2],
        },
        "K-Nearest Neighbors": {"regressor__n_neighbors": [1, 3]},
        "Random Forest": {
            "regressor__max_depth": [None, 12],
            "regressor__min_samples_leaf": [1, 2],
        },
        "Extra Trees": {
            "regressor__max_depth": [None, 12],
            "regressor__min_samples_leaf": [1, 2],
        },
        "Gradient Boosting": {
            "regressor__learning_rate": [0.05, 0.1],
            "regressor__max_depth": [2, 3],
        },
        "Neural Network (MLP)": {
            "regressor__hidden_layer_sizes": [(32,), (64,)],
            "regressor__alpha": [0.01, 0.1],
        },
    }
    rows: list[dict[str, Any]] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    for model_name, estimator in models.items():
        pipeline = Pipeline(
            [("preprocessor", _preprocessor()), ("regressor", estimator)]
        )
        training_started = perf_counter()
        pipeline.fit(x_train, y_train)
        training_time = perf_counter() - training_started
        prediction_started = perf_counter()
        predictions = pipeline.predict(x_test)
        prediction_time = perf_counter() - prediction_started
        rows.append(
            {
                "Model": model_name,
                "R2": float(r2_score(y_test, predictions)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "MAE": float(mean_absolute_error(y_test, predictions)),
                "Training Time (s)": float(training_time),
                "Prediction Time (s)": float(prediction_time),
            }
        )
        predictions_by_model[model_name] = predictions

    results = pd.DataFrame(rows)
    results["Tuned parameters"] = ""
    if tune_best:
        winner_index = results["R2"].idxmax()
        winner_name = str(results.at[winner_index, "Model"])
        winner_pipeline = Pipeline(
            [
                ("preprocessor", _preprocessor()),
                ("regressor", models[winner_name]),
            ]
        )
        search = GridSearchCV(
            winner_pipeline,
            parameter_grids[winner_name],
            scoring="r2",
            cv=min(3, len(y_train)),
            n_jobs=1,
            refit=True,
        )
        tuning_started = perf_counter()
        search.fit(x_train, y_train)
        tuning_time = perf_counter() - tuning_started
        tuned_predictions = search.best_estimator_.predict(x_test)
        results.loc[winner_index, "R2"] = r2_score(y_test, tuned_predictions)
        results.loc[winner_index, "RMSE"] = np.sqrt(
            mean_squared_error(y_test, tuned_predictions)
        )
        results.loc[winner_index, "MAE"] = mean_absolute_error(
            y_test, tuned_predictions
        )
        results.loc[winner_index, "Training Time (s)"] = tuning_time
        results.loc[winner_index, "Tuned parameters"] = repr(search.best_params_)
        predictions_by_model[winner_name] = tuned_predictions

    results = results.sort_values(
        ["R2", "RMSE"], ascending=[False, True]
    ).reset_index(drop=True)
    results.insert(0, "Rank", np.arange(1, len(results) + 1))
    return results, predictions_by_model, y_test


def compare_clusters(
    dataset: pd.DataFrame,
    *,
    excluded_columns: tuple[str, ...] = (),
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    """Compare clustering models with sampled, label-free quality metrics."""
    if dataset.empty or len(dataset) < 10:
        raise ValueError("Upload a dataset with at least 10 rows for clustering.")
    if len(dataset) > MAX_ROWS:
        raise ValueError(
            f"Dataset has {len(dataset):,} rows; the interactive benchmark limit "
            f"is {MAX_ROWS:,} rows."
        )
    if dataset.columns.duplicated().any():
        raise ValueError("Dataset column names must be unique.")

    features = dataset.drop(
        columns=[column for column in excluded_columns if column in dataset.columns]
    ).copy()
    if features.shape[1] == 0:
        raise ValueError("Select at least one input feature for clustering.")
    if features.shape[1] > MAX_FEATURES:
        raise ValueError(
            f"Dataset has {features.shape[1]} input features; the interactive "
            f"benchmark limit is {MAX_FEATURES}."
        )
    numeric_columns = features.select_dtypes(include=np.number).columns
    features.loc[:, numeric_columns] = features.loc[:, numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    features = features.dropna(axis=1, how="all")
    if features.shape[1] == 0:
        raise ValueError("The dataset has no usable input features.")
    high_cardinality = [
        str(column)
        for column in features.select_dtypes(exclude=np.number).columns
        if features[column].nunique(dropna=True) > MAX_CATEGORIES_PER_FEATURE
    ]
    if high_cardinality:
        raise ValueError(
            "Categorical features have too many distinct values for clustering "
            "(limit 1,000 per column): "
            + ", ".join(high_cardinality[:5])
        )
    if not 2 <= n_clusters < len(features):
        raise ValueError("Choose a cluster count from 2 to one less than the row count.")

    transformed = _preprocessor().fit_transform(features)
    if sparse.issparse(transformed):
        if transformed.shape[1] > 50:
            components = min(50, transformed.shape[1] - 1, transformed.shape[0] - 1)
            transformed = TruncatedSVD(
                n_components=components, random_state=random_state
            ).fit_transform(transformed)
        else:
            transformed = transformed.toarray()
    matrix = np.asarray(transformed, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("The selected columns do not produce usable clustering features.")
    if not np.isfinite(matrix).all():
        raise ValueError("The selected columns contain values that cannot be clustered.")

    models: dict[str, Any] = {
        "K-Means": KMeans(
            n_clusters=n_clusters, n_init=5, max_iter=150, random_state=random_state
        ),
        "Mini-Batch K-Means": MiniBatchKMeans(
            n_clusters=n_clusters,
            n_init=5,
            max_iter=150,
            batch_size=1024,
            random_state=random_state,
        ),
        "Gaussian Mixture": GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            max_iter=100,
            random_state=random_state,
        ),
    }
    evaluation_size = min(MAX_CLUSTER_EVALUATION_ROWS, len(matrix))
    evaluation_indices = np.random.default_rng(random_state).choice(
        len(matrix), size=evaluation_size, replace=False
    )
    evaluation_matrix = matrix[evaluation_indices]
    rows: list[dict[str, Any]] = []
    labels_by_model: dict[str, np.ndarray] = {}
    for model_name, estimator in models.items():
        training_started = perf_counter()
        if model_name == "Gaussian Mixture":
            estimator.fit(matrix)
            labels = estimator.predict(matrix)
        else:
            labels = estimator.fit_predict(matrix)
        training_time = perf_counter() - training_started
        evaluation_labels = labels[evaluation_indices]
        unique_labels = np.unique(evaluation_labels)
        if 1 < len(unique_labels) < evaluation_size:
            silhouette = float(silhouette_score(evaluation_matrix, evaluation_labels))
            davies_bouldin = float(
                davies_bouldin_score(evaluation_matrix, evaluation_labels)
            )
            calinski_harabasz = float(
                calinski_harabasz_score(evaluation_matrix, evaluation_labels)
            )
        else:
            silhouette = davies_bouldin = calinski_harabasz = float("nan")
        rows.append(
            {
                "Model": model_name,
                "Silhouette": silhouette,
                "Davies-Bouldin": davies_bouldin,
                "Calinski-Harabasz": calinski_harabasz,
                "Clusters found": int(len(np.unique(labels))),
                "Training Time (s)": float(training_time),
            }
        )
        labels_by_model[model_name] = labels

    results = pd.DataFrame(rows).sort_values(
        ["Silhouette", "Davies-Bouldin"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)
    results.insert(0, "Rank", np.arange(1, len(results) + 1))
    return results, labels_by_model, pd.DataFrame(matrix, index=dataset.index)
