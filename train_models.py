"""
Paddy productivity classification - training + evaluation script.

The raw Kaggle "Paddy Dataset" has a continuous target (Paddy yield in Kg).
For this classification assignment the target is converted to a 3-class label:

    yield per hectare = Paddy yield (Kg) / Hectares
    Low / Medium / High  = tertiles (33rd / 67th percentile) of yield per hectare

Usage:
    python train_models.py --data paddydataset.csv

Outputs (written next to this script):
    model/<name>.joblib      one fitted Pipeline (preprocessing + classifier) per model
    model/metadata.joblib    feature list, class order, bin edges, etc. (used by the app)
    test_data.csv            held-out test split (features + Yield_Class)
    metrics_results.csv      6 metrics x 5 models on the test split
"""
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 11
TARGET = "Yield_Class"
CLASS_ORDER = ["Low", "Medium", "High"]
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()            # "Hectares " -> "Hectares"
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)  # exact duplicates would leak across the split
    print(f"Rows: {n_raw} raw -> {len(df)} after dropping {n_raw - len(df)} duplicates")

    yield_per_ha = df["Paddy yield(in Kg)"] / df["Hectares"]
    low_cut, high_cut = yield_per_ha.quantile([1 / 3, 2 / 3]).tolist()
    df[TARGET] = pd.cut(yield_per_ha, [-np.inf, low_cut, high_cut, np.inf],
                        labels=CLASS_ORDER).astype(str)
    df = df.drop(columns=["Paddy yield(in Kg)"])
    print("Class balance:\n", df[TARGET].value_counts().to_string())
    return df, (low_cut, high_cut)


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])


def candidate_models():
    """(estimator, param grid) for each of the five required classifiers."""
    return {
        "Logistic Regression": (LogisticRegression(max_iter=3000),
                                {"clf__C": [0.01, 0.1, 1, 10]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                          {"clf__max_depth": [3, 5, 8, 12],
                           "clf__min_samples_leaf": [1, 5, 10]}),
        "KNN": (KNeighborsClassifier(),
                {"clf__n_neighbors": [5, 11, 21, 31],
                 "clf__weights": ["uniform", "distance"]}),
        "Naive Bayes": (GaussianNB(),
                        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}),
        "Support Vector Machine": (SVC(probability=True, random_state=SEED),
                                   {"clf__C": [0.5, 1, 5, 10],
                                    "clf__gamma": ["scale", 0.001, 0.01]}),
    }


def evaluate(pipe, X, y):
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba, multi_class="ovr", average="macro",
                             labels=list(pipe.classes_)),
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted", zero_division=0),
        "F1": f1_score(y, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


def main(data_path):
    MODEL_DIR.mkdir(exist_ok=True)
    df, (low_cut, high_cut) = load_and_prepare(data_path)

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    print(f"{X.shape[1]} features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = {}
    for name, (estimator, grid) in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor(numeric_cols, categorical_cols)),
                         ("clf", estimator)])
        search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1).fit(X_train, y_train)
        best = search.best_estimator_
        rows[name] = evaluate(best, X_test, y_test)
        print(f"{name:24s} best={search.best_params_}  cv_f1macro={search.best_score_:.4f}  "
              f"test_acc={rows[name]['Accuracy']:.4f}")
        file_stem = name.lower().replace(" ", "_")
        joblib.dump(best, MODEL_DIR / f"{file_stem}.joblib", compress=3)

    joblib.dump({
        "target_col": TARGET,
        "class_order": CLASS_ORDER,
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "bin_edges_kg_per_ha": (low_cut, high_cut),
        "model_files": {n: n.lower().replace(" ", "_") + ".joblib" for n in rows},
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
    }, MODEL_DIR / "metadata.joblib")

    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    test_df.to_csv(BASE_DIR / "test_data.csv", index=False)

    results = pd.DataFrame(rows).T.round(4)
    results.index.name = "ML Model Name"
    results.to_csv(BASE_DIR / "metrics_results.csv")
    print("\n", results.to_string())
    print(f"\nSaved {len(rows)} models, metadata, test_data.csv ({len(test_df)} rows), metrics_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="paddydataset.csv", help="path to raw Kaggle CSV")
    main(parser.parse_args().data)
