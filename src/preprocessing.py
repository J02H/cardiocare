"""CardioCare preprocessing utilities.

All learnable transforms (imputers, scaler, encoder, feature selector) live
*inside* an sklearn Pipeline so they are only ever fit on the training fold.
This is the single source of truth for data loading, target binarisation,
clinical range validation, and pipeline construction. It is imported by the
notebook, train.py, inference.py, monitor.py and the tests.

No leakage by construction: nothing here calls .fit on raw data; fitting is
the caller's job and always happens after train_test_split.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# Continuous features that get median imputation + StandardScaler.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Discrete / coded features that get most_frequent imputation + OneHotEncoder.
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"

# Clinically reasonable ranges used by validate_input_ranges().
# Values outside these bounds are flagged as invalid (likely data-entry error).
CLINICAL_RANGES: dict[str, tuple[float, float]] = {
    "age": (18, 120),
    "trestbps": (0, 250),   # resting blood pressure (mm Hg)
    "chol": (0, 600),       # serum cholesterol (mg/dl)
    "thalach": (0, 250),    # max heart rate achieved
    "oldpeak": (0, 10),     # ST depression
}


# ---------------------------------------------------------------------------
# Loading & target handling
# ---------------------------------------------------------------------------
def load_heart_data(path: str | Path) -> pd.DataFrame:
    """Load the heart dataset from CSV with a friendly error if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Run `python data/download_data.py` first, or place a UCI/Kaggle "
            "heart.csv at data/heart.csv. See README.md for instructions."
        )
    df = pd.read_csv(path)
    if df.shape[1] == 0:
        raise ValueError(f"{path} appears to be empty.")
    return df


def binarize_target(df: pd.DataFrame, target_col: str = TARGET) -> pd.DataFrame:
    """Convert a (possibly multiclass) target to binary: 0 = no disease,
    any value > 0 -> 1 = heart disease present. Idempotent for binary input."""
    if target_col not in df.columns:
        raise KeyError(f"target column '{target_col}' not in dataframe.")
    out = df.copy()
    out[target_col] = (pd.to_numeric(out[target_col], errors="coerce")
                       .fillna(0) > 0).astype(int)
    return out


def clean_frame(df: pd.DataFrame, target_col: str = TARGET) -> pd.DataFrame:
    """Drop fully-empty columns and exact duplicate rows.

    These are dataset-level hygiene steps (not learnable transforms), so doing
    them here does not cause leakage: no statistics are learned from the data.
    """
    out = df.copy()
    # Drop columns that are entirely NaN (empty columns).
    out = out.dropna(axis=1, how="all")
    # Drop exact duplicate rows.
    out = out.drop_duplicates().reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_input_ranges(
    df: pd.DataFrame, ranges: dict[str, tuple[float, float]] | None = None
) -> pd.DataFrame:
    """Return a per-feature validation report for clinically-bounded columns.

    The result has one row per validated feature and columns:
        feature, lo, hi, n_out_of_range, valid
    `valid` is True when no in-range-checkable value falls outside [lo, hi]
    (NaNs are ignored - they are handled later by imputation).
    """
    ranges = ranges or CLINICAL_RANGES
    rows = []
    for feat, (lo, hi) in ranges.items():
        if feat not in df.columns:
            continue
        col = pd.to_numeric(df[feat], errors="coerce")
        mask_oob = (col < lo) | (col > hi)
        n_oob = int(mask_oob.sum())
        rows.append({
            "feature": feat, "lo": lo, "hi": hi,
            "n_out_of_range": n_oob, "valid": n_oob == 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------
def build_preprocessor(
    numeric_features: Iterable[str] = NUMERIC_FEATURES,
    categorical_features: Iterable[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """ColumnTransformer: median-impute + scale numerics; mode-impute +
    one-hot encode categoricals. All transforms are unfitted - they fit only
    when the enclosing pipeline is fit on the training split (no leakage)."""
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipe, list(numeric_features)),
        ("cat", categorical_pipe, list(categorical_features)),
    ], remainder="drop")


def build_model_pipeline(
    model,
    numeric_features: Iterable[str] = NUMERIC_FEATURES,
    categorical_features: Iterable[str] = CATEGORICAL_FEATURES,
    use_feature_selection: bool = False,
) -> Pipeline:
    """Assemble preprocessor -> (optional) feature selection -> estimator.

    Feature selection uses a RandomForest-based SelectFromModel and lives
    *inside* the pipeline, so it is fit only on the training fold."""
    steps = [("preprocessor", build_preprocessor(
        numeric_features, categorical_features))]
    if use_feature_selection:
        steps.append((
            "feature_selection",
            SelectFromModel(
                RandomForestClassifier(
                    n_estimators=200, random_state=RANDOM_STATE),
                threshold="median",
            ),
        ))
    steps.append(("model", model))
    return Pipeline(steps=steps)


def split_X_y(df: pd.DataFrame, target_col: str = TARGET):
    """Split a (binarised) frame into feature matrix X and target vector y,
    keeping only the known feature columns that are present."""
    feats = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feats].copy()
    y = df[target_col].copy() if target_col in df.columns else None
    return X, y
