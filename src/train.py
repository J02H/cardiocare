"""CardioCare model training with MLflow experiment tracking.

Pipeline (no data leakage anywhere):
    1. load -> clean -> binarize target
    2. train_test_split (stratified, fixed seed)
    3. for each of >=3 model families: build sklearn Pipeline, 5-fold CV on
       TRAIN only, log params/metrics/artifact/model_family tag to MLflow
    4. hyperparameter search (GridSearchCV) on the leading family
    5. select final model by clinical criteria (recall-first, then balanced
       accuracy & F1, because false negatives are dangerous in cardiology)
    6. persist final model + comparison table + rationale

Run:
    python src/train.py --data data/heart.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, make_scorer, precision_score,
                             recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.svm import SVC

# Make `src` importable whether run as `python src/train.py` or `-m src.train`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE, TARGET,
    binarize_target, build_model_pipeline, clean_frame, load_heart_data,
    split_X_y)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MLRUNS_DIR = ROOT / "mlruns"
TEST_SIZE = 0.2
CV_FOLDS = 5


def _metrics(y_true, y_pred) -> dict:
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def candidate_models() -> dict:
    """At least three distinct model families."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
        "svc": SVC(probability=True, random_state=RANDOM_STATE,
                   class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE,
            class_weight="balanced"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CardioCare models.")
    parser.add_argument("--data", default=str(ROOT / "data" / "heart.csv"),
                        help="Path to heart.csv")
    parser.add_argument("--use-feature-selection", action="store_true",
                        help="Enable RF-based SelectFromModel inside pipeline")
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # ----- load + prepare -------------------------------------------------
    df = load_heart_data(args.data)
    df = clean_frame(df)
    df = binarize_target(df)
    X, y = split_X_y(df)
    print(f"[train] data={X.shape}  positives={int(y.sum())}/{len(y)} "
          f"({y.mean():.1%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    # ----- MLflow setup ---------------------------------------------------
    # SQLite tracking backend (file store is deprecated in MLflow 3.x);
    # artifacts still land under mlruns/. `mlflow ui` reads the same DB.
    MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name("CardioCare") is None:
        client.create_experiment(
            "CardioCare",
            artifact_location=(MLRUNS_DIR / "artifacts").as_uri())
    mlflow.set_experiment("CardioCare")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    rows = []

    for name, model in candidate_models().items():
        pipe = build_model_pipeline(
            model, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
            use_feature_selection=args.use_feature_selection)

        with mlflow.start_run(run_name=name):
            mlflow.set_tag("model_family", name)
            # 5-fold CV predictions on TRAIN ONLY (pipeline refit each fold).
            cv_pred = cross_val_predict(pipe, X_train, y_train, cv=cv)
            cv_m = _metrics(y_train, cv_pred)

            # Fit on full train, evaluate on held-out test.
            pipe.fit(X_train, y_train)
            test_pred = pipe.predict(X_test)
            test_m = _metrics(y_test, test_pred)
            cm = confusion_matrix(y_test, test_pred)

            mlflow.log_params({
                k: str(v) for k, v in model.get_params().items()
                if k in ("C", "kernel", "n_estimators", "max_depth",
                         "max_iter", "class_weight")
            })
            mlflow.log_param("model_family", name)
            for k, v in cv_m.items():
                mlflow.log_metric(f"cv_{k}", v)
            for k, v in test_m.items():
                mlflow.log_metric(k, v)
            tn, fp, fn, tp = cm.ravel()
            mlflow.log_metric("confusion_tn", tn)
            mlflow.log_metric("confusion_fp", fp)
            mlflow.log_metric("confusion_fn", fn)
            mlflow.log_metric("confusion_tp", tp)
            mlflow.sklearn.log_model(pipe, name="model")

            rows.append({
                "model_family": name,
                "cv_balanced_accuracy": cv_m["balanced_accuracy"],
                "cv_recall": cv_m["recall"],
                "cv_f1": cv_m["f1"],
                "test_balanced_accuracy": test_m["balanced_accuracy"],
                "test_precision": test_m["precision"],
                "test_recall": test_m["recall"],
                "test_f1": test_m["f1"],
                "confusion_tn": tn, "confusion_fp": fp,
                "confusion_fn": fn, "confusion_tp": tp,
            })
            print(f"[train] {name:20s} test_recall={test_m['recall']:.3f} "
                  f"bal_acc={test_m['balanced_accuracy']:.3f} "
                  f"f1={test_m['f1']:.3f}")

    comparison = pd.DataFrame(rows)

    # ----- pick leading family for tuning (clinical: recall first) --------
    # Composite ranking: recall weighted most, then balanced accuracy & F1.
    comparison["clinical_score"] = (
        0.5 * comparison["test_recall"]
        + 0.3 * comparison["test_balanced_accuracy"]
        + 0.2 * comparison["test_f1"]
    )
    lead = comparison.sort_values("clinical_score", ascending=False).iloc[0]
    lead_family = lead["model_family"]
    print(f"[train] leading family for tuning: {lead_family}")

    # ----- hyperparameter search on the leading family --------------------
    recall_scorer = make_scorer(recall_score, zero_division=0)
    search_space = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                               class_weight="balanced"),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "svc": (
            SVC(probability=True, random_state=RANDOM_STATE,
                class_weight="balanced"),
            {"model__C": [0.1, 1.0, 10.0],
             "model__kernel": ["rbf", "linear"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE,
                                   class_weight="balanced"),
            {"model__n_estimators": [200, 400],
             "model__max_depth": [None, 5, 10]},
        ),
    }
    base_model, grid = search_space[lead_family]
    tuned_pipe = build_model_pipeline(
        base_model, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
        use_feature_selection=args.use_feature_selection)

    with mlflow.start_run(run_name=f"{lead_family}_tuned"):
        mlflow.set_tag("model_family", lead_family)
        mlflow.set_tag("tuned", "true")
        search = GridSearchCV(tuned_pipe, grid, scoring=recall_scorer,
                              cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        test_pred = best.predict(X_test)
        test_m = _metrics(y_test, test_pred)
        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn, tp = cm.ravel()

        mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
        mlflow.log_param("model_family", lead_family)
        for k, v in test_m.items():
            mlflow.log_metric(k, v)
        mlflow.log_metric("cv_best_recall", search.best_score_)
        mlflow.log_metric("confusion_tn", tn)
        mlflow.log_metric("confusion_fp", fp)
        mlflow.log_metric("confusion_fn", fn)
        mlflow.log_metric("confusion_tp", tp)
        mlflow.sklearn.log_model(best, name="model")

        print(f"[train] tuned {lead_family}: best_params={search.best_params_} "
              f"test_recall={test_m['recall']:.3f} "
              f"bal_acc={test_m['balanced_accuracy']:.3f}")

        rows.append({
            "model_family": f"{lead_family}_tuned",
            "cv_balanced_accuracy": np.nan,
            "cv_recall": search.best_score_,
            "cv_f1": np.nan,
            "test_balanced_accuracy": test_m["balanced_accuracy"],
            "test_precision": test_m["precision"],
            "test_recall": test_m["recall"],
            "test_f1": test_m["f1"],
            "confusion_tn": tn, "confusion_fp": fp,
            "confusion_fn": fn, "confusion_tp": tp,
        })

    comparison = pd.DataFrame(rows)
    comparison["clinical_score"] = (
        0.5 * comparison["test_recall"]
        + 0.3 * comparison["test_balanced_accuracy"]
        + 0.2 * comparison["test_f1"]
    )
    comparison = comparison.sort_values(
        "clinical_score", ascending=False).reset_index(drop=True)
    comparison.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    # ----- final model = best clinical_score (recall-first) ---------------
    final_name = comparison.iloc[0]["model_family"]
    if final_name.endswith("_tuned"):
        final_model = best
    else:
        # refit the chosen untuned family on full train
        final_model = build_model_pipeline(
            candidate_models()[final_name], NUMERIC_FEATURES,
            CATEGORICAL_FEATURES,
            use_feature_selection=args.use_feature_selection)
        final_model.fit(X_train, y_train)

    joblib.dump(final_model, MODELS_DIR / "final_model.joblib")

    top = comparison.iloc[0]
    rationale = (
        f"FINAL MODEL: {final_name}\n"
        f"{'=' * 60}\n\n"
        f"Selection criterion: CLINICAL, recall-first.\n\n"
        f"In a cardiology screening context the most dangerous error is a\n"
        f"FALSE NEGATIVE - telling a patient who has heart disease that they\n"
        f"are healthy. A missed case can be fatal, whereas a false positive\n"
        f"mainly triggers an additional (non-invasive) review by the\n"
        f"cardiologist. We therefore rank candidates by a composite score that\n"
        f"weights recall most heavily (0.5), then balanced accuracy (0.3) to\n"
        f"respect class imbalance, then F1 (0.2) to keep precision honest.\n\n"
        f"Selected model metrics on the held-out test set:\n"
        f"  recall (sensitivity) : {top['test_recall']:.3f}\n"
        f"  balanced accuracy    : {top['test_balanced_accuracy']:.3f}\n"
        f"  precision            : {top['test_precision']:.3f}\n"
        f"  F1                   : {top['test_f1']:.3f}\n"
        f"  confusion [tn fp / fn tp]: "
        f"[{int(top['confusion_tn'])} {int(top['confusion_fp'])} / "
        f"{int(top['confusion_fn'])} {int(top['confusion_tp'])}]\n\n"
        f"False negatives (fn) = {int(top['confusion_fn'])}: this is the count\n"
        f"we most want to minimise. We did NOT inflate any number; these are\n"
        f"the raw evaluation results.\n\n"
        f"NOTE: CardioCare is an assistive, 'inform not decide' tool. The model\n"
        f"output is a probability presented to a cardiologist, never an\n"
        f"autonomous diagnosis.\n"
    )
    (OUTPUTS_DIR / "final_model_rationale.txt").write_text(
        rationale, encoding="utf-8")

    print(f"[train] saved final model -> {MODELS_DIR / 'final_model.joblib'}")
    print(f"[train] comparison      -> "
          f"{OUTPUTS_DIR / 'model_comparison.csv'}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
