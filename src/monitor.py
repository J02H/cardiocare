"""CardioCare monitoring & data-drift detection.

What it does:
    1. Logs an instrumented inference pass (timestamp, model_version,
       input_shape, predictions, actual labels) to logs/inference.log.
    2. Builds a drifted copy of the test set by shifting continuous features
       (chol mean +30; trestbps +15; oldpeak +0.5).
    3. Runs scipy.stats.ks_2samp per continuous feature (reference vs drifted)
       and flags features with p < 0.05.
    4. Compares balanced accuracy on the clean test set vs the drifted set so
       input drift can be linked to performance decay.
    5. Writes outputs/drift_report.csv, outputs/drift_summary.txt and a
       time-series plot outputs/performance_over_time.png.

CLI:
    python src/monitor.py --data data/heart.csv --model models/final_model.joblib
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import (NUMERIC_FEATURES, RANDOM_STATE, TARGET,  # noqa: E402
                           binarize_target, clean_frame, load_heart_data,
                           split_X_y)

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"
OUTPUTS_DIR = ROOT / "outputs"
MODEL_VERSION = "1.0"
TEST_SIZE = 0.2
ALPHA = 0.05

# Artificial shifts applied to continuous features to simulate drift.
DRIFT_SHIFTS = {"chol": 30.0, "trestbps": 15.0, "oldpeak": 0.5}

LOGS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "inference.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("cardiocare.monitor")


def instrumented_predict(model, X, y_true, tag: str):
    """Run prediction and log monitoring telemetry."""
    preds = model.predict(X)
    bal_acc = balanced_accuracy_score(y_true, preds)
    logger.info(
        "MONITOR | ts=%s | model_version=%s | tag=%s | input_shape=%s | "
        "n_pred_positive=%d | balanced_accuracy=%.4f",
        datetime.utcnow().isoformat(timespec="seconds"), MODEL_VERSION, tag,
        tuple(X.shape), int(preds.sum()), bal_acc,
    )
    return preds, bal_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="CardioCare drift monitor.")
    parser.add_argument("--data", default=str(ROOT / "data" / "heart.csv"))
    parser.add_argument("--model", default=str(ROOT / "models" /
                                               "final_model.joblib"))
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run train.py first.")
    model = joblib.load(model_path)

    df = load_heart_data(args.data)
    df = clean_frame(df)
    df = binarize_target(df)
    X, y = split_X_y(df)

    # Recreate the SAME split train.py used so the reference distribution and
    # test set are consistent.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    # ----- clean (reference) inference -----------------------------------
    _, bal_acc_clean = instrumented_predict(model, X_test, y_test, "clean_test")

    # ----- build drifted copy --------------------------------------------
    X_drift = X_test.copy()
    for feat, shift in DRIFT_SHIFTS.items():
        if feat in X_drift.columns:
            X_drift[feat] = pd.to_numeric(X_drift[feat], errors="coerce") + shift
    # Also inflate variance of chol to make the shift more realistic.
    if "chol" in X_drift.columns:
        mu = X_drift["chol"].mean()
        X_drift["chol"] = mu + (X_drift["chol"] - mu) * 1.3

    _, bal_acc_drift = instrumented_predict(model, X_drift, y_test,
                                            "drifted_test")

    # ----- KS test per continuous feature (train ref vs drifted) ---------
    rows = []
    for feat in NUMERIC_FEATURES:
        if feat not in X_train.columns:
            continue
        ref = pd.to_numeric(X_train[feat], errors="coerce").dropna()
        cur = pd.to_numeric(X_drift[feat], errors="coerce").dropna()
        stat, pval = ks_2samp(ref, cur)
        rows.append({
            "feature": feat,
            "ks_statistic": stat,
            "p_value": pval,
            "drift_flag": bool(pval < ALPHA),
            "shift_applied": DRIFT_SHIFTS.get(feat, 0.0),
        })
    drift_report = pd.DataFrame(rows)
    drift_report.to_csv(OUTPUTS_DIR / "drift_report.csv", index=False)

    flagged = drift_report[drift_report["drift_flag"]]["feature"].tolist()
    delta = bal_acc_clean - bal_acc_drift

    logger.info("KS drift flagged features: %s", flagged or "none")
    logger.info("balanced_accuracy clean=%.4f drifted=%.4f delta=%.4f",
                bal_acc_clean, bal_acc_drift, delta)

    # ----- summary --------------------------------------------------------
    summary = (
        "CardioCare Drift Monitoring Summary\n"
        "===================================\n\n"
        f"Model version          : {MODEL_VERSION}\n"
        f"Reference (train) rows  : {len(X_train)}\n"
        f"Test rows               : {len(X_test)}\n"
        f"Significance level alpha: {ALPHA}\n\n"
        f"Artificial drift applied: {DRIFT_SHIFTS}\n"
        "  (chol variance also inflated x1.3 to mimic a sensor/recalibration\n"
        "   change in a downstream lab.)\n\n"
        "KS test (Kolmogorov-Smirnov, train vs drifted) per continuous "
        "feature:\n"
    )
    for _, r in drift_report.iterrows():
        summary += (f"  - {r['feature']:9s}: KS={r['ks_statistic']:.3f} "
                    f"p={r['p_value']:.3e} "
                    f"{'DRIFT' if r['drift_flag'] else 'ok'}\n")
    summary += (
        f"\nFlagged (p < {ALPHA}): {flagged or 'none'}\n\n"
        f"Balanced accuracy (clean test) : {bal_acc_clean:.4f}\n"
        f"Balanced accuracy (drifted set): {bal_acc_drift:.4f}\n"
        f"Performance decay (delta)      : {delta:.4f}\n\n"
        "Interpretation: KS flags an input-distribution shift on the shifted\n"
        "features; the accompanying drop in balanced accuracy demonstrates the\n"
        "link between input drift and performance decay. In production this\n"
        "would trigger the retraining / human-review policy described in the\n"
        "report (drift-triggered retrain + scheduled retrain, with a clinician\n"
        "in the loop to prevent a runaway feedback loop).\n"
    )
    (OUTPUTS_DIR / "drift_summary.txt").write_text(summary, encoding="utf-8")

    # ----- performance-over-time plot (simulated weekly timestamps) -------
    # Simulate weekly monitoring where drift grows over time.
    weeks = 8
    base = datetime(2026, 1, 1)
    timestamps = [base + timedelta(weeks=w) for w in range(weeks)]
    accs = []
    rng = np.random.default_rng(RANDOM_STATE)
    for w in range(weeks):
        frac = w / (weeks - 1)  # 0 -> 1 growing drift
        Xw = X_test.copy()
        for feat, shift in DRIFT_SHIFTS.items():
            if feat in Xw.columns:
                Xw[feat] = pd.to_numeric(Xw[feat], errors="coerce") + shift * frac
        preds = model.predict(Xw)
        accs.append(balanced_accuracy_score(y_test, preds))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(timestamps, accs, marker="o", color="#c0392b", linewidth=2)
    ax.axhline(bal_acc_clean, ls="--", color="#2980b9",
               label=f"baseline = {bal_acc_clean:.3f}")
    ax.set_title("CardioCare: balanced accuracy over time under growing drift")
    ax.set_xlabel("Monitoring week")
    ax.set_ylabel("Balanced accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "performance_over_time.png", dpi=120)
    plt.close(fig)

    print(summary)
    print(f"[monitor] drift_report -> {OUTPUTS_DIR / 'drift_report.csv'}")
    print(f"[monitor] plot         -> "
          f"{OUTPUTS_DIR / 'performance_over_time.png'}")


if __name__ == "__main__":
    main()
