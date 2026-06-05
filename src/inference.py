"""CardioCare batch inference.

Loads the trained sklearn pipeline, runs predictions on a CSV of patient
records, and writes predictions + probabilities to disk. All steps are logged.

CLI:
    python src/inference.py \
        --input data/sample_input.csv \
        --model models/final_model.joblib \
        --output outputs/predictions.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import (ALL_FEATURES, TARGET, binarize_target,  # noqa: E402
                           validate_input_ranges)

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"
MODEL_VERSION = "1.0"

LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "inference.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("cardiocare.inference")


def load_model(model_path: str | Path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Train a model first:  python src/train.py --data data/heart.csv"
        )
    return joblib.load(model_path)


def predict_frame(model, df: pd.DataFrame) -> pd.DataFrame:
    """Run prediction on a raw input dataframe and return a result frame."""
    # Keep only known feature columns that exist; the pipeline imputes the rest.
    feats = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feats].copy()

    # Validate clinical ranges (report only; does not block inference).
    report = validate_input_ranges(X)
    bad = report[~report["valid"]]
    if not bad.empty:
        logger.warning("Out-of-range values detected in: %s",
                       ", ".join(bad["feature"].tolist()))

    preds = model.predict(X)
    proba = model.predict_proba(X)
    # Probability of the positive class (heart disease = 1).
    classes = list(model.classes_)
    pos_idx = classes.index(1) if 1 in classes else 1
    p_disease = proba[:, pos_idx]

    out = df.copy()
    out["prediction"] = preds.astype(int)
    out["prob_no_disease"] = proba[:, classes.index(0)] \
        if 0 in classes else (1 - p_disease)
    out["prob_disease"] = p_disease
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="CardioCare inference.")
    parser.add_argument("--input", default=str(ROOT / "data" /
                                               "sample_input.csv"))
    parser.add_argument("--model", default=str(ROOT / "models" /
                                               "final_model.joblib"))
    parser.add_argument("--output", default=str(ROOT / "outputs" /
                                                "predictions.csv"))
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {in_path}. "
            "Try data/sample_input.csv (shipped with the repo).")

    df = pd.read_csv(in_path)
    model = load_model(args.model)

    logger.info("model_version=%s | input=%s | input_shape=%s",
                MODEL_VERSION, in_path.name, df.shape)

    result = predict_frame(model, df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    n_pos = int(result["prediction"].sum())
    logger.info("predictions: %d rows | disease=%d | no_disease=%d | -> %s",
                len(result), n_pos, len(result) - n_pos, out_path)
    logger.info("REMINDER: CardioCare informs, it does not decide. Outputs are "
                "decision-support for a clinician, not a diagnosis.")
    print(result[["prediction", "prob_no_disease", "prob_disease"]]
          .to_string(index=True))


if __name__ == "__main__":
    main()
