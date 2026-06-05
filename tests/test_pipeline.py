"""Unit tests for the CardioCare pipeline.

Run:  python -m unittest discover -s tests

Covers the four rubric-required tests plus extras:
  1. prediction shape matches input rows
  2. predict_proba in [0,1] and each row sums to ~1
  3. clinical input range validation works (e.g. chol in [0,600])
  4. determinism: same seed + same input -> identical output
  +  preprocessing pipeline fit/transform succeeds
  +  target binarization correctness
  +  sample_input.csv inference smoke test
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import (  # noqa: E402
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE, TARGET,
    binarize_target, build_model_pipeline, build_preprocessor, clean_frame,
    load_heart_data, split_X_y, validate_input_ranges)

DATA = ROOT / "data" / "heart.csv"
SAMPLE = ROOT / "data" / "sample_input.csv"


def _fit_pipeline():
    """Helper: train a small deterministic pipeline on the heart data."""
    df = binarize_target(clean_frame(load_heart_data(DATA)))
    X, y = split_X_y(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    pipe = build_model_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    pipe.fit(X_tr, y_tr)
    return pipe, X_te, y_te


class TestCardioCarePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipe, cls.X_test, cls.y_test = _fit_pipeline()

    # --- Required #1: prediction shape matches input rows ---------------
    def test_prediction_shape_matches_input(self):
        preds = self.pipe.predict(self.X_test)
        self.assertEqual(preds.shape[0], self.X_test.shape[0])
        self.assertEqual(preds.ndim, 1)

    # --- Required #2: predict_proba in [0,1] and rows sum to ~1 ---------
    def test_predict_proba_valid_distribution(self):
        proba = self.pipe.predict_proba(self.X_test)
        self.assertTrue(np.all(proba >= 0.0))
        self.assertTrue(np.all(proba <= 1.0))
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-6)

    # --- Required #3: clinical range validation works -------------------
    def test_clinical_range_validation(self):
        good = pd.DataFrame({"chol": [200, 300, 400],
                             "trestbps": [120, 130, 140],
                             "thalach": [150, 160, 170],
                             "age": [40, 50, 60],
                             "oldpeak": [1.0, 2.0, 0.5]})
        rep = validate_input_ranges(good)
        self.assertTrue(rep["valid"].all())

        bad = good.copy()
        bad.loc[0, "chol"] = 9999  # impossible cholesterol
        rep_bad = validate_input_ranges(bad)
        chol_row = rep_bad[rep_bad["feature"] == "chol"].iloc[0]
        self.assertFalse(bool(chol_row["valid"]))
        self.assertEqual(int(chol_row["n_out_of_range"]), 1)

    # --- Required #4: determinism (same seed -> same output) ------------
    def test_determinism_same_seed(self):
        pipe_a, X_te_a, _ = _fit_pipeline()
        pipe_b, X_te_b, _ = _fit_pipeline()
        np.testing.assert_array_equal(pipe_a.predict(X_te_a),
                                      pipe_b.predict(X_te_b))
        np.testing.assert_allclose(pipe_a.predict_proba(X_te_a),
                                   pipe_b.predict_proba(X_te_b), atol=1e-12)

    # --- Extra: preprocessor fit/transform succeeds ---------------------
    def test_preprocessor_fit_transform(self):
        df = binarize_target(clean_frame(load_heart_data(DATA)))
        X, _ = split_X_y(df)
        pre = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        Xt = pre.fit_transform(X)
        self.assertEqual(Xt.shape[0], X.shape[0])
        self.assertGreater(Xt.shape[1], len(NUMERIC_FEATURES))  # one-hot expand

    # --- Extra: target binarization correctness -------------------------
    def test_binarize_target(self):
        df = pd.DataFrame({TARGET: [0, 1, 2, 3, 4]})
        out = binarize_target(df)
        self.assertListEqual(out[TARGET].tolist(), [0, 1, 1, 1, 1])
        self.assertTrue(set(out[TARGET].unique()).issubset({0, 1}))

    # --- Extra: sample_input.csv inference smoke test -------------------
    def test_sample_input_inference(self):
        sample = pd.read_csv(SAMPLE)
        feats = [c for c in ALL_FEATURES if c in sample.columns]
        preds = self.pipe.predict(sample[feats])
        self.assertEqual(len(preds), len(sample))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
