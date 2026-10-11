"""Tests for ncr_model_eval.py: the shared scoring rules every model is compared with."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ncr_model_eval as E  # noqa: E402
import ncr_preprocessing as P  # noqa: E402

FIVE = P.POLICY_LABELS["five_class"]
THREE = P.POLICY_LABELS["three_class"]


class ScoringRules(unittest.TestCase):
    def test_macro_f1_ignores_classes_absent_from_the_block(self):
        # Block without Not Hazardous (code 0), like fold 4 and the test period: a perfect model must score 1.0.
        y = np.array([1, 2, 2, 3, 2, 1])
        self.assertAlmostEqual(E.score_block(y, y, FIVE)["macro_f1"], 1.0)

    def test_predicting_an_absent_class_still_counts_as_an_error(self):
        y = np.array([1, 2, 2, 3])
        pred = np.array([0, 2, 2, 3])        # a Caution day called Not Hazardous
        s = E.score_block(y, pred, FIVE)
        self.assertLess(s["macro_f1"], 1.0)
        self.assertAlmostEqual(s["accuracy"], 0.75)

    def test_focus_class_follows_the_policy(self):
        self.assertEqual(E.focus_code(FIVE), 3)
        self.assertEqual(E.focus_code(THREE), 2)

    def test_danger_recall_and_false_alarm_rate(self):
        y = np.array([3, 3, 3, 3, 2, 2, 2, 2])
        pred = np.array([3, 3, 3, 2, 3, 2, 2, 2])
        s = E.score_block(y, pred, FIVE)
        self.assertAlmostEqual(s["danger_recall"], 0.75)
        self.assertAlmostEqual(s["danger_false_alarm_rate"], 0.25)

    def test_partial_balance_weights_come_from_fit_labels(self):
        from sklearn.linear_model import LogisticRegression
        X = np.arange(40, dtype=float).reshape(-1, 1)
        y = np.array([0] * 30 + [1] * 10)
        full = E.PartialBalance(LogisticRegression(), power=1.0).fit(X, y)
        self.assertAlmostEqual(full.class_weight_[1], 40 / (2 * 10))
        none = E.PartialBalance(LogisticRegression(), power=0.0).fit(X, y)
        self.assertEqual(none.class_weight_, {0: 1.0, 1: 1.0})
        half = E.PartialBalance(LogisticRegression(), power=0.5).fit(X, y)
        self.assertAlmostEqual(half.class_weight_[1], 2.0 ** 0.5)
        self.assertEqual(list(half.classes_), [0, 1])

    def test_auc_skips_classes_without_positives(self):
        y = np.array([1, 1, 2, 2])
        proba = np.array([[0.0, 0.9, 0.1], [0.0, 0.8, 0.2], [0.0, 0.1, 0.9], [0.0, 0.3, 0.7]])
        self.assertAlmostEqual(E.macro_auc_present(y, proba, np.array([0, 1, 2])), 1.0)


if __name__ == "__main__":
    unittest.main()
