"""Shared cross-validation scoring for the three HeatCast NCR models (project Sections 5 and 6).

Every model is scored with the same folds (``prep.cv_splits``), the same metrics and the same rules, so the
comparison in Section 6 is fair. The test partition is never touched here.

* Macro F1 is computed over the classes that **occur in the block being scored**. Fold 4 has no Not Hazardous
  rows and the test period has none either; counting an absent class as F1 = 0 would punish every model for a
  class that cannot be scored.
* The "focus" class is Danger (``Danger or above`` under the three-class policy): its recall is the main
  supporting metric, because a missed Danger day is the costly error.
* ``persistence_predictions`` gives the naive baseline (tomorrow's class = today's class) on any block.

Typical use::

    import ncr_preprocessing as P, ncr_model_eval as E
    prep = P.prepare_dataset(P.DEFAULT_CSV)
    X, y = prep.X(prep.train, "hi_lags"), prep.y(prep.train)
    res = E.cross_validate_model(pipeline, X, y, prep.cv_splits, prep.class_labels)
    res.summary()            # mean and SD over the 5 folds
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)

import ncr_modeling_utils as U
import ncr_preprocessing as P

PRIMARY_METRIC = "macro_f1"
FOCUS_NAMES = ("Danger", "Danger or above")


def focus_code(class_labels) -> int:
    """Integer code of the Danger class under the active target policy."""
    for i, name in enumerate(class_labels):
        if name in FOCUS_NAMES:
            return i
    raise ValueError(f"no Danger class in {class_labels}")


def macro_auc_present(y_true, proba, model_classes) -> float:
    """One-vs-rest ROC-AUC averaged over the classes present in ``y_true`` (sklearn's multi-class AUC refuses
    a block that lacks one of the model's classes)."""
    y_true = np.asarray(y_true)
    aucs = []
    for col, c in enumerate(model_classes):
        pos = y_true == c
        if pos.any() and (~pos).any():
            aucs.append(roc_auc_score(pos, proba[:, col]))
    return float(np.mean(aucs)) if aucs else float("nan")


def score_block(y_true, y_pred, class_labels, proba=None, model_classes=None) -> dict:
    """All metrics for one validation block (or the final test set). Keys are flat so a list of them is a table."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = np.unique(y_true)
    focus = focus_code(class_labels)
    out = {
        "macro_f1": f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "qw_kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "macro_precision": precision_recall_fscore_support(y_true, y_pred, labels=present, average="macro",
                                                           zero_division=0)[0],
        "macro_recall": precision_recall_fscore_support(y_true, y_pred, labels=present, average="macro",
                                                        zero_division=0)[1],
    }
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[focus], zero_division=0)
    out.update({"danger_precision": p[0], "danger_recall": r[0], "danger_f1": f[0]})
    # share of non-Danger days that were flagged as Danger (the cost of a false alarm)
    neg = y_true != focus
    out["danger_false_alarm_rate"] = float((y_pred[neg] == focus).mean()) if neg.any() else float("nan")
    if proba is not None:
        out["roc_auc_ovr"] = macro_auc_present(y_true, proba, model_classes)
    return {k: float(v) for k, v in out.items()}


def per_class_table(y_true, y_pred, class_labels) -> pd.DataFrame:
    """Precision / recall / F1 / support for every class that occurs in ``y_true`` or ``y_pred``."""
    used = np.union1d(np.unique(y_true), np.unique(y_pred))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=used, zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s},
                        index=pd.Index([class_labels[k] for k in used], name="class"))


def confusion_frame(y_true, y_pred, class_labels) -> pd.DataFrame:
    used = np.union1d(np.unique(y_true), np.unique(y_pred))
    names = [class_labels[k] for k in used]
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=used),
                        index=pd.Index(names, name="actual"), columns=pd.Index(names, name="predicted"))


def persistence_predictions(frame: pd.DataFrame, target_policy: str) -> np.ndarray:
    """Naive baseline: tomorrow's class equals today's class (HeatLevelToday), encoded like the target."""
    return P.encode_target(frame["HeatLevelToday"], target_policy)


class PartialBalance(ClassifierMixin, BaseEstimator):
    """Class weights between none and fully balanced, learned from the labels passed to ``fit`` (so from each
    training fold only): ``weight_c = (n / (k * n_c)) ** power``. ``power=1`` equals ``class_weight="balanced"``,
    ``power=0`` is unweighted, ``0.5`` is the square root. Works with any classifier that takes ``class_weight``
    (Logistic Regression, Random Forest); for Gradient Boosting pass the same weights as ``sample_weight``."""

    def __init__(self, estimator, power: float = 1.0):
        self.estimator = estimator
        self.power = power

    def fit(self, X, y):
        w = {c: v ** self.power for c, v in P.class_weights(y).items()}
        self.class_weight_ = w
        self.estimator_ = clone(self.estimator).set_params(class_weight=w).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


@dataclass
class CVResult:
    folds: pd.DataFrame          # one row of metrics per fold
    oof_true: np.ndarray         # validation labels of all folds, concatenated
    oof_pred: np.ndarray         # out-of-fold predictions (each row predicted by a model that never saw it)
    oof_index: np.ndarray        # positions of those rows in the training frame
    fit_seconds: float

    def summary(self) -> pd.Series:
        m, s = self.folds.mean(), self.folds.std(ddof=1)
        return pd.concat([m.add_suffix("_mean"), s.add_suffix("_sd")]).sort_index()

    def mean(self, metric: str = PRIMARY_METRIC) -> float:
        return float(self.folds[metric].mean())

    def sd(self, metric: str = PRIMARY_METRIC) -> float:
        return float(self.folds[metric].std(ddof=1))


def _fit_score_fold(model, X, y, tr, va, class_labels):
    import time
    m = clone(model)
    t0 = time.perf_counter()
    m.fit(X.iloc[tr], y[tr])
    secs = time.perf_counter() - t0
    pred = m.predict(X.iloc[va])
    proba = m.predict_proba(X.iloc[va]) if hasattr(m, "predict_proba") else None
    return score_block(y[va], pred, class_labels, proba, getattr(m, "classes_", None)), pred, secs


def cross_validate_model(model, X: pd.DataFrame, y, folds, class_labels, n_jobs: int = -1) -> CVResult:
    """Fit a fresh clone of ``model`` (a Pipeline that includes ``P.make_preprocessor``) on every training fold and
    score it on that fold's validation block. Nothing outside ``folds`` is used."""
    y = np.asarray(y)
    runs = Parallel(n_jobs=n_jobs)(delayed(_fit_score_fold)(model, X, y, tr, va, class_labels) for tr, va in folds)
    table = pd.DataFrame([r[0] for r in runs], index=pd.Index(range(1, len(folds) + 1), name="fold"))
    idx = np.concatenate([va for _, va in folds])
    return CVResult(table, y[idx], np.concatenate([r[1] for r in runs]), idx, float(sum(r[2] for r in runs)))


def baseline_cv(frame: pd.DataFrame, folds, class_labels, target_policy: str) -> pd.DataFrame:
    """Persistence baseline scored on the same validation blocks as the models."""
    y = P.encode_target(frame[U.TARGET], target_policy)
    pred = persistence_predictions(frame, target_policy)
    rows = [score_block(y[va], pred[va], class_labels) for _, va in folds]
    return pd.DataFrame(rows, index=pd.Index(range(1, len(folds) + 1), name="fold"))
