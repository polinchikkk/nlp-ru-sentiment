"""Macro-F1 без neutral."""

from sklearn.metrics import f1_score
import numpy as np

__all__ = ["macro_f1"]


def macro_f1(y_true, y_pred, ignore_neutral: bool = True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if ignore_neutral:  # neutral = 1
        mask = y_true != 1
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        labels = [0, 2]
    else:
        labels = [0, 1, 2]

    return f1_score(y_true, y_pred, labels=labels, average="macro")
