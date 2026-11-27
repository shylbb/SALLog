import numpy as np


def binary_metrics(labels, predictions):
    y, p = np.asarray(labels), np.asarray(predictions)
    if y.ndim != 1 or p.shape != y.shape or not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("Metrics need aligned binary labels and predictions")
    tp, tn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())
    fp, fn = int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
    return {"n": len(y), "TP": tp, "TN": tn, "FP": fp, "FN": fn, "errors": fp + fn,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def best_threshold(probabilities, labels):
    p, y = np.asarray(probabilities, dtype=np.float32), np.asarray(labels)
    if len(p) == 0 or not np.any(y == 1):
        return .5
    best, score = .5, -1.
    for threshold in np.unique(np.quantile(p, np.linspace(0, 1, 501))):
        f1 = binary_metrics(y, (p >= threshold).astype(int))["f1"]
        if f1 > score:
            best, score = float(threshold), f1
    return best

