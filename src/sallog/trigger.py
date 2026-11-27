"""The single published calling rule: U * abs(p - q50)**2."""
import hashlib
import math
import numpy as np


def normalized(values, dtype=np.float32):
    values = np.asarray(values, dtype=dtype)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite feature matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero vectors cannot define cosine similarity")
    return values / norms


def balanced_indices(labels, size=5000, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    per_class = min(size // 2, *(len(g) for g in groups))
    if per_class < 1:
        raise ValueError("A spatial reference bank needs both training classes")
    parts = [np.sort(rng.choice(g, per_class, replace=False)) for g in groups]
    result = np.concatenate(parts)
    rng.shuffle(result)
    return result


def risk_scores(probabilities, queries, reference_features, reference_labels, k=50, exponent=2., batch_size=256):
    p = np.asarray(probabilities, dtype=np.float32)
    q, r = normalized(queries), normalized(reference_features)
    labels = np.asarray(reference_labels)
    if (p.shape != (len(q),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or labels.shape != (len(r),) or not np.isin(labels, [0, 1]).all() or k < 1 or exponent <= 0):
        raise ValueError("Invalid risk inputs")
    effective_k = min(int(k), len(r))
    support = np.empty(len(q), dtype=np.float32)
    class_similarities = np.empty((len(q), 2), dtype=np.float64)
    # Gate statistics are computed independently of the global Top-2 reading examples.
    gate_q, gate_r = normalized(queries, np.float64), normalized(reference_features, np.float64)
    for start in range(0, len(q), int(batch_size)):
        stop = min(start + int(batch_size), len(q))
        scores = q[start:stop] @ r.T
        near = np.argpartition(-scores, effective_k - 1, axis=1)[:, :effective_k]
        support[start:stop] = labels[near].mean(axis=1)
        gate_scores = gate_q[start:stop] @ gate_r.T
        for label in (0, 1):
            if not np.any(labels == label):
                raise ValueError("Both classes must exist in the spatial bank")
            class_similarities[start:stop, label] = np.clip(gate_scores[:, labels == label].max(axis=1), -1, 1)
    u = 1 - 2 * np.abs(p - .5)
    return (u * np.abs(p - support) ** exponent).astype(np.float32), support, class_similarities, effective_k


def rank(scores, sample_ids, seed=20250301):
    prefix = f"trigger-random-v1:{int(seed)}:".encode()
    ties = np.asarray([int.from_bytes(hashlib.blake2b(prefix + str(s).encode(), digest_size=8).digest(), "big") / 2**64
                       for s in sample_ids], dtype=np.float32)
    values = np.asarray(scores)
    if len(values) != len(sample_ids) or len(set(sample_ids)) != len(sample_ids) or not np.isfinite(values).all():
        raise ValueError("Invalid ranking identities or scores")
    return np.lexsort((ties.astype(np.float64), -values.astype(np.float64)))


def budget_count(n, fraction):
    if n < 1 or not 0 < fraction <= 1:
        raise ValueError("Positive population and budget in (0,1] required")
    return min(n, max(1, math.floor(n * fraction + .5)))

