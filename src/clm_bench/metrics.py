"""Calibration metrics for variable-size candidate sets.

Every item may have a different number of options (ARC has 3-5), so logits are a
list of 1-D arrays rather than one matrix.
"""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / t
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def probs_list(logits: list[np.ndarray], t: float = 1.0) -> list[np.ndarray]:
    return [softmax(z, t) for z in logits]


def nll(logits: list[np.ndarray], gold: np.ndarray, t: float = 1.0) -> float:
    out = 0.0
    for z, g in zip(logits, gold):
        z = np.asarray(z, dtype=np.float64) / t
        m = z.max()
        out += -(z[g] - m - np.log(np.exp(z - m).sum()))
    return out / len(gold)


def brier(probs: list[np.ndarray], gold: np.ndarray) -> float:
    """Multi-class Brier score: mean over items of sum_k (p_k - y_k)^2. Range [0, 2]."""
    tot = 0.0
    for p, g in zip(probs, gold):
        y = np.zeros_like(p)
        y[g] = 1.0
        tot += float(((p - y) ** 2).sum())
    return tot / len(gold)


def top1(probs: list[np.ndarray], gold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conf = np.array([p.max() for p in probs])
    correct = np.array([int(p.argmax() == g) for p, g in zip(probs, gold)])
    return conf, correct


def reliability(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        bins.append({"lo": float(lo), "hi": float(hi), "n": n,
                     "conf": float(conf[m].mean()) if n else None,
                     "acc": float(correct[m].mean()) if n else None})
    return bins


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> tuple[float, float]:
    """(expected calibration error, maximum calibration error), equal-width bins on top-1 confidence."""
    tot, worst, n = 0.0, 0.0, len(conf)
    for b in reliability(conf, correct, n_bins):
        if b["n"]:
            gap = abs(b["acc"] - b["conf"])
            tot += b["n"] / n * gap
            worst = max(worst, gap)
    return tot, worst


def fit_temperature(logits: list[np.ndarray], gold: np.ndarray) -> float:
    """Temperature T minimising NLL of softmax(z / T): coarse log-grid, then golden-section refine."""
    grid = np.exp(np.linspace(np.log(0.02), np.log(50.0), 241))
    losses = [nll(logits, gold, t) for t in grid]
    i = int(np.argmin(losses))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    gr = (np.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(60):
        if nll(logits, gold, c) < nll(logits, gold, d):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return float((a + b) / 2)


def summarize(logits: list[np.ndarray], gold: np.ndarray, t: float = 1.0, n_bins: int = 15) -> dict:
    p = probs_list(logits, t)
    conf, correct = top1(p, gold)
    e, mce = ece(conf, correct, n_bins)
    return {"n": int(len(gold)), "accuracy": float(correct.mean()), "mean_confidence": float(conf.mean()),
            "ece": e, "mce": mce, "nll": nll(logits, gold, t), "brier": brier(p, gold),
            "chance_accuracy": float(np.mean([1.0 / len(z) for z in logits]))}
