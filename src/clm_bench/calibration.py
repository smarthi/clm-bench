"""Calibration + distractor-sensitivity benchmark for CLM-8B.

For each dataset:
  1. Encode every state and candidate once (disk-cached).
  2. Score with CLM heads (and the no-head ``raw`` ablation).
  3. Report accuracy, ECE, MCE, NLL, Brier at the checkpoint's own temperature.
  4. Fit post-hoc temperature scaling on a random half, evaluate on the other half,
     repeated over several random splits (mean ± sd).
  5. (small-K datasets) Append neutral distractor options and measure how much the
     absolute probabilities move. Because CLM is a softmax over independently scored
     candidates, relative odds between the original options cannot change (IIA);
     what can change is every absolute number a threshold policy reads.
"""
from __future__ import annotations

import random

import numpy as np

from . import metrics as M
from .data import Item

DISTRACTORS = ["None of the above.", "I don't know."]


def item_logits(items: list[Item], s_emb: np.ndarray, c_embs: list[np.ndarray], scorer, raw: bool):
    out = []
    for i, it in enumerate(items):
        s = s_emb[i:i + 1]
        z = scorer.raw_logits(s, c_embs[i]) if raw else scorer.logits(s, c_embs[i])
        out.append(np.asarray(z).reshape(-1))
    return out


def temperature_study(logits: list[np.ndarray], gold: np.ndarray, splits: int, seed: int) -> dict:
    rows = []
    n = len(gold)
    for k in range(splits):
        idx = list(range(n))
        random.Random(seed + k).shuffle(idx)
        cal, test = idx[: n // 2], idx[n // 2:]
        zc, gc = [logits[i] for i in cal], gold[cal]
        zt, gt = [logits[i] for i in test], gold[test]
        t = M.fit_temperature(zc, gc)
        before, after = M.summarize(zt, gt, 1.0), M.summarize(zt, gt, t)
        rows.append({"T": t, "before": before, "after": after})

    def agg(which: str, key: str):
        v = np.array([r[which][key] for r in rows])
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}

    keys = ["accuracy", "mean_confidence", "ece", "mce", "nll", "brier"]
    t = np.array([r["T"] for r in rows])
    return {"splits": splits, "T": {"mean": float(t.mean()), "sd": float(t.std(ddof=1)) if len(t) > 1 else 0.0},
            "test_before": {k: agg("before", k) for k in keys},
            "test_after": {k: agg("after", k) for k in keys}}


def distractor_study(items: list[Item], s_emb, c_embs, d_embs_per_item, scorer, thresholds=(0.5, 0.9)) -> dict:
    """Compare probabilities with the original options vs. original + distractors."""
    p_gold_b, p_gold_a, p_top_b, p_top_a = [], [], [], []
    to_distractor, acc_b, acc_a, iia_err = 0, [], [], []
    over = {t: [0, 0] for t in thresholds}
    crossed_down = {t: 0 for t in thresholds}
    for i, it in enumerate(items):
        s = s_emb[i:i + 1]
        k = len(it.candidates)
        zb = scorer.logits(s, c_embs[i]).reshape(-1)
        za = scorer.logits(s, np.concatenate([c_embs[i], d_embs_per_item[i]])).reshape(-1)
        pb, pa = M.softmax(zb), M.softmax(za)
        p_gold_b.append(pb[it.gold]); p_gold_a.append(pa[it.gold])
        p_top_b.append(pb.max()); p_top_a.append(pa.max())
        acc_b.append(int(pb.argmax() == it.gold)); acc_a.append(int(pa.argmax() == it.gold))
        if pa.argmax() >= k:
            to_distractor += 1
        renorm = pa[:k] / pa[:k].sum()
        iia_err.append(float(np.abs(renorm - pb).max()))
        for t in thresholds:
            over[t][0] += int(pb.max() >= t)
            over[t][1] += int(pa.max() >= t)
            crossed_down[t] += int(pb.max() >= t and pa[:k].max() < t)
    n = len(items)
    return {
        "n": n,
        "distractors": DISTRACTORS + ["<correct answer text from a different question>"],
        "mean_p_gold": {"original": float(np.mean(p_gold_b)), "with_distractors": float(np.mean(p_gold_a))},
        "mean_p_top": {"original": float(np.mean(p_top_b)), "with_distractors": float(np.mean(p_top_a))},
        "mean_abs_delta_p_gold": float(np.mean(np.abs(np.array(p_gold_a) - np.array(p_gold_b)))),
        "accuracy": {"original": float(np.mean(acc_b)), "with_distractors": float(np.mean(acc_a))},
        "top1_became_distractor_rate": to_distractor / n,
        "share_above_threshold": {str(t): {"original": over[t][0] / n, "with_distractors": over[t][1] / n}
                                  for t in thresholds},
        "fell_below_threshold_rate": {str(t): crossed_down[t] / n for t in thresholds},
        "iia_check_max_abs_error": float(np.max(iia_err)),
    }


def run_dataset(name: str, items: list[Item], embedder, scorer, cache, splits: int, seed: int,
                distractors: bool, n_bins: int) -> tuple[dict, dict]:
    s_emb = cache.get_many(embedder, [it.state for it in items], label=f"{name} states")
    all_c = [c for it in items for c in it.candidates]
    c_all = cache.get_many(embedder, all_c, label=f"{name} candidates")
    c_embs, k = [], 0
    for it in items:
        c_embs.append(c_all[k:k + len(it.candidates)])
        k += len(it.candidates)
    gold = np.array([it.gold for it in items])

    res: dict = {"dataset": name, "n": len(items),
                 "k": {"min": int(min(len(it.candidates) for it in items)),
                       "max": int(max(len(it.candidates) for it in items))}}
    curves = {}
    for model, raw in (("clm", False), ("raw_qwen3_no_heads", True)):
        z = item_logits(items, s_emb, c_embs, scorer, raw)
        res[model] = {"at_T1": M.summarize(z, gold, 1.0, n_bins),
                      "temperature_scaling": temperature_study(z, gold, splits, seed)}
        if not raw:
            t_all = M.fit_temperature(z, gold)
            for label, t in (("T=1", 1.0), (f"T={t_all:.2f} (fit on all)", t_all)):
                conf, corr = M.top1(M.probs_list(z, t), gold)
                curves[label] = M.reliability(conf, corr, n_bins)

    if distractors:
        rng = random.Random(seed)
        d_texts = []
        for i, it in enumerate(items):
            j = rng.randrange(len(items) - 1)
            j = j + 1 if j >= i else j
            other = items[j].candidates[items[j].gold]
            d_texts.append(DISTRACTORS + [other])
        d_flat = cache.get_many(embedder, [t for ds in d_texts for t in ds], label=f"{name} distractors")
        per = len(DISTRACTORS) + 1
        d_embs = [d_flat[i * per:(i + 1) * per] for i in range(len(items))]
        res["distractor_sensitivity"] = distractor_study(items, s_emb, c_embs, d_embs, scorer)
    return res, curves


def plot_reliability(curves: dict, title: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="perfect calibration")
    for (label, bins), color in zip(curves.items(), ("#b3261e", "#1f5fa8")):
        xs = [b["conf"] for b in bins if b["n"]]
        ys = [b["acc"] for b in bins if b["n"]]
        ns = [b["n"] for b in bins if b["n"]]
        ax.plot(xs, ys, marker="o", color=color, label=label)
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7, color=color)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Top-1 confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
