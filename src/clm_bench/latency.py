"""Cached vs. uncached latency for CLM decisions as the candidate set grows.

Two per-query paths, both timed end to end on this machine:

* ``uncached`` – a stateless call: encode the state AND all K candidates, project
  both, score.  What a single request costs with nothing reused.
* ``cached``   – the agent-loop case CLM is built for: the K candidate projections
  were computed once up front; per query only the state is encoded, projected and
  dotted against the cached matrix.

The one-time cost of building the action cache is reported separately.  Absolute
milliseconds depend on hardware (the CLM blog measured an H100); the ratio between
the two paths on the same machine is the portable result.
"""
from __future__ import annotations

import time

import numpy as np

from .embedders import sync


def _pct(v: list[float], q: float) -> float:
    return float(np.percentile(np.array(v), q)) if v else float("nan")


def _stats(total: list[float], enc: list[float], head: list[float]) -> dict:
    return {"n": len(total), "p50_ms": _pct(total, 50), "p95_ms": _pct(total, 95),
            "mean_ms": float(np.mean(total)) if total else float("nan"),
            "encode_p50_ms": _pct(enc, 50), "head_and_score_p50_ms": _pct(head, 50)}


def time_cached(embedder, scorer, states: list[str], cands: list[str], n: int, budget_s: float) -> dict:
    t0 = time.perf_counter()
    za = scorer.project_actions(embedder.embed(cands))
    sync(scorer.device)
    build_ms = (time.perf_counter() - t0) * 1000
    tot, enc, head = [], [], []
    start = time.perf_counter()
    for i in range(n):
        a = time.perf_counter()
        e = embedder.embed([states[i % len(states)]])
        b = time.perf_counter()
        logits = scorer.logits_from_projected(scorer.project_states(e), za)
        sync(scorer.device)
        c = time.perf_counter()
        _ = int(np.argmax(logits))
        tot.append((c - a) * 1000); enc.append((b - a) * 1000); head.append((c - b) * 1000)
        if time.perf_counter() - start > budget_s and len(tot) >= 3:
            break
    out = _stats(tot, enc, head)
    out["action_cache_build_ms"] = build_ms
    return out


def time_uncached(embedder, scorer, states: list[str], cands: list[str], n: int, budget_s: float) -> dict:
    tot, enc, head = [], [], []
    start = time.perf_counter()
    for i in range(n):
        a = time.perf_counter()
        e = embedder.embed([states[i % len(states)]] + cands)
        b = time.perf_counter()
        logits = scorer.logits_from_projected(scorer.project_states(e[:1]), scorer.project_actions(e[1:]))
        sync(scorer.device)
        c = time.perf_counter()
        _ = int(np.argmax(logits))
        tot.append((c - a) * 1000); enc.append((b - a) * 1000); head.append((c - b) * 1000)
        if time.perf_counter() - start > budget_s and len(tot) >= 3:
            break
    return _stats(tot, enc, head)


def run(embedder, scorer, states: list[str], pool: list[str], ks: list[int], n_cached: int, n_uncached: int,
        warmup: int, budget_s: float) -> dict:
    ks = [k for k in ks if k <= len(pool)]
    print(f"[latency] warmup x{warmup}", flush=True)
    for i in range(warmup):
        embedder.embed([states[i % len(states)]] + pool[:8])
        scorer.logits(embedder.embed([states[0]]), embedder.embed(pool[:4]))
    sync(scorer.device)
    rows = []
    for k in ks:
        cands = pool[:k]
        print(f"[latency] K={k}: cached", flush=True)
        cached = time_cached(embedder, scorer, states, cands, n_cached, budget_s)
        print(f"[latency] K={k}: cached p50 {cached['p50_ms']:.1f} ms; uncached...", flush=True)
        uncached = time_uncached(embedder, scorer, states, cands, n_uncached, budget_s)
        print(f"[latency] K={k}: uncached p50 {uncached['p50_ms']:.1f} ms", flush=True)
        rows.append({"k": k, "cached": cached, "uncached": uncached,
                     "speedup_p50": uncached["p50_ms"] / cached["p50_ms"]})
    return {"rows": rows, "n_states": len(states)}


def plot(result: dict, path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [r["k"] for r in result["rows"]]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for mode, color in (("uncached", "#b3261e"), ("cached", "#1f5fa8")):
        p50 = [r[mode]["p50_ms"] for r in result["rows"]]
        p95 = [r[mode]["p95_ms"] for r in result["rows"]]
        ax.plot(ks, p50, marker="o", color=color, label=f"{mode} (p50)")
        ax.fill_between(ks, p50, p95, color=color, alpha=0.12, linewidth=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Candidate actions per decision (K)")
    ax.set_ylabel("Latency per decision (ms, log)")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
