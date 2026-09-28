"""clm-bench command line.

  clm-bench sanity                         # 1-minute end-to-end check on the real model
  clm-bench calibration --datasets arc banking77 --n 500
  clm-bench latency --ks 4 16 77 256 1024
  clm-bench parity --vllm-url http://gpu-box:8090/v1/embeddings   # HF vs vLLM embeddings

Add ``--backend fake --random-heads --datasets synthetic`` for an offline plumbing test.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

import numpy as np


# ----------------------------------------------------------------------------- shared
def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["hf", "vllm", "fake"], default="hf",
                   help="hf = transformers on MPS/CUDA/CPU (default, works on a Mac); vllm = OpenAI-style "
                        "/v1/embeddings server; fake = plumbing test")
    p.add_argument("--model", default=None, help="HF model id for --backend hf (default Qwen/Qwen3-8B)")
    p.add_argument("--device", default=None, help="mps | cuda | cpu (auto-detected)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--batch-size", type=int, default=16, help="encoder batch size")
    p.add_argument("--vllm-url", default=None, help="for --backend vllm (default http://127.0.0.1:8090/v1/embeddings)")
    p.add_argument("--checkpoint", default=None, help="CLM head .pt (default: download CLM-v0.1-8B reference head)")
    p.add_argument("--random-heads", action="store_true", help="random-init heads (offline plumbing test only)")
    p.add_argument("--out", default="results", help="results root directory")
    p.add_argument("--seed", type=int, default=13)


def machine_info() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    if sys.platform == "darwin":
        for key, name in (("machdep.cpu.brand_string", "chip"), ("hw.memsize", "memory_bytes")):
            try:
                info[name] = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True,
                                            timeout=5).stdout.strip()
            except Exception:
                pass
        if info.get("memory_bytes", "").isdigit():
            info["memory_gb"] = round(int(info.pop("memory_bytes")) / 2**30)
    return info


def build(args):
    from .embedders import make_embedder, pick_device
    from .scorer import CLMScorer
    if args.backend == "hf" and sys.platform == "darwin" and "memory_gb" in machine_info():
        gb = machine_info()["memory_gb"]
        if gb < 32:
            print(f"[warn] {gb} GB unified memory: Qwen3-8B in bf16 needs ~17 GB for weights alone. "
                  "Expect swapping or an out-of-memory error.", flush=True)
    print(f"[setup] loading encoder backend={args.backend}", flush=True)
    emb = make_embedder(args.backend, model=args.model, device=args.device, dtype=args.dtype,
                        batch_size=args.batch_size, vllm_url=args.vllm_url)
    head_device = emb.device if emb.device not in ("remote",) else pick_device(args.device)
    print(f"[setup] loading CLM heads on {head_device}", flush=True)
    scorer = CLMScorer(args.checkpoint, head_device, random_heads=args.random_heads)
    return emb, scorer


def run_dir(args, kind: str) -> str:
    d = os.path.join(args.out, f"{datetime.now():%Y%m%d-%H%M%S}-{kind}-{args.backend}")
    os.makedirs(d, exist_ok=True)
    return d


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if callable(o):
        return getattr(o, "__name__", str(o))
    return str(o)


def write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)


def f3(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pct(x) -> str:
    return f"{100 * x:.1f}%"


# ----------------------------------------------------------------------------- sanity
def cmd_sanity(args) -> None:
    from clm.schema import state_text
    emb, scorer = build(args)
    probes = [
        ("Who wrote the play Romeo and Juliet?", None,
         ["William Shakespeare", "Christopher Marlowe", "Charles Dickens", "Jane Austen"]),
        ("Customer: my invoice was charged twice and nobody answers the phone!",
         "Which team should handle this?", ["Charges, invoices, refunds", "Bugs and outages"]),
    ]
    for ctx, q, cands in probes:
        s = state_text(ctx, q)
        se, ce = emb.embed([s]), emb.embed(cands)
        for label, z in (("clm", scorer.logits(se, ce)[0]), ("raw", scorer.raw_logits(se, ce)[0])):
            p = np.exp(z - z.max()); p /= p.sum()
            ranked = sorted(zip(cands, p), key=lambda kv: -kv[1])
            print(f"\n[{label}] {ctx[:60]}")
            for c, v in ranked:
                print(f"   {v:6.3f}  {c}")
    print("\nExpect CLM to rank 'William Shakespeare' and 'Charges, invoices, refunds' first. "
          "If it does not, stop: the embeddings do not match what the heads were trained on.")


# ----------------------------------------------------------------------------- calibration
def cmd_calibration(args) -> None:
    from . import calibration as C
    from .data import load
    from .embcache import EmbeddingCache
    emb, scorer = build(args)
    out = run_dir(args, "calibration")
    tag = f"{emb.name}-{(args.model or 'Qwen3-8B').replace('/', '_')}-{args.dtype}"
    cache = EmbeddingCache(os.path.join(args.out, ".emb_cache", f"{tag}.npz"))
    report = {"machine": machine_info(), "encoder": emb.describe(), "heads": scorer.describe(),
              "args": vars(args), "datasets": {}}
    t0 = time.time()
    for name in args.datasets:
        print(f"[data] loading {name}", flush=True)
        items = load(name, args.n, args.seed, args.banking_format, args.banking_labels)
        use_distractors = not args.no_distractors and max(len(i.candidates) for i in items) <= 10
        res, curves = C.run_dataset(name, items, emb, scorer, cache, args.splits, args.seed,
                                    use_distractors, args.bins)
        report["datasets"][name] = res
        C.plot_reliability(curves, f"CLM-8B reliability — {name} (n={len(items)})",
                           os.path.join(out, f"reliability_{name}.png"))
    report["wall_seconds"] = round(time.time() - t0, 1)
    report["encoder_stats"] = vars(emb.stats)
    write_json(os.path.join(out, "calibration.json"), report)
    md = calibration_markdown(report)
    with open(os.path.join(out, "calibration.md"), "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"[done] {out}")


def calibration_markdown(r: dict) -> str:
    L = ["# CLM-8B calibration benchmark", "",
         f"Encoder: `{r['encoder']}`  ", f"Heads: `{r['heads']['checkpoint']}` "
         f"(logit scale {r['heads']['logit_scale']}, {r['heads']['head_params']:,} params)  ",
         f"Machine: `{r['machine']}`", ""]
    for name, d in r["datasets"].items():
        L += [f"## {name}  (n={d['n']}, K={d['k']['min']}–{d['k']['max']})", ""]
        if name == "banking77":
            a = r.get("args", {})
            L += [f"Input layout: state `{a.get('banking_format', 'suffix')}`, "
                  f"labels `{a.get('banking_labels', 'plain')}`.", ""]
        L += [
              "At the checkpoint's own temperature (T=1), full set:", "",
              "| Scorer | Accuracy | Chance | Mean conf. | ECE | MCE | NLL | Brier |",
              "|---|---|---|---|---|---|---|---|"]
        for m, lab in (("clm", "CLM-8B"), ("raw_qwen3_no_heads", "Raw Qwen3-8B cosine (no heads)")):
            s = d[m]["at_T1"]
            L.append(f"| {lab} | {pct(s['accuracy'])} | {pct(s['chance_accuracy'])} | {f3(s['mean_confidence'])} "
                     f"| {f3(s['ece'])} | {f3(s['mce'])} | {f3(s['nll'])} | {f3(s['brier'])} |")
        ts = d["clm"]["temperature_scaling"]
        b, a = ts["test_before"], ts["test_after"]
        L += ["", f"Post-hoc temperature scaling for CLM-8B, fit on a random half, scored on the other half, "
                  f"{ts['splits']} splits (mean ± sd). Fitted T = {ts['T']['mean']:.2f} ± {ts['T']['sd']:.2f}.", "",
              "| Metric (held-out half) | Before | After |", "|---|---|---|"]
        for k in ("ece", "mce", "nll", "brier", "mean_confidence", "accuracy"):
            L.append(f"| {k} | {b[k]['mean']:.3f} ± {b[k]['sd']:.3f} | {a[k]['mean']:.3f} ± {a[k]['sd']:.3f} |")
        ds = d.get("distractor_sensitivity")
        if ds:
            L += ["", "Distractor sensitivity (original options vs. + 'None of the above.', \"I don't know.\", "
                      "and another question's correct answer):", "",
                  "| Measure | Original | With distractors |", "|---|---|---|",
                  f"| Mean P(gold) | {f3(ds['mean_p_gold']['original'])} | {f3(ds['mean_p_gold']['with_distractors'])} |",
                  f"| Mean P(top-1) | {f3(ds['mean_p_top']['original'])} | {f3(ds['mean_p_top']['with_distractors'])} |",
                  f"| Accuracy | {pct(ds['accuracy']['original'])} | {pct(ds['accuracy']['with_distractors'])} |"]
            for t, v in ds["share_above_threshold"].items():
                L.append(f"| Share with top-1 ≥ {t} | {pct(v['original'])} | {pct(v['with_distractors'])} |")
            L += ["", f"Mean |ΔP(gold)| = {ds['mean_abs_delta_p_gold']:.3f}. A distractor became top-1 on "
                      f"{pct(ds['top1_became_distractor_rate'])} of items. "
                  + " ".join(f"{pct(v)} of items fell below {t} on their original options."
                             for t, v in ds["fell_below_threshold_rate"].items())
                  + f" IIA check (should be ~0): {ds['iia_check_max_abs_error']:.2e}."]
        L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------------------- latency
def cmd_latency(args) -> None:
    from . import latency as LT
    from .data import BANKING_INSTRUCTION, banking77_candidate_pool, load, synthetic_pool
    emb, scorer = build(args)
    out = run_dir(args, "latency")
    if args.pool == "synthetic":
        pool = synthetic_pool()
        states = [it.state for it in load("synthetic", args.n_states, args.seed)]
    else:
        print("[data] loading BANKING77 for states and candidate pool", flush=True)
        pool = banking77_candidate_pool()
        states = [it.state for it in load("banking77", args.n_states, args.seed)]   # test split; pool is train
    res = LT.run(emb, scorer, states, pool, args.ks, args.n_cached, args.n_uncached, args.warmup,
                 args.max_seconds_per_cell)
    report = {"machine": machine_info(), "encoder": emb.describe(), "heads": scorer.describe(),
              "args": vars(args), "pool": args.pool, **res}
    write_json(os.path.join(out, "latency.json"), report)
    LT.plot(res, os.path.join(out, "latency.png"), f"CLM-8B decision latency, {report['machine'].get('chip') or report['machine']['machine']}")
    md = latency_markdown(report)
    with open(os.path.join(out, "latency.md"), "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"[done] {out}")


def latency_markdown(r: dict) -> str:
    L = ["# CLM-8B latency: cached vs. uncached action embeddings", "",
         f"Encoder: `{r['encoder']}`  ", f"Machine: `{r['machine']}`  ",
         f"Candidate pool: {r['pool']}; one state per decision, {r['n_states']} distinct states.", "",
         "| K | Cached p50 (ms) | Cached p95 | Uncached p50 (ms) | Uncached p95 | Speedup (p50) | "
         "Cache build, one-time (ms) | Cached: encode / head+score p50 (ms) |",
         "|---|---|---|---|---|---|---|---|"]
    for row in r["rows"]:
        c, u = row["cached"], row["uncached"]
        L.append(f"| {row['k']} | {c['p50_ms']:.1f} | {c['p95_ms']:.1f} | {u['p50_ms']:.1f} | {u['p95_ms']:.1f} "
                 f"| {row['speedup_p50']:.1f}× | {c['action_cache_build_ms']:.0f} "
                 f"| {c['encode_p50_ms']:.1f} / {c['head_and_score_p50_ms']:.2f} |")
    L += ["", "Absolute times are for this machine only; the CLM blog's figures are from an H100. "
              "Compare the two columns with each other, not with the blog."]
    return "\n".join(L)


# ----------------------------------------------------------------------------- banking formats
def cmd_banking_formats(args) -> None:
    """Why did BANKING77 collapse? Same 500 messages, different state layouts and label wordings."""
    from .data import LABEL_STYLES, STATE_FORMATS, load_banking77
    from .embcache import EmbeddingCache
    emb, scorer = build(args)
    out = run_dir(args, "banking-formats")
    tag = f"{emb.name}-{(args.model or 'Qwen3-8B').replace('/', '_')}-{args.dtype}"
    cache = EmbeddingCache(os.path.join(args.out, ".emb_cache", f"{tag}.npz"))

    def mean_offdiag_cos(x: np.ndarray) -> float:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        g = x @ x.T
        n = len(x)
        return float((g.sum() - n) / (n * n - n))

    rows = []
    for fmt in STATE_FORMATS:
        for style in LABEL_STYLES:
            items = load_banking77(args.n, args.seed, fmt, style)
            gold = np.array([it.gold for it in items])
            s = cache.get_many(emb, [it.state for it in items], label=f"states[{fmt}]")
            c = cache.get_many(emb, items[0].candidates, label=f"labels[{style}]")
            zs = scorer.project_states(s).float().cpu().numpy()
            za = scorer.project_actions(c).float().cpu().numpy()
            z = scorer.scale * zs @ za.T
            rank = (z > z[np.arange(len(gold)), gold][:, None]).sum(1) + 1
            pred = z.argmax(1)
            zc = zs - zs.mean(0)
            zc /= np.linalg.norm(zc, axis=1, keepdims=True)
            rows.append({"state_format": fmt, "label_style": style, "n": len(items),
                         "accuracy": float((pred == gold).mean()), "top5": float((rank <= 5).mean()),
                         "median_gold_rank": float(np.median(rank)),
                         "distinct_predictions": int(len(set(pred.tolist()))),
                         "most_common_prediction_share": float(np.bincount(pred).max() / len(pred)),
                         "mean_state_cosine_projected": mean_offdiag_cos(zs),
                         "centered_accuracy_diagnostic": float(((zc @ za.T).argmax(1) == gold).mean())})
            print(f"[fmt] state={fmt:6s} labels={style:8s} acc={rows[-1]['accuracy']:.3f} "
                  f"top5={rows[-1]['top5']:.3f}", flush=True)
    write_json(os.path.join(out, "banking_formats.json"), {"machine": machine_info(), "rows": rows})
    L = ["# BANKING77 input-format ablation (CLM-8B, zero-shot)", "",
         "State formats: `suffix` = message + blank line + question (CLM's documented layout); "
         "`none` = message only; `prefix` = question first. Label styles: `plain` = 'card arrival'; "
         "`sentence` = 'The customer is asking about card arrival.'", "",
         "| State | Labels | Accuracy | Top-5 | Median gold rank /77 | Distinct predictions | "
         "Share of most common prediction | Mean state cosine | Centered acc. (diagnostic) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['state_format']} | {r['label_style']} | {pct(r['accuracy'])} | {pct(r['top5'])} "
                 f"| {r['median_gold_rank']:.0f} | {r['distinct_predictions']} "
                 f"| {pct(r['most_common_prediction_share'])} | {r['mean_state_cosine_projected']:.3f} "
                 f"| {pct(r['centered_accuracy_diagnostic'])} |")
    md = "\n".join(L)
    with open(os.path.join(out, "banking_formats.md"), "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"[done] {out}")


# ----------------------------------------------------------------------------- parity
def cmd_parity(args) -> None:
    """Cosine between HF-transformers and vLLM embeddings of the same texts (needs both)."""
    from .embedders import HFEmbedder, VLLMEmbedder
    texts = ["Who wrote the play Romeo and Juliet?", "William Shakespeare",
             "Customer: my invoice was charged twice and nobody answers the phone!\n\nWhich team should handle this?",
             "card arrival", "I still have not received my new card, I ordered over a week ago."]
    hf = HFEmbedder(args.model or "Qwen/Qwen3-8B", args.device, args.dtype, args.batch_size)
    vl = VLLMEmbedder(args.vllm_url or "http://127.0.0.1:8090/v1/embeddings")
    a, b = hf.embed(texts), vl.embed(texts)
    cos = (a * b).sum(-1)
    for t, c in zip(texts, cos):
        print(f"{c:.5f}  {t[:70]!r}")
    print(f"min cosine {cos.min():.5f} (expect > 0.99 for bf16 on both sides)")


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="clm-bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sanity", help="quick end-to-end check on two probes")
    add_common(s)
    s.set_defaults(fn=cmd_sanity)

    c = sub.add_parser("calibration", help="accuracy, ECE, NLL, Brier, temperature scaling, distractors")
    add_common(c)
    c.add_argument("--datasets", nargs="+", default=["arc", "banking77"])
    c.add_argument("--n", type=int, default=500, help="items per dataset (0 = all)")
    c.add_argument("--splits", type=int, default=5, help="random calibration/test splits for temperature scaling")
    c.add_argument("--bins", type=int, default=15)
    c.add_argument("--no-distractors", action="store_true")
    c.add_argument("--banking-format", choices=["suffix", "none", "prefix"], default="suffix",
                   help="BANKING77 state layout: suffix = message then question (CLM's documented default); "
                        "none = message only; prefix = question first")
    c.add_argument("--banking-labels", choices=["plain", "sentence"], default="plain",
                   help="BANKING77 candidate wording: plain = 'card arrival'; "
                        "sentence = 'The customer is asking about card arrival.'")
    c.set_defaults(fn=cmd_calibration)

    l = sub.add_parser("latency", help="cached vs uncached decision latency vs K")
    add_common(l)
    l.add_argument("--ks", type=int, nargs="+", default=[4, 16, 77, 256, 1024])
    l.add_argument("--pool", choices=["banking77", "synthetic"], default="banking77")
    l.add_argument("--n-states", type=int, default=64)
    l.add_argument("--n-cached", type=int, default=30)
    l.add_argument("--n-uncached", type=int, default=8)
    l.add_argument("--warmup", type=int, default=3)
    l.add_argument("--max-seconds-per-cell", type=float, default=180.0,
                   help="stop a (K, mode) cell early after this long (min 3 queries)")
    l.set_defaults(fn=cmd_latency)

    b = sub.add_parser("banking-formats", help="BANKING77 state-layout / label-wording ablation")
    add_common(b)
    b.add_argument("--n", type=int, default=500)
    b.set_defaults(fn=cmd_banking_formats)

    p = sub.add_parser("parity", help="HF vs vLLM embedding agreement")
    add_common(p)
    p.set_defaults(fn=cmd_parity)

    args = ap.parse_args(argv)
    if getattr(args, "n", None) == 0:
        args.n = None
    args.fn(args)


if __name__ == "__main__":
    main()
