# Instructions for coding agents (Claude Code, Codex, etc.)

This repo benchmarks CLM-8B's calibration, distractor sensitivity and cached/uncached latency. Read `README.md` first.

## Run order

1. `./setup.sh && source .venv/bin/activate`
2. `pytest -q`. All tests must pass before touching the model.
3. `clm-bench sanity`. CLM **must** rank "William Shakespeare" and "Charges, invoices, refunds" first. If it doesn't, the embeddings don't match what the heads were trained on. Stop and report; don't tune around it.
4. `clm-bench calibration`
5. `clm-bench latency`
6. Report by pasting the generated `calibration.md` and `latency.md` and attaching the PNGs from the run's `results/` subfolder.

## Rules

- **Don't change metric definitions** in `src/clm_bench/metrics.py` (ECE = 15 equal-width bins on top-1 confidence; Brier = multi-class sum of squares). If you think one is wrong, say so and ask first.
- **Don't change how texts are formatted** (`clm.schema.state_text`, plain candidate text). That formatting matches CLM's training layout.
- **Don't modify the installed `contrastive-lm` package.** It's installed with `--no-deps` on purpose: its vLLM dependency has no macOS wheels.
- **Keep the encoder faithful**: Qwen3-8B, last-token pooling, left padding, L2 normalisation, max 2048 tokens.
- **Don't push, publish or upload results anywhere.** Everything stays local.

## Common problems

| Symptom | Fix |
|---|---|
| Out of memory, heavy swapping | `--batch-size 4`; close other apps; the Mac needs ≥ 32 GB |
| Garbage or NaN outputs on MPS | `--dtype float16`, or upgrade torch; last resort `--device cpu` (slow) |
| `load_dataset` schema/column error | Fix the loader in `src/clm_bench/data.py` so it yields the same `Item` fields; don't change formatting |
| Hugging Face 401/403 | `huggingface-cli login` (Qwen3-8B and the datasets are public) |
| A latency cell takes forever | It stops after `--max-seconds-per-cell` (min 3 queries); lower `--n-uncached` or drop `--ks 1024` |

## Layout

```text
src/clm_bench/embedders.py    hf (transformers) | vllm | fake encoders -> [n, 4096] L2-normalised
src/clm_bench/scorer.py       CLM heads: scale * cosine; raw no-head ablation
src/clm_bench/data.py         ARC-Challenge, BANKING77, synthetic -> Item(state, candidates, gold)
src/clm_bench/metrics.py      ECE, MCE, NLL, Brier, temperature fit
src/clm_bench/calibration.py  calibration + temperature scaling + distractor study
src/clm_bench/latency.py      cached vs uncached timing
src/clm_bench/cli.py          clm-bench sanity | calibration | latency | parity
tests/                        metric unit tests
```
