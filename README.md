# clm-bench

Independent checks of two claims about **CLM-8B** ([Contrastive Language Models](https://contrastive-lm.notion.site), Kwok et al., Sept 2026), run on your own hardware:

1. **Calibration.** CLM's probabilities are a softmax over `scale × cosine(state, candidate)` with no calibration objective. How well calibrated are they out of the box, and how much does post-hoc temperature scaling fix?
2. **Set dependence.** Because every candidate is scored independently, relative odds between options never change when you add or remove one (independence of irrelevant alternatives). The absolute probabilities do. How far do they move when neutral distractors like "None of the above." are added, and how many decisions cross a 0.5 or 0.9 auto-act threshold as a result?
3. **Caching.** How much of CLM's speed comes from cached action embeddings? The benchmark times cached and uncached decisions as the candidate count K grows from 4 to 1,024.

## Requirements

- An Apple Silicon Mac with **32 GB or more** of unified memory (Qwen3-8B in bf16 is about 17 GB), or any Linux box with a 24 GB+ NVIDIA GPU.
- Python 3.10–3.12 and about 20 GB of free disk for the Qwen3-8B weights (downloaded from Hugging Face on first run).

vLLM, CLM's reference encoder server, has no macOS build. On a Mac the encoder runs through Hugging Face transformers on MPS, using the same last-token pooling and L2 normalisation. On a CUDA box you can use either backend, and `clm-bench parity` checks that they agree.

## Quick start

```bash
./setup.sh                      # venv + deps + contrastive-lm (installed without its vLLM dependency)
source .venv/bin/activate
pytest -q                       # metric unit tests, no model needed

clm-bench sanity                # ~2 min incl. model load; must rank Shakespeare first
clm-bench calibration           # ARC-Challenge + BANKING77, 500 items each
clm-bench latency               # K = 4, 16, 77, 256, 1024
```

Each run writes to `results/<timestamp>-<kind>-<backend>/`:

- `calibration.md` / `calibration.json` and one `reliability_<dataset>.png` per dataset
- `latency.md` / `latency.json` / `latency.png`

Encoder embeddings for the calibration run are cached under `results/.emb_cache/`, so re-runs and analysis changes don't re-encode anything.

Rough timings on an M-series Mac: `sanity` a couple of minutes, `calibration` 5–20 minutes, `latency` up to about 30 minutes. The largest uncached cells stop after `--max-seconds-per-cell` (default 180 s, minimum 3 queries).

## What each benchmark does

### Calibration (`clm-bench calibration`)

| Dataset | Items | Candidates | Why |
|---|---|---|---|
| ARC-Challenge test (`allenai/ai2_arc`) | 500 sampled | 3–5 answer texts | Same shape as CLM's Nemotron Q&A pre-training: question = state, answer = action |
| BANKING77 test (`mteb/banking77`) | 500 sampled | all 77 intent names | Typed classification with large K, the regime CLM's zero-shot suite didn't cover |

Formatting follows `clm.schema`: the state head sees `context + "\n\n" + question`, and the action head sees each option's plain text.

The run reports:

- Accuracy, chance accuracy, mean confidence, **ECE** (15 equal-width bins), MCE, NLL and multi-class Brier, all at the checkpoint's own temperature. The same metrics are reported for a **no-heads ablation** (raw Qwen3-8B cosine), which shows what the heads add.
- **Temperature scaling**: T is fit by NLL on a random half and scored on the other half, over 5 random splits (mean ± sd).
- **Distractor sensitivity** (datasets with ≤ 10 options): the run adds "None of the above.", "I don't know." and a correct answer taken from a *different* question. It reports the shift in P(gold) and P(top-1), threshold crossings at 0.5 and 0.9, and how often a distractor becomes top-1. It also includes an IIA check that should print about 1e-16, which confirms the ratios among original options are untouched.

### Latency (`clm-bench latency`)

States are BANKING77 test messages. Candidates are K distinct BANKING77 training utterances, which are realistic, similar-length texts. Per decision:

- **uncached**: encode the state and all K candidates, project them, and score. This is a stateless request.
- **cached**: the K action projections are built once, and that one-time cost is reported separately. Per decision, only the state is encoded, projected and dotted with the cached matrix.

The report gives p50/p95 per cell, the speedup, and the cached path's split between encoding and head-plus-score time.

**Compare the two columns with each other, not with the blog.** The blog's milliseconds come from an H100, while these come from your machine.

## Useful flags

```text
--n 1000 | --n 0 (all)        items per dataset
--datasets arc                 run one dataset
--batch-size 8                 lower if memory is tight
--dtype float16                if bf16 misbehaves on your torch/MPS build
--backend vllm --vllm-url URL  use a vLLM pooling server instead of transformers
--checkpoint path/to/head.pt   evaluate a fine-tuned head instead of the reference one
--backend fake --random-heads --datasets synthetic   offline plumbing test (numbers meaningless)
```

## Caveats worth stating when you publish

- These results cover the **reference checkpoint zero-shot**, not the fine-tuned verifier heads behind the DeepSWE and Terminal-Bench numbers.
- Temperature scaling fixes the average miscalibration. It does not fix set dependence: absolute probabilities still move when the candidate set changes.
- BANKING77 is scored against label names ("card arrival"). A richer description per intent would likely raise accuracy, so treat that number as a floor.
- The HF-transformers path runs in bf16 on MPS. On a CUDA box, `clm-bench parity` confirms it matches vLLM (expected cosine > 0.99).
