# CLM-8B calibration benchmark

Encoder: `{'backend': 'hf', 'model': 'Qwen/Qwen3-8B', 'device': 'mps', 'dtype': 'bfloat16', 'batch_size': 16, 'load_seconds': 2.3}`  
Heads: `/Users/suneel.marti/.cache/clm/CLM_v0.1-8B.pt` (logit scale 100.0, 18,887,680 params)  
Machine: `{'platform': 'macOS-26.5.2-arm64-arm-64bit', 'machine': 'arm64', 'python': '3.11.14', 'torch': '2.14.0', 'transformers': '5.17.0', 'chip': 'Apple M3 Max', 'memory_gb': 128}`

## arc  (n=500, K=3–5)

At the checkpoint's own temperature (T=1), full set:

| Scorer | Accuracy | Chance | Mean conf. | ECE | MCE | NLL | Brier |
|---|---|---|---|---|---|---|---|
| CLM-8B | 40.6% | 25.0% | 0.723 | 0.324 | 0.488 | 1.902 | 0.846 |
| Raw Qwen3-8B cosine (no heads) | 27.0% | 25.0% | 0.521 | 0.251 | 0.711 | 1.958 | 0.880 |

Post-hoc temperature scaling for CLM-8B, fit on a random half, scored on the other half, 5 splits (mean ± sd). Fitted T = 4.35 ± 0.20.

| Metric (held-out half) | Before | After |
|---|---|---|
| ece | 0.319 ± 0.014 | 0.068 ± 0.019 |
| mce | 0.536 ± 0.083 | 0.453 ± 0.260 |
| nll | 1.907 ± 0.052 | 1.276 ± 0.010 |
| brier | 0.846 ± 0.017 | 0.690 ± 0.005 |
| mean_confidence | 0.722 ± 0.012 | 0.409 ± 0.010 |
| accuracy | 0.412 ± 0.011 | 0.412 ± 0.011 |

Distractor sensitivity (original options vs. + 'None of the above.', "I don't know.", and another question's correct answer):

| Measure | Original | With distractors |
|---|---|---|
| Mean P(gold) | 0.389 | 0.381 |
| Mean P(top-1) | 0.723 | 0.712 |
| Accuracy | 40.6% | 40.4% |
| Share with top-1 ≥ 0.5 | 82.8% | 81.4% |
| Share with top-1 ≥ 0.9 | 26.0% | 24.4% |

Mean |ΔP(gold)| = 0.008. A distractor became top-1 on 3.8% of items. 4.4% of items fell below 0.5 on their original options. 1.8% of items fell below 0.9 on their original options. IIA check (should be ~0): 2.22e-16.

## banking77  (n=500, K=77–77)

At the checkpoint's own temperature (T=1), full set:

| Scorer | Accuracy | Chance | Mean conf. | ECE | MCE | NLL | Brier |
|---|---|---|---|---|---|---|---|
| CLM-8B | 2.4% | 1.3% | 0.318 | 0.294 | 0.834 | 5.363 | 1.107 |
| Raw Qwen3-8B cosine (no heads) | 0.6% | 1.3% | 0.880 | 0.874 | 0.938 | 7.682 | 1.765 |

Post-hoc temperature scaling for CLM-8B, fit on a random half, scored on the other half, 5 splits (mean ± sd). Fitted T = 4.58 ± 0.48.

| Metric (held-out half) | Before | After |
|---|---|---|
| ece | 0.297 ± 0.008 | 0.016 ± 0.011 |
| mce | 0.835 ± 0.007 | 0.046 ± 0.036 |
| nll | 5.237 ± 0.128 | 4.147 ± 0.025 |
| brier | 1.106 ± 0.008 | 0.982 ± 0.001 |
| mean_confidence | 0.321 ± 0.009 | 0.039 ± 0.004 |
| accuracy | 0.024 ± 0.009 | 0.024 ± 0.009 |
