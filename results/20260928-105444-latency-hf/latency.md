# CLM-8B latency: cached vs. uncached action embeddings

Encoder: `{'backend': 'hf', 'model': 'Qwen/Qwen3-8B', 'device': 'mps', 'dtype': 'bfloat16', 'batch_size': 16, 'load_seconds': 7.5}`  
Machine: `{'platform': 'macOS-26.5.2-arm64-arm-64bit', 'machine': 'arm64', 'python': '3.11.14', 'torch': '2.14.0', 'transformers': '5.17.0', 'chip': 'Apple M3 Max', 'memory_gb': 128}`  
Candidate pool: banking77; one state per decision, 64 distinct states.

| K | Cached p50 (ms) | Cached p95 | Uncached p50 (ms) | Uncached p95 | Speedup (p50) | Cache build, one-time (ms) | Cached: encode / head+score p50 (ms) |
|---|---|---|---|---|---|---|---|
| 4 | 81.2 | 102.4 | 197.4 | 292.8 | 2.4× | 120 | 80.3 / 0.87 |
| 16 | 80.8 | 83.1 | 544.5 | 852.8 | 6.7× | 503 | 80.0 / 0.86 |
| 77 | 82.3 | 95.3 | 2306.5 | 2513.7 | 28.0× | 2278 | 81.3 / 0.99 |
| 256 | 81.8 | 86.8 | 8368.8 | 8622.0 | 102.3× | 8354 | 80.9 / 0.91 |
| 1024 | 85.0 | 91.2 | 40200.6 | 46509.0 | 472.8× | 39588 | 84.1 / 0.90 |

Absolute times are for this machine only; the CLM blog's figures are from an H100. Compare the two columns with each other, not with the blog.