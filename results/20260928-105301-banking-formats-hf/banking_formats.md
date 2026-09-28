# BANKING77 input-format ablation (CLM-8B, zero-shot)

State formats: `suffix` = message + blank line + question (CLM's documented layout); `none` = message only; `prefix` = question first. Label styles: `plain` = 'card arrival'; `sentence` = 'The customer is asking about card arrival.'

| State | Labels | Accuracy | Top-5 | Median gold rank /77 | Distinct predictions | Share of most common prediction | Mean state cosine | Centered acc. (diagnostic) |
|---|---|---|---|---|---|---|---|---|
| suffix | plain | 2.4% | 16.6% | 24 | 18 | 75.8% | 0.887 | 18.0% |
| suffix | sentence | 8.4% | 26.6% | 19 | 19 | 42.8% | 0.887 | 26.0% |
| none | plain | 13.8% | 36.0% | 10 | 34 | 21.6% | 0.690 | 25.0% |
| none | sentence | 19.2% | 43.8% | 8 | 45 | 13.4% | 0.690 | 28.2% |
| prefix | plain | 12.2% | 32.4% | 12 | 33 | 26.2% | 0.736 | 22.6% |
| prefix | sentence | 16.0% | 39.4% | 10 | 39 | 38.4% | 0.736 | 28.0% |