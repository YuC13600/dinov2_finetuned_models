# dinov2_finetuned_models
Dinov2 models for Spect-coral project

## Models

| Model Name | Batch Size | Same Area Negative | Box Size | TestSet Accuracy | TestSet Loss| N-Benchmark A37 | N-Benchmark A38 | N-Benchmark A39 | N-Benchmark A40 | N-Benchmark Avg | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20250812_152526 | 32 | ❌ | bbox | 92.6% | 0.1659 | 50.00% | 51.61% | 48.15% | 43.24% | 48.25% | <hr> |
| 20251007_133126 | 32 | ✅ | bbox | 88.8% | 0.2523 | 46.88% | 41.94% | 33.33% | 35.14% | 39.32% | <hr> |
| 20251008_094017 | 16 | ✅ | bbox | 90.4% | 0.1636 | 37.50% | 48.39% | 37.04% | 37.84% | 40.19% | <hr> |
| 20251008_234015 | 64 | ✅ | bbox | N/A | N/A | N/A | N/A | N/A | N/A | N/A | CUDA out of memory |
| 20251014_183603 | 16 | ❌ | bbox | 92.8% | 0.1012 | 37.50% | 38.71% | 44.44% | 43.24% | 40.97% | <hr> |
| 20251015_165008 | 16 | ✅ | whole image | 92.7% | 0.1330 | 62.50% | 61.29% | 55.56% | 78.38% | 64.43% | <hr> |
| 20251016_133229 | 16 | ❌ | whole image | 97.9% | 0.429 | 56.25% | 58.06% | 74.07% | 64.86% | 63.31% | <hr> |

> `N-Benchmark(Nearest Benchmark)`: The accuracy rate of identifying the nearest coral as the correct specimen when comparing coral data in areas 37-40 across 2022 and 2023.