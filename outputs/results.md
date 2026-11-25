| system | BLEU-4 | ROUGE-L | METEOR | CIDEr | CLIPScore | mean chars | <= 125 chars % | sentence case + period % | non-visible term % | mentions a person % | mentions the image % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BLIP-base, zero-shot | 2.3 | 22.1 | 18.0 | 18.5 | 77.9 | 48.0 | 97.8 | 0.0 | 3.9 | 9.7 | 8.6 |
| Florence-2-base, zero-shot (caption prompt) | 1.8 | 22.6 | 17.4 | 16.5 | 77.9 | 46.8 | 100.0 | 97.6 | 0.2 | 0.7 | 0.3 |
| Florence-2-base, zero-shot (detailed-caption prompt) | 0.0 | 10.2 | 8.3 | 5.3 | 65.4 | 59.1 | 100.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| Florence-2-base, fine-tuned (ours) | 37.3 | 63.9 | 64.0 | 281.6 | 80.5 | 91.4 | 94.2 | 100.0 | 0.0 | 0.0 | 0.0 |
| Reference alt-text (LLM rewrite of the H&M caption) |  |  |  |  | 79.7 | 94.6 | 90.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| Raw H&M caption |  |  |  |  | 79.3 | 163.3 | 27.9 | 0.0 | 87.6 | 0.0 | 0.0 |
