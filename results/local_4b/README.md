# Local runs, MedGemma 1.5 4B

Run on a MacBook with Ollama, 17 October 2026. MedQA-US 4-option, seed 0,
answer options shuffled, `MEDGEMMA_MAX_TOKENS=2048`, one shared 4-hour budget.

| File | Path | Questions |
|---|---|---|
| `basic.json` | single agent | 90 (all) |
| `adaptive.json` | classifier chooses the path | 57 (budget ran out) |

```bash
python3 analyze.py results/local_4b/basic.json results/local_4b/adaptive.json
```

| | Accuracy | Per question |
|---|---|---|
| single agent, 90 questions | 64.4% [54.1, 73.6] | 6.3 s |
| single agent, the 57 shared | 59.6% [46.7, 71.4] | |
| multi-agent, 57 questions | 57.9% [45.0, 69.8] | 167.0 s |

Paired over the 57 shared questions: 29 both correct, 5 single-agent only,
4 multi-agent only, 19 neither; McNemar exact p = 1.000.

These files were written before `picked_via` and `trace` were added to the
output, so those fields are absent. `analyze.py` handles that.
