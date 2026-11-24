# Results

Every run of the thesis so far. Run IDs match the experiment-log appendix of
the thesis; `scripts/run_tables.py` builds its tables from these files:

```bash
python3 scripts/run_tables.py > results/run_tables.tex
```

Common setup: MedQA-US test set (4 options), first *n* questions, seed 0,
options shuffled with a per-question seed, temperature 0, **no few-shot
examples** (the data has no training split).

| Hardware | Model | Serving | Output cap |
|---|---|---|---|
| GPU (G) | MedGemma 27B text-it | vLLM, Roihu GPU partition | 4096 |
| CPU (C, S) | MedGemma 27B text-it | llama.cpp Q8_0, Roihu CPU partition | 4096 |
| Laptop (L) | MedGemma 1.5 4B | Ollama, MacBook | 2048 |

## Answering runs

| ID | File | Path | Variant | n | Accuracy |
|---|---|---|---|---|---|
| G1 | `roihu_27b/roihu_27b_basic_seed0.json` | single agent | published | 500 | 86.8% |
| G2 | `roihu_27b/roihu_27b_basic_run2_seed0.json` | single agent | published, repeat of G1 | 500 | 85.2% |
| G3 | `roihu_27b/roihu_27b_adaptive_seed0.json` | adaptive | published | 500 | 85.2% |
| G4 | `roihu_27b/roihu_27b_intermediate_seed0.json` | team (forced) | published | 200 | 72.5% |
| G5 | `roihu_27b/roihu_27b_intermediate_diag_seed0.json` | team (forced) | published, talk replies traced | 50 | 68.0% |
| G6 | `roihu_27b/roihu_27b_intermediate_fixed_seed0.json` | team (forced) | `--fixed` | 100 | 78.0% |
| G7 | `roihu_27b/roihu_27b_adaptive_fixed_seed0.json` | adaptive | `--fixed` | 100 | 78.0% |
| L1 | `local_4b/basic.json` | single agent | published | 90 | 63.3% |
| L2 | `local_4b/adaptive.json` | adaptive | published | 57 | 57.9% |

## Classifier-only runs

| ID | File | What |
|---|---|---|
| G8 | `roihu_27b/roihu_27b_adaptive_classify_seed0.json` | code prompt, 200 q, each reply parsed as upstream and by a one-word follow-up (`classify_check.py`) |
| C1 | `routing/cpu_27b_adaptive_paperclf_test_seed0.json` | paper prompt, 5-question test |
| C2 | `routing/cpu_27b_adaptive_paperclf_compare_seed0.json` | paper prompt vs code prompt, 50 q |
| C3 | `routing/cpu_27b_adaptive_paperclf_seed0.json` | paper prompt, stopped at 77 of 200 q (resumable) |

## Protection layer

| ID | Where | What |
|---|---|---|
| S1 | Roihu CPU (`output/cpu_27b_*` on Roihu, not yet copied here) | smoke test: 3 q, single agent, inject, defended |
| D* | `local_4b_defence/` | local 4B test, every attack with and without the layer: `bash scripts/local_defence.sh basic 50`, then `team 10`, then `report` |

The local test needs `data/medqa/passages.jsonl` and `data/medqa/attacks.json`
from Roihu (they are generated there by `prepare_passages.py` and
`make_attacks.py`), so the attacks match the GPU evaluation.

## Older field sets

L1 and L2 predate `picked_via` and `trace`; G1 to G5 predate the cost fields
(`calls`, `prompt_tokens`, `completion_tokens`, `seconds`). The per-question
times for those runs come from the Slurm job logs.
