# MedQA-US test set

`test.jsonl`: 1,273 USMLE-style questions with **four** answer options (A-D),
one JSON object per line:

```json
{"question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
 "answer_idx": "B", "answer": "text of option B"}
```

- Source: MedQA, Jin et al., "What Disease Does This Patient Have? A
  Large-scale Open Domain Question Answering Dataset from Medical Exams",
  Applied Sciences, 2021. Copyright (c) 2022 Di Jin, MIT licence.
- Converted into this layout from the MIRAGE benchmark's `benchmark.json`.
- The four-option variant is the one MedGemma's published MedQA scores use.
- There is no `train.jsonl`, so the few-shot examplers MDAgents normally uses are
  absent and a run prints a warning. Prompts therefore differ from the published
  setup. The training split is in the original MedQA release.
