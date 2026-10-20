# Securing multi-agent LLM systems for clinical decision-making

Code for the bachelor's thesis *Securing Multi-Agent LLM Systems for Clinical
Decision-Making Against Prompt Injection* (Veeti Nurmikoski, Tampere University, 2026).

This repository is a fork of MDAgents (Kim et al., NeurIPS 2024), a multi-agent system that answers medical questions. It is the system the thesis tries to protect. The fork makes it run on the open model MedGemma instead of OpenAI models, and adds the tools to measure it. The original README is kept as UPSTREAM_README.md.

The thesis question

Agents in MDAgents pass each other's output as plain text, so an attacker who gets text into one agent's input (for example through a retrieved document) can influence every agent after it. The thesis adds a CaMeL-style capability layer that tracks where each value came from and checks it before use, and asks:

1. **RQ1** How well does it stop injected content?
2. **RQ2** What does it cost in diagnostic accuracy?
3. **RQ3** What does it cost in compute, time and money?

RQ2 and RQ3 need a baseline: how accurate and how expensive is the unprotected system? That is what this repository measures. The capability layer is not in the code yet.

## What has been done

| Step | Status |
|---|---|
| MDAgents running on MedGemma (local, Ollama) | done |
| Single agent vs adaptive multi-agent, MedGemma 1.5 4B, MedQA | done, small sample |
| Reliable answer extraction from free text | done |
| Same comparison with MedGemma 27B on Roihu, 500+ questions | next |
| Capability layer and attack suite | not started |

First results (MedGemma 1.5 4B, MedQA-US 4-option, seed 0):

| | Questions | Accuracy | Time per question |
|---|---|---|---|
| single agent | 90 | 64.4% [54.1, 73.6] | 6.3 s |
| adaptive multi-agent | 57 | 57.9% [45.0, 69.8] | 167.0 s |

## How the code is organised

```
main.py                 the experiment loop (one configuration, resumable)
utils.py                MDAgents itself: classifier, recruiter, debate (upstream, patched)
medgemma_backend.py     talks to the model server (Ollama or vLLM)
answer.py               turns the system's free-text decision into one answer letter
analyze.py              accuracy, confidence intervals, paired comparison
scripts/run_local.sh    run both configurations on a laptop
scripts/roihu.sh        run on the CSC Roihu supercomputer (Slurm)
data/medqa/             the 1,273 MedQA-US test questions
results/local_4b/       the runs above
```

## Run locally (Ollama)

```bash
ollama serve &
ollama pull medgemma:4b           # check the exact tag with `ollama list`

bash scripts/run_local.sh         # up to 90 questions, 4-hour budget, both arms
```

Or a single run:

```bash
export MEDGEMMA_BASE_URL=http://localhost:11434/v1
export MEDGEMMA_MODEL=medgemma:4b
export MEDGEMMA_API_KEY=ollama

python3 main.py --difficulty basic --num_samples 20 --data_root ./data
```

`--difficulty` is `adaptive` (the classifier chooses a path, the real MDAgents
protocol) or `basic` / `intermediate` / `advanced` to force one path.

## Changes from upstream

Files: `medgemma_backend.py`, `answer.py`, `analyze.py`, `scripts/`, `data/medqa/`
and `results/` are new; `main.py` is rewritten

Changes to `utils.py`:

1. Every model call goes through the configured model. Upstream hardcodes
   `gpt-3.5` / `gpt-4o-mini` in the complexity classifier, both recruiters and
   the group abstraction, so most of a "MedGemma run" would still call OpenAI.
2. `determine_difficulty` returned `None` for an unrecognised classifier reply,
   after which upstream silently re-used the previous question's answer. It now
   defaults to `basic` and warns.
3. The recruiter output is parsed tolerantly. Upstream crashes when the model
   does not follow the prompted format, which MedGemma 4B often does not. After
   two retries a default roster is used and a warning is printed.
4. `pptree` is optional (it does not install on current Python), and one f-string
   that needs Python 3.12 is fixed.
5. `TRACE` records what happened inside a question (see the result record above)