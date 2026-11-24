# Securing multi-agent LLM systems for clinical decision-making

Code for the bachelor's thesis *Securing Multi-Agent LLM Systems for Clinical
Decision-Making Against Prompt Injection* (Veeti Nurmikoski, Tampere University, 2026).

The thesis takes MDAgents, a multi-agent system that answers medical questions,
runs it on the open model MedGemma, attacks it through the reference passages
its agents read, and adds a CaMeL-style capability layer that stops text from
those passages from reaching other agents or the final decision. It asks:

1. **RQ1** How well does the capability layer stop injected content?
2. **RQ2** What does it cost in accuracy when there is no attack?
3. **RQ3** What does it cost in computation, time and money?

## Based on MDAgents

The multi-agent pipeline in `utils.py` is the MDAgents code by Kim et al.,
changed as described below. Original repository:
[mitmedialab/MDAgents](https://github.com/mitmedialab/MDAgents).

```bibtex
@misc{kim2024mdagentsadaptivecollaborationllms,
  title  = {MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making},
  author = {Yubin Kim and Chanwoo Park and Hyewon Jeong and Yik Siu Chan and Xuhai Xu and Daniel McDuff and Hyeonhoon Lee and Marzyeh Ghassemi and Cynthia Breazeal and Hae Won Park},
  year   = {2024}, eprint = {2404.15155}, archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2404.15155}
}
```

The protection layer adapts CaMeL (Debenedetti et al., 2025, arXiv:2503.18813).
The questions are MedQA (Jin et al., 2021; see `data/medqa/README.md`).

## Status

| Step | Status |
|---|---|
| MDAgents on MedGemma 4B (laptop) and 27B (Roihu GPU and CPU) | done |
| Baseline: single agent, adaptive, forced team, 500/200/100 questions | done |
| Two upstream defects found and repaired behind `--fixed` | done |
| Classifier routing with the released and the paper's prompt | done |
| Simulated retrieval and four attacks | done |
| Capability layer, smoke-tested on Roihu | done |
| Capability layer on the local 4B model | in progress |
| Full evaluation on the GPU (each attack, with and without the layer) | waiting for GPU time |

All runs and their numbers: [`results/README.md`](results/README.md).

## Files

```
main.py               experiment loop: one condition per run, paired, resumable
utils.py              MDAgents pipeline (upstream) + repairs, retrieval, defence hooks
capabilities.py       the protection layer: provenance, quarantine, policy
medgemma_backend.py   model calls (Ollama, vLLM or llama.cpp) and cost counters
answer.py             reads a stated "Answer: (X)" from free text
prepare_passages.py   BM25 retrieval of 3 textbook passages per question
make_attacks.py       the attacker's passage for each question and attack
analyze.py            accuracy, Wilson intervals, paired McNemar test
attack_report.py      attack success, flagged cases and cost against a clean run
classify_check.py     runs only the complexity classifier
scripts/              Roihu GPU/CPU jobs, local runs, local defence test, run tables
data/medqa/           the 1,273 MedQA-US test questions
results/              every run so far
```

## Switches

Every condition is a flag (or environment variable). With all of them off the
system behaves as published.

| Flag | Variable | Effect |
|---|---|---|
| `--difficulty` | | `adaptive` (classifier decides) or force `basic` / `intermediate` / `advanced` |
| `--fixed` | `MDAGENTS_FIXED=1` | repair the classifier parser and the empty moderator vote |
| `--classifier paper` | `MDAGENTS_CLASSIFIER` | the classifier prompt printed in the MDAgents paper |
| `--passages FILE` | `MDAGENTS_PASSAGES` | show retrieved passages to the answering agents |
| `--attack NAME` | `MDAGENTS_ATTACK` | `poison`, `inject`, `propagate` or `availability` replaces one passage |
| `--defence capability` | `MDAGENTS_DEFENCE` | enable the protection layer |

## Run on a laptop (Ollama)

```bash
export MEDGEMMA_BASE_URL=http://localhost:11434/v1 MEDGEMMA_MODEL=medgemma:4b \
       MEDGEMMA_API_KEY=ollama MEDGEMMA_MAX_TOKENS=2048 MDAGENTS_DATA=data

python3 main.py --difficulty basic --num_samples 10 --output output/basic10.json
python3 analyze.py output/basic10.json

bash scripts/local_defence.sh basic 50     # protection layer, single agent
bash scripts/local_defence.sh team 10      # protection layer, repaired team
bash scripts/local_defence.sh report
```

## Run on Roihu

`scripts/roihu.sh` (GPU, vLLM) and `scripts/roihu_cpu.sh` (CPU, llama.cpp;
one-time setup with `scripts/setup_cpu.sh`) are Slurm job scripts:

```bash
sbatch --time=12:00:00 scripts/roihu.sh basic 500 run1
EXTRA_ARGS="--fixed --passages data/medqa/passages.jsonl --attack inject --defence capability" \
  sbatch --time=24:00:00 scripts/roihu.sh intermediate 100 inject_def
```

## Changes from upstream

New files: everything except `utils.py`. `main.py` is rewritten.

In `utils.py`:

1. Every model call goes to MedGemma; upstream hard-codes GPT models in several places.
2. Recruiter and team output is parsed tolerantly (upstream crashes on format
   deviations); a default roster is used and recorded if parsing fails.
3. `TRACE` records what happens inside each question (roster, talk replies,
   moderator input, marker spread, defence decision).
4. `--fixed` repairs two defects: the classifier parser returns `basic` whenever
   the word appears anywhere in the reply (98% of questions), and when no expert
   asks to talk in round 1 the moderator votes over `None` (70% of team questions).
5. Retrieval (`reference_block`) and the defence hooks (`extract_choice`,
   `defended_answer`, the defended branches) are added behind their switches.

Left as published: `round_opinions` is filled only for round 1, and the advanced
path passes only the initial team's report to the final decision maker.

## Known limits

- No training split, so no few-shot examples: prompts differ from the published setup.
- One seed. Repeating G1 changed 27 of 500 answers (serving noise), so small
  differences between conditions are not effects.
- The advanced path has no retrieval or defence yet.
