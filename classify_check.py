"""Why does the classifier send almost every question to the single agent?

For each question this asks MDAgents' complexity classifier twice and saves
both raw replies:

  1. exactly as MDAgents asks it (same prompt, same agent set-up), together
     with the label MDAgents' parser takes from the reply;
  2. the same prompt plus "answer with one word", which shows the label the
     model actually chooses.

If the two disagree, the routing comes from the parser, not from the model.

With --prompt paper it instead asks the prompt printed in the MDAgents paper
(low/moderate/high, ending in "**Answer:**(") and records the path it gives,
using the same parsing as main.py --classifier paper. --compare also asks the
released code's prompt with a one-word answer, on the same server, so both
prompts are measured on the same model build.
Takes the same arguments as main.py, so scripts/roihu.sh can run it:

    PY_SCRIPT=classify_check.py sbatch scripts/roihu.sh adaptive 200 classify
"""

import argparse
import json
import os
import random
import re
from collections import Counter

from tqdm import tqdm

import utils
from utils import Agent, setup_model, load_data, create_question

LABELS = ('basic', 'intermediate', 'advanced')
INSTRUCTION = ('You are a medical expert who conducts initial assessment and your '
               'job is to decide the difficulty/complexity of the medical query.')


def prompt_for(question):
    return (f"Now, given the medical query as below, you need to decide the "
            f"difficulty/complexity of it:\n{question}.\n\nPlease indicate the "
            f"difficulty/complexity of the medical query among below options:\n"
            f"1) basic: a single medical agent can output an answer.\n"
            f"2) intermediate: number of medical experts with different expertise "
            f"should dicuss and make final decision.\n"
            f"3) advanced: multiple teams of clinicians from different departments "
            f"need to collaborate with each other to make final decision.")


def upstream_label(reply):
    r = reply.lower()
    if 'basic' in r or '1)' in r:
        return 'basic'
    if 'intermediate' in r or '2)' in r:
        return 'intermediate'
    if 'advanced' in r or '3)' in r:
        return 'advanced'
    return 'basic'


def labels_mentioned(reply):
    """Labels in the order they first appear in the reply."""
    found = []
    for m in re.finditer(r'\b(basic|intermediate|advanced)\b', reply.lower()):
        if m.group(1) not in found:
            found.append(m.group(1))
    return found


def ask(model, message, first_message=True):
    agent = Agent(instruction=INSTRUCTION, role='medical expert', model_info=model)
    agent.chat(INSTRUCTION) 
    return agent.chat(message)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='medqa')
    p.add_argument('--model', default='medgemma')
    p.add_argument('--difficulty', default='adaptive')
    p.add_argument('--num_samples', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data_root', default=None)
    p.add_argument('--output', default='output/classify_check.json')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--prompt', choices=['code', 'paper'], default='code')
    p.add_argument('--compare', action='store_true',
                   help="with --prompt paper: also ask the code prompt, one-word answer")
    args = p.parse_args()

    setup_model(args.model)
    test_qa, _ = load_data(args.dataset, args.data_root)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    rows = []
    if args.resume and os.path.exists(args.output):
        rows = json.load(open(args.output))
    done = {r['index'] for r in rows}

    for index, sample in enumerate(tqdm(test_qa[:args.num_samples])):
        if index in done:
            continue
        random.seed(args.seed * 100000 + index)
        question, _ = create_question(sample, args.dataset)

        if args.prompt == 'paper':
            rows.append(paper_row(index, question, args))
            json.dump(rows, open(args.output, 'w'), indent=2)
            continue

        reply = ask(args.model, prompt_for(question))
        forced = ask(args.model, prompt_for(question) +
                     "\n\nAnswer with exactly one word: basic, intermediate or advanced.")
        chosen = labels_mentioned(forced)

        rows.append({
            'index': index,
            'reply': reply,
            'upstream_label': upstream_label(reply),
            'labels_mentioned': labels_mentioned(reply),
            'forced_reply': forced,
            'forced_label': chosen[0] if chosen else None,
        })
        json.dump(rows, open(args.output, 'w'), indent=2)

    summarise_paper(rows) if args.prompt == 'paper' else summarise(rows)


def paper_row(index, question, args):
    utils.TRACE.clear()
    agent = Agent(instruction=INSTRUCTION, role='medical expert', model_info=args.model)
    agent.chat(INSTRUCTION)
    reply = agent.chat(utils.PAPER_CLASSIFIER_PROMPT.format(question=question))
    path = utils.paper_difficulty(agent, reply)
    row = {'index': index, 'paper_reply': reply, 'paper_path': path,
           'parsed_from': utils.TRACE.get('classifier_parsed_from'),
           'choice_reply': utils.TRACE.get('classifier_choice_reply')}
    if args.compare:
        forced = ask(args.model, prompt_for(question) +
                     "\n\nAnswer with exactly one word: basic, intermediate or advanced.")
        chosen = labels_mentioned(forced)
        row.update(forced_reply=forced, forced_label=chosen[0] if chosen else None)
    return row


def summarise_paper(rows):
    n = len(rows)
    print(f'\n{n} questions, paper prompt')
    paths = Counter(r['paper_path'] for r in rows)
    print('paper prompt routes to:', {k: f'{v} ({v / n:.0%})' for k, v in paths.items()})
    print('parsed from           :', dict(Counter(r['parsed_from'] for r in rows)))
    if any('forced_label' in r for r in rows):
        print('code prompt, one word :', dict(Counter(r.get('forced_label') for r in rows)))
        print('(paper, code)         :', dict(Counter((r['paper_path'], r.get('forced_label')) for r in rows)))
    words = sorted(len(r['paper_reply'].split()) for r in rows)
    print(f'paper reply length in words: median {words[n // 2] if n else 0}, max {words[-1] if n else 0}')


def summarise(rows):
    n = len(rows)
    print(f'\n{n} questions')
    print('MDAgents routes to :', dict(Counter(r['upstream_label'] for r in rows)))
    print('model chooses      :', dict(Counter(r['forced_label'] for r in rows)))
    both = Counter((r['upstream_label'], r['forced_label']) for r in rows)
    print('(routed, chosen)   :', dict(both))
    lists_all = sum(1 for r in rows if len(r['labels_mentioned']) >= 2)
    print(f'replies that mention two or more labels: {lists_all}/{n}')
    print(f"replies containing '1)': {sum('1)' in r['reply'] for r in rows)}/{n}")
    words = sorted(len(r['reply'].split()) for r in rows)
    print(f'reply length in words: median {words[n // 2] if n else 0}, max {words[-1] if n else 0}')


if __name__ == '__main__':
    main()
