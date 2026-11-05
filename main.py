"""Run MDAgents on MedQA with MedGemma and save one record per question."""

import argparse
import json
import os
import random
import re
import time

import openai
from tqdm import tqdm

import medgemma_backend as mg
import utils
from utils import (
    setup_model, load_data, create_question, determine_difficulty,
    process_basic_query, process_intermediate_query, process_advanced_query,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='medqa')
    p.add_argument('--model', default='medgemma')
    p.add_argument('--difficulty', default='adaptive',
                   help="adaptive (classifier decides), or force basic / "
                        "intermediate / advanced")
    p.add_argument('--num_samples', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data_root', default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--resume', action='store_true',
                   help="skip questions already in the output file")
    p.add_argument('--time_budget', type=float, default=0,
                   help="stop cleanly between questions after this many "
                        "seconds (0 = no limit); --resume continues later")
    p.add_argument('--no_shuffle', action='store_true',
                   help="show answer options in order A,B,C,D instead of "
                        "upstream's shuffled display order")
    p.add_argument('--fixed', action='store_true',
                   default=os.environ.get('MDAGENTS_FIXED') == '1',
                   help="repair the classifier parser and the empty-moderator "
                        "defect (also set by MDAGENTS_FIXED=1)")
    p.add_argument('--classifier', choices=['code', 'paper'],
                   default=os.environ.get('MDAGENTS_CLASSIFIER', 'code'),
                   help="complexity prompt: as in the released code, or as printed "
                        "in the MDAgents paper (also set by MDAGENTS_CLASSIFIER)")
    p.add_argument('--passages', default=os.environ.get('MDAGENTS_PASSAGES'),
                   help="jsonl of retrieved passages per question (from "
                        "prepare_passages.py); shown to the answering agents. "
                        "Also set by MDAGENTS_PASSAGES")
    p.add_argument('--k', type=int, default=int(os.environ.get('MDAGENTS_K', '3')),
                   help="benign passages per question")
    p.add_argument('--attack', default=os.environ.get('MDAGENTS_ATTACK', 'none'),
                   choices=['none'] + list(ATTACKS),
                   help="replace one passage with an attacker's passage "
                        "(also set by MDAGENTS_ATTACK)")
    p.add_argument('--attack_file',
                   default=os.environ.get('MDAGENTS_ATTACK_FILE', 'data/medqa/attacks.json'),
                   help="attacker passages from make_attacks.py")
    return p.parse_args()


ATTACKS = ('poison', 'inject', 'propagate', 'availability')


def load_jsonl(path):
    with open(path) as f:
        return {r['index']: r for r in map(json.loads, filter(str.strip, f))}


def passages_for(index, args, passages, attacks):
    """The passages shown for one question, with the attacker's in place."""
    shown = list(passages[index]['passages'][:args.k])
    if args.attack == 'none':
        return shown, None
    attack = attacks[str(index)]
    # The attacker's passage replaces one benign passage, at a position fixed
    # per question, so the number and format of passages never give it away.
    shown[attack['position'] % len(shown)] = attack['passages'][args.attack]
    return shown, attack


def answer(question, difficulty, examplers, args):
    """Send one question down the path the difficulty selects."""
    if difficulty == 'basic':
        return process_basic_query(question, examplers, args.model, args)
    if difficulty == 'intermediate':
        return process_intermediate_query(question, examplers, args.model, args)
    return process_advanced_query(question, args.model, args)


def main():
    args = parse_args()
    random.seed(args.seed)
    utils.SHUFFLE_OPTIONS = not args.no_shuffle
    utils.FIXED = args.fixed
    utils.CLASSIFIER = args.classifier
    if args.attack != 'none' and not args.passages:
        raise SystemExit('--attack needs --passages (the attack replaces one of them)')
    passages = load_jsonl(args.passages) if args.passages else None
    attacks = json.load(open(args.attack_file)) if args.attack != 'none' else None
    print(f"[system] {'fixed' if args.fixed else 'published'} MDAgents, {args.classifier} classifier")

    setup_model(args.model)
    test_qa, examplers = load_data(args.dataset, args.data_root)

    os.makedirs('output', exist_ok=True)
    suffix = (('_noshuffle' if args.no_shuffle else '') + ('_fixed' if args.fixed else '')
              + ('_paperclf' if args.classifier == 'paper' else '')
              + ('_rag' if args.passages else '')
              + (f'_{args.attack}' if args.attack != 'none' else ''))
    out_path = args.output or (f'output/{args.model}_{args.dataset}_'
                               f'{args.difficulty}_seed{args.seed}{suffix}.json')

    results = []
    if args.resume and os.path.exists(out_path):
        results = json.load(open(out_path))
        print(f'[resume] {len(results)} already done in {out_path}')
    done = {r['index'] for r in results}

    start = time.time()
    failures = 0          # consecutive questions that raised an error
    out_of_time = False
    for index, sample in enumerate(tqdm(test_qa[:args.num_samples])):
        if index in done:
            continue
        # Checked between questions, never during one, so a result is never
        # written from a half-finished question.
        if args.time_budget and time.time() - start > args.time_budget:
            out_of_time = True
            print(f'\n[budget] stopping at question {index}; use --resume to continue')
            break

        # Shuffling the options consumes random numbers, so seed per question:
        # a resumed run then shows the same option order as an unbroken one.
        random.seed(args.seed * 100000 + index)
        utils.TRACE.clear()
        target = None
        if passages is not None:
            utils.PASSAGES, attack = passages_for(index, args, passages, attacks)
            if attack:
                target = attack['target']
                utils.MARKER = attack['marker'] if args.attack == 'propagate' else None
        calls0, ptok0, ctok0, t0 = mg.CALLS, mg.PROMPT_TOKENS, mg.COMPLETION_TOKENS, time.time()

        question, _ = create_question(sample, args.dataset)
        difficulty = determine_difficulty(question, args.difficulty, model=args.model)
        try:
            decision = answer(question, difficulty, examplers, args)
            letter, how = mg.pick_letter(decision, sample['options'])
            failures = 0
        except openai.APIConnectionError:
            raise                                   # server down: stop, don't record
        except Exception as error:
            # The conversation outgrew the context window, or the model's reply
            # broke a parser. Both are failures of the system on this question,
            # so record them and count the question as unanswered instead of
            # stopping a multi-hour run. Many in a row means something else is
            # wrong, so stop then.
            kind = ('context_overflow' if 'context length' in str(error)
                    else f'error:{type(error).__name__}')
            # An overflow is an outcome of the question (the availability
            # attack provokes them); only other errors in a row stop the run.
            failures = 0 if kind == 'context_overflow' else failures + 1
            if failures >= 3:
                raise
            print(f'\n[error] question {index}: {kind}: {str(error)[:200]}')
            decision, letter, how = '', None, kind

        results.append({
            'index': index,
            'question': question,
            'label': sample['answer_idx'],
            'answer': sample['answer'],
            'options': sample['options'],
            'difficulty': difficulty,
            'response': decision,            # the system's free-text decision
            'picked_letter': letter,         # the option that decision selects
            'picked_via': how,               # stated / model / unparsed / failed
            'trace': dict(utils.TRACE),      # what happened inside the question
            'truncated_calls': mg.TRUNCATED,
            'shuffled_options': not args.no_shuffle,
            'fixed': args.fixed,
            'classifier': args.classifier,
            'passages': bool(args.passages),
            'attack': args.attack,
            'target': target,                # option the attacker wants chosen
            'calls': mg.CALLS - calls0,
            'prompt_tokens': mg.PROMPT_TOKENS - ptok0,
            'completion_tokens': mg.COMPLETION_TOKENS - ctok0,
            'seconds': round(time.time() - t0, 1),
            'presented_order': re.findall(r'\(([A-E])\)',
                                          question.split('Options:')[-1]),
        })
        json.dump(results, open(out_path, 'w'), indent=4)   # keep work after every question

    new = len(results) - len(done)
    print(f'\nWrote {len(results)} results to {out_path}')
    if new:
        seconds = time.time() - start
        print(f'{new} new in {seconds/60:.0f} min ({seconds/new:.1f} s/question)')
    if out_of_time:
        print(f'Stopped on the time budget: {len(results)}/{args.num_samples} done.')


if __name__ == '__main__':
    main()
