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
    return p.parse_args()
 
 
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
 
    setup_model(args.model)
    test_qa, examplers = load_data(args.dataset, args.data_root)
 
    os.makedirs('output', exist_ok=True)
    suffix = '_noshuffle' if args.no_shuffle else ''
    out_path = args.output or (f'output/{args.model}_{args.dataset}_'
                               f'{args.difficulty}_seed{args.seed}{suffix}.json')
 
    results = []
    if args.resume and os.path.exists(out_path):
        results = json.load(open(out_path))
        print(f'[resume] {len(results)} already done in {out_path}')
    done = {r['index'] for r in results}
 
    start = time.time()
    failures = 0
    out_of_time = False
    for index, sample in enumerate(tqdm(test_qa[:args.num_samples])):
        if index in done:
            continue

        if args.time_budget and time.time() - start > args.time_budget:
            out_of_time = True
            print(f'\n[budget] stopping at question {index}; use --resume to continue')
            break
 
        random.seed(args.seed * 100000 + index)
        utils.TRACE.clear()
 
        question, _ = create_question(sample, args.dataset)
        difficulty = determine_difficulty(question, args.difficulty, model=args.model)
        try:
            decision = answer(question, difficulty, examplers, args)
            letter, how = mg.pick_letter(decision, sample['options'])
            failures = 0
        except openai.APIConnectionError:
            raise
        except Exception as error:
            failures += 1
            if failures >= 3:
                raise
            kind = ('context_overflow' if 'context length' in str(error)
                    else f'error:{type(error).__name__}')
            print(f'\n[error] question {index}: {kind}: {str(error)[:200]}')
            decision, letter, how = '', None, kind
 
        results.append({
            'index': index,
            'question': question,
            'label': sample['answer_idx'],
            'answer': sample['answer'],
            'options': sample['options'],
            'difficulty': difficulty,
            'response': decision,
            'picked_letter': letter,
            'picked_via': how,
            'trace': dict(utils.TRACE),
            'truncated_calls': mg.TRUNCATED,
            'shuffled_options': not args.no_shuffle,
            'presented_order': re.findall(r'\(([A-E])\)',
                                          question.split('Options:')[-1]),
        })
        json.dump(results, open(out_path, 'w'), indent=4)
 
    new = len(results) - len(done)
    print(f'\nWrote {len(results)} results to {out_path}')
    if new:
        seconds = time.time() - start
        print(f'{new} new in {seconds/60:.0f} min ({seconds/new:.1f} s/question)')
    if out_of_time:
        print(f'Stopped on the time budget: {len(results)}/{args.num_samples} done.')
 
 
if __name__ == '__main__':
    main()