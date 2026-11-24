"""Build the LaTeX tables of every run from the result files.

    python3 scripts/run_tables.py > results/run_tables.tex

Every number in the thesis appendix comes from here, so rerun it when a new
run is added (and add the run to RUNS below).
"""
import json
import math
import os
import sys
from collections import Counter

R = 'results'
# id, file, model/hardware, path, variant
RUNS = [
    ('G1', 'roihu_27b/roihu_27b_basic_seed0.json', '27B GPU', 'single agent', 'published'),
    ('G2', 'roihu_27b/roihu_27b_basic_run2_seed0.json', '27B GPU', 'single agent', 'published (repeat)'),
    ('G3', 'roihu_27b/roihu_27b_adaptive_seed0.json', '27B GPU', 'adaptive', 'published'),
    ('G4', 'roihu_27b/roihu_27b_intermediate_seed0.json', '27B GPU', 'team (forced)', 'published'),
    ('G5', 'roihu_27b/roihu_27b_intermediate_diag_seed0.json', '27B GPU', 'team (forced)', 'published (traced)'),
    ('G6', 'roihu_27b/roihu_27b_intermediate_fixed_seed0.json', '27B GPU', 'team (forced)', 'repaired'),
    ('G7', 'roihu_27b/roihu_27b_adaptive_fixed_seed0.json', '27B GPU', 'adaptive', 'repaired'),
    ('L1', 'local_4b/basic.json', '4B laptop', 'single agent', 'published'),
    ('L2', 'local_4b/adaptive.json', '4B laptop', 'adaptive', 'published'),
]
PAIRS = [('G1', 'G2'), ('G1', 'G3'), ('G1', 'G4'), ('G4', 'G6'), ('G1', 'G6'),
         ('G3', 'G7'), ('G1', 'G7'), ('L1', 'L2'), ('G1', 'L1')]


def load(path):
    return {r['index']: r for r in json.load(open(os.path.join(R, path)))}


def ok(r):
    return r.get('picked_letter') == r['label']


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def pct(k, n):
    return f'{100 * k / n:.1f}' if n else '--'


def fmt_p(p):
    return '$<0.001$' if p < 0.001 else f'{p:.3f}'


def table(caption, label, cols, header, rows, note=None):
    out = ['\\begin{table}[H]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{4pt}',
           f'\\caption{{{caption}}}', f'\\label{{{label}}}',
           f'\\begin{{tabular}}{{{cols}}}', '\\toprule', header + ' \\\\', '\\midrule']
    out += [' & '.join(r) + ' \\\\' for r in rows]
    out += ['\\bottomrule', '\\end{tabular}']
    if note:
        out += ['\\par\\smallskip', f'\\parbox{{\\linewidth}}{{\\footnotesize {note}}}']
    out += ['\\end{table}', '']
    return '\n'.join(out)


data = {rid: load(f) for rid, f, *_ in RUNS}
meta = {rid: (hw, path, var) for rid, _, hw, path, var in RUNS}
tex = []

# 1. run register
rows = []
for rid, _, hw, path, var in RUNS:
    d = data[rid]
    n, k = len(d), sum(ok(r) for r in d.values())
    lo, hi = wilson(k, n)
    rows.append([rid, hw, path, var, str(n), f'{100 * k / n:.1f} [{lo:.1f}, {hi:.1f}]'])
tex.append(table('Every answering run: accuracy on MedQA with a 95\\% Wilson interval.',
                 'tab:runs_register', 'llllrl',
                 'Run & Model & Path & Variant & Questions & Accuracy (\\%)', rows,
                 'Every run uses the first $n$ questions of the MedQA-US test set, seed 0, '
                 'shuffled options and no few-shot examples. A question without an answer '
                 'letter (context overflow, unparsed reply) counts as wrong.'))

# 2. routing
rows = []
for rid in ('G3', 'G7', 'L2'):
    d = data[rid]
    c = Counter(r['difficulty'] for r in d.values())
    n = len(d)
    parser = 'released code' if rid != 'G7' else 'released code, repaired parser'
    rows.append([rid, parser, str(n)] + [f"{c.get(p, 0)} ({pct(c.get(p, 0), n)})"
                                          for p in ('basic', 'intermediate', 'advanced')])
cl = json.load(open(os.path.join(R, 'roihu_27b/roihu_27b_adaptive_classify_seed0.json')))
for key, name in (('upstream_label', 'released code'), ('forced_label', 'released code, repaired parser')):
    c = Counter(r[key] for r in cl)
    rows.append(['G8', name, str(len(cl))] + [f"{c.get(p, 0)} ({pct(c.get(p, 0), len(cl))})"
                                              for p in ('basic', 'intermediate', 'advanced')])
for f, rid in (('routing/cpu_27b_adaptive_paperclf_compare_seed0.json', 'C2'),
               ('routing/cpu_27b_adaptive_paperclf_seed0.json', 'C3')):
    pc = json.load(open(os.path.join(R, f)))
    c = Counter(r['paper_path'] for r in pc)
    rows.append([rid, 'paper prompt', str(len(pc))] + [f"{c.get(p, 0)} ({pct(c.get(p, 0), len(pc))})"
                                                       for p in ('basic', 'intermediate', 'advanced')])
    if 'forced_label' in pc[0]:
        c = Counter(r['forced_label'] for r in pc)
        rows.append([rid, 'released code, repaired parser', str(len(pc))] +
                    [f"{c.get(p, 0)} ({pct(c.get(p, 0), len(pc))})" for p in ('basic', 'intermediate', 'advanced')])
tex.append(table('Where the complexity classifier sends questions, by prompt and parser: number of questions (\\%).',
                 'tab:runs_routing', 'lllrrr',
                 'Run & Prompt and parser & $n$ & Basic & Intermediate & Advanced', rows,
                 'G8, C2 and C3 run only the classifier. G8 parses each reply twice: as the released code '
                 'does, and by asking the same agent for its choice in one word. C2 and C3 run on the CPU '
                 'partition; C3 stopped after 77 of 200 questions.'))

# 3. team behaviour
rows = []
for rid in ('G3', 'G4', 'G5', 'G6', 'G7'):
    d = data[rid]
    team = [r for r in d.values() if r['difficulty'] == 'intermediate']
    t = [r.get('trace', {}) for r in team]
    over = sum(r.get('picked_via') == 'context_overflow' for r in team)
    none = sum(1 for x in t if x.get('moderator_got_none'))
    r1 = sum(1 for x in t if x.get('moderator_given_round1'))
    talk = [y for x in t for y in x.get('talk_replies', [])]
    yes = sum('yes' in y['reply'].lower() for y in talk)
    rounds = Counter(x.get('rounds_with_answers') for x in t if 'rounds_with_answers' in x)
    answered = sum(v for k, v in rounds.items() if k)
    rows.append([rid, meta[rid][2], str(len(team)), f'{none} ({pct(none, len(team))})', str(r1),
                 f'{yes}/{len(talk)}' if talk else '--', str(answered), str(over)])
tex.append(table('Behaviour of the team on the questions routed or forced to it.',
                 'tab:runs_team', 'llrrrrrr',
                 "Run & Variant & $n$ & Empty vote & Round 1 & Talk `yes' & Debated & Overflow", rows,
                 "$n$: questions that reached the team. Empty vote: no expert asked to talk in round 1, so "
                 "the loop ended without final answers and the moderator voted over \\texttt{None}. "
                 "Round 1: the repair gave the moderator the round-1 opinions instead. Talk `yes': "
                 "replies to `do you want to talk to any expert?', recorded from G5 on. Debated: "
                 "questions with at least one completed debate round. Overflow: the conversation exceeded "
                 "the context window."))

# 4. paired comparisons
rows = []
for a, b in PAIRS:
    A, B = data[a], data[b]
    s = sorted(set(A) & set(B))
    oa = sum(ok(A[i]) and not ok(B[i]) for i in s)
    ob = sum(ok(B[i]) and not ok(A[i]) for i in s)
    same = sum(A[i].get('picked_letter') == B[i].get('picked_letter') for i in s)
    rows.append([f'{a} vs {b}', str(len(s)), pct(sum(ok(A[i]) for i in s), len(s)),
                 pct(sum(ok(B[i]) for i in s), len(s)), str(oa), str(ob),
                 pct(same, len(s)), fmt_p(mcnemar(oa, ob))])
tex.append(table('Paired comparisons on the questions both runs answered.',
                 'tab:runs_pairs', 'lrrrrrrr',
                 'Runs & $n$ & Acc.\\ 1st (\\%) & Acc.\\ 2nd (\\%) & Only 1st & Only 2nd & Same (\\%) & $p$',
                 rows, "Only 1st: correct in the first run and wrong in the second. Same: identical answer letter. "
                 "$p$: exact two-sided McNemar test on the discordant pairs."))

# 5. answer extraction
rows = []
for rid, *_ in RUNS:
    d = data[rid]
    c = Counter(r.get('picked_via') for r in d.values())
    if None in c:
        continue
    n = len(d)
    other = n - c['stated'] - c['model']
    rows.append([rid, str(n), f"{c['stated']} ({pct(c['stated'], n)})", f"{c['model']} ({pct(c['model'], n)})",
                 str(c['unparsed']), str(c['context_overflow']), str(other - c['unparsed'] - c['context_overflow'])])
tex.append(table('How the answer letter was obtained from the system\'s final text.',
                 'tab:runs_extraction', 'lrrrrrr',
                 'Run & $n$ & Stated in text & Extraction call & Unparsed & Overflow & Other', rows,
                 'The local runs L1 and L2 predate this field.'))

print('% Generated by scripts/run_tables.py -- do not edit by hand.\n')
print('\n'.join(tex))
