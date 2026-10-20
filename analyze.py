"""Score one run, or compare two runs question by question.

    python analyze.py results/local_4b/basic.json
    python analyze.py results/local_4b/basic.json results/local_4b/adaptive.json

With two files, only the questions both runs answered are compared (paired),
using McNemar's exact test. Accuracy comes with a Wilson 95% interval.
"""

import json
import math
import sys
from collections import Counter

from answer import final_letter


def wilson(correct, n, z=1.96):
    """Wilson score interval for a proportion, in percent."""
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centre - half), 100 * (centre + half)


def mcnemar_exact(only_a, only_b):
    """Two-sided exact test on the questions where the two runs disagree."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(path):
    """{question index: record}, with the scored letter added."""
    records = {}
    for r in json.load(open(path)):
        r['letter'] = final_letter(r)
        r['correct'] = r['letter'] == r['label']
        records[r['index']] = r
    return records


def report(path, records):
    n = len(records)
    correct = sum(r['correct'] for r in records.values())
    low, high = wilson(correct, n)
    print(path)
    print(f"  n = {n}   accuracy {correct}/{n} = {100 * correct / n:.1f}%"
          f"   95% CI [{low:.1f}, {high:.1f}]")
    print(f"  unscored (no letter found): {sum(r['letter'] is None for r in records.values())}")
    print(f"  letter obtained: {dict(Counter(r.get('picked_via', '?') for r in records.values()))}")

    paths = Counter(r['difficulty'] for r in records.values())
    if len(paths) > 1:
        print("  by path:")
        for name in ('basic', 'intermediate', 'advanced'):
            group = [r for r in records.values() if r['difficulty'] == name]
            if group:
                k = sum(r['correct'] for r in group)
                print(f"    {name:<13} {k}/{len(group)} = {100 * k / len(group):.1f}%")

    traces = [r['trace'] for r in records.values() if r.get('trace')]
    if traces:
        none = sum(t.get('moderator_got_none', False) for t in traces)
        print(f"  intermediate path: moderator given no agent answers in "
              f"{none}/{len(traces)} questions")
        fallback = sum(t.get('default_roster', False) for t in traces)
        if fallback:
            print(f"  default roster used (recruiter output unparsable): {fallback}")
    print()


def compare(path_a, a, path_b, b):
    shared = sorted(set(a) & set(b))
    both = sum(a[i]['correct'] and b[i]['correct'] for i in shared)
    only_a = sum(a[i]['correct'] and not b[i]['correct'] for i in shared)
    only_b = sum(b[i]['correct'] and not a[i]['correct'] for i in shared)
    neither = len(shared) - both - only_a - only_b
    same = sum(a[i]['letter'] == b[i]['letter'] for i in shared)

    print(f"Paired comparison over the {len(shared)} questions both runs answered")
    print(f"  A = {path_a}\n  B = {path_b}")
    for name, records in (('A', a), ('B', b)):
        k = sum(records[i]['correct'] for i in shared)
        low, high = wilson(k, len(shared))
        print(f"  {name} accuracy on these: {k}/{len(shared)} = "
              f"{100 * k / len(shared):.1f}%  [{low:.1f}, {high:.1f}]")
    print(f"  both correct {both}   only A {only_a}   only B {only_b}   neither {neither}")
    print(f"  identical answers: {same}/{len(shared)} ({100 * same / len(shared):.0f}%)")
    print(f"  McNemar exact p = {mcnemar_exact(only_a, only_b):.3f}")


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    runs = [(path, load(path)) for path in sys.argv[1:]]
    for path, records in runs:
        report(path, records)
    if len(runs) == 2:
        compare(runs[0][0], runs[0][1], runs[1][0], runs[1][1])
