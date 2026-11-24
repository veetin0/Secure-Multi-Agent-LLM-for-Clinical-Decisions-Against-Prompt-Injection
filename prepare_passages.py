"""Retrieve benign reference passages for each MedQA question.

The thesis simulates the retrieval step of MDAgents + MedRAG: each question is
shown with the passages a retriever would return from the textbook corpus that
ships with MedQA. This script does that retrieval once, offline, so every
experimental condition sees exactly the same passages.

Corpus: MedRAG's chunked copy of the 18 MedQA textbooks (Hugging Face dataset
MedRAG/textbooks). Retriever: BM25 (k1=1.2, b=0.75) over question + options.

Run once on the Roihu login node (CPU only, a few minutes):

    module load python-vllm/0.19
    python3 prepare_passages.py --num_samples 500

Writes data/medqa/passages.jsonl, one line per question:
    {"index": 0, "passages": [{"id", "title", "text", "score"}, ...]}
"""

import argparse
import glob
import json
import math
import os
import re
from collections import Counter

STOP = set("""a an the of and or in on at to for with by from is are was were be been
being this that these those which who whom what when where how why it its as
than then there their they them he she his her him patient patients most
likely following year old man woman presents shows""".split())


def tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in STOP and len(t) > 1]


def corpus_files(cache):
    from huggingface_hub import snapshot_download
    root = snapshot_download("MedRAG/textbooks", repo_type="dataset",
                             allow_patterns=["*.jsonl"], cache_dir=cache)
    files = sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True))
    if not files:
        raise SystemExit(f"no .jsonl files found under {root}")
    return files


def read_corpus(files):
    for path in files:
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    text = d.get("content") or d.get("contents") or ""
                    yield d.get("id", ""), d.get("title", os.path.basename(path)), text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--data_root", default="data")
    p.add_argument("--k", type=int, default=5, help="passages kept per question")
    p.add_argument("--max_words", type=int, default=120,
                   help="passages are cut to this many words")
    p.add_argument("--cache", default=os.environ.get("HF_HOME"))
    p.add_argument("--output", default=None)
    args = p.parse_args()

    test = [json.loads(l) for l in open(os.path.join(args.data_root, "medqa", "test.jsonl"))
            if l.strip()][:args.num_samples]
    queries = [Counter(tokens(q["question"] + " " + " ".join(q["options"].values())))
               for q in test]
    vocab = set().union(*queries)

    files = corpus_files(args.cache)
    postings = {w: [] for w in vocab}
    lengths, meta = [], []
    for doc_id, title, text in read_corpus(files):
        toks = tokens(text)
        n = len(lengths)
        lengths.append(len(toks))
        meta.append((doc_id, title, text))
        for w, tf in Counter(t for t in toks if t in vocab).items():
            postings[w].append((n, tf))
    N, avg = len(lengths), sum(lengths) / max(1, len(lengths))
    print(f"corpus: {N} chunks from {len(files)} files, {len(vocab)} query words")

    k1, b = 1.2, 0.75
    out = args.output or os.path.join(args.data_root, "medqa", "passages.jsonl")
    with open(out, "w") as f:
        for index, q in enumerate(queries):
            scores = Counter()
            for w in q:
                plist = postings[w]
                if not plist:
                    continue
                idf = math.log(1 + (N - len(plist) + 0.5) / (len(plist) + 0.5))
                for n, tf in plist:
                    scores[n] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[n] / avg))
            top = []
            for n, score in scores.most_common(args.k):
                doc_id, title, text = meta[n]
                words = text.split()
                text = " ".join(words[:args.max_words]) + (" ..." if len(words) > args.max_words else "")
                top.append({"id": doc_id, "title": title, "text": text, "score": round(score, 2)})
            f.write(json.dumps({"index": index, "passages": top}) + "\n")
    print(f"wrote {len(queries)} questions to {out}")


if __name__ == "__main__":
    main()
