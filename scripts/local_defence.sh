#!/usr/bin/env bash
# Test the protection layer (capabilities.py) with the local MedGemma 4B.
#
#   bash scripts/local_defence.sh basic 50     # single-agent path, about 1.5 h
#   bash scripts/local_defence.sh team 10      # team path (--fixed), overnight
#   bash scripts/local_defence.sh report       # print the results
#
# Every condition is run twice: without and with --defence capability.
# Needs data/medqa/passages.jsonl and data/medqa/attacks.json (copy them from
# Roihu so the attacks are the same as in the GPU evaluation).
# Interrupted? Run the same command again: finished questions are skipped.

set -euo pipefail
cd "$(dirname "$0")/.."

STAGE=${1:-basic}
N=${2:-}
OUT=results/local_4b_defence

export MEDGEMMA_BASE_URL=${MEDGEMMA_BASE_URL:-http://localhost:11434/v1}
export MEDGEMMA_MODEL=${MEDGEMMA_MODEL:-medgemma:4b}
export MEDGEMMA_API_KEY=${MEDGEMMA_API_KEY:-ollama}
export MEDGEMMA_MAX_TOKENS=${MEDGEMMA_MAX_TOKENS:-2048}
export MDAGENTS_DATA=${MDAGENTS_DATA:-data}
export MDAGENTS_PASSAGES=data/medqa/passages.jsonl

report() {
  for path in basic team; do
    [ -f "$OUT/${path}_clean.json" ] || continue
    echo "=================== $path path, undefended ==================="
    python3 attack_report.py "$OUT/${path}_clean.json" $(ls "$OUT"/${path}_{poison,inject,propagate,availability}.json 2>/dev/null)
    echo "=================== $path path, defended ==================="
    python3 attack_report.py "$OUT/${path}_clean_capability.json" $(ls "$OUT"/${path}_{poison,inject,propagate,availability}_capability.json 2>/dev/null)
    echo "=================== $path path, cost of the layer (clean vs clean defended) ==================="
    python3 attack_report.py "$OUT/${path}_clean.json" "$OUT/${path}_clean_capability.json"
  done
}
if [ "$STAGE" = report ]; then report; exit 0; fi

for f in data/medqa/passages.jsonl data/medqa/attacks.json; do
  [ -f "$f" ] || { echo "Missing $f. Copy it from Roihu first (see results/README.md)."; exit 1; }
done
curl -sf "${MEDGEMMA_BASE_URL%/v1}/api/tags" > /dev/null \
  || { echo "No Ollama server at ${MEDGEMMA_BASE_URL%/v1}. Start the Ollama app or run: ollama serve"; exit 1; }
mkdir -p "$OUT"

case "$STAGE" in
  basic) DIFF=basic;        N=${N:-50}; ATTACKS="none poison inject propagate"; FIXED=0 ;;
  team)  DIFF=intermediate; N=${N:-10}; ATTACKS="none poison inject propagate availability"; FIXED=1 ;;
  *) echo "Usage: $0 basic|team [questions] | report"; exit 1 ;;
esac
export MDAGENTS_FIXED=$FIXED

run() {   # run <attack> <defence>
  local name=$1; [ "$1" = none ] && name=clean
  local tag=""; [ "$2" = capability ] && tag=_capability
  local out="$OUT/${STAGE}_${name}${tag}.json"
  echo; echo ">>> $STAGE path, attack $1, defence $2  ->  $out"
  MDAGENTS_ATTACK=$1 MDAGENTS_DEFENCE=$2 \
    python3 main.py --difficulty "$DIFF" --num_samples "$N" --output "$out" --resume
}

for attack in $ATTACKS; do
  run "$attack" none
  run "$attack" capability
done

echo; echo "Done. Results:"; report
