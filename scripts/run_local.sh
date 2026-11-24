#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

QUESTIONS=${1:-90}
HOURS=${2:-4}
BUDGET=$(python3 -c "print(int($HOURS * 3600))")

export MEDGEMMA_BASE_URL=${MEDGEMMA_BASE_URL:-http://localhost:11434/v1}
export MEDGEMMA_MODEL=${MEDGEMMA_MODEL:-medgemma:4b}
export MEDGEMMA_API_KEY=${MEDGEMMA_API_KEY:-ollama}
export MEDGEMMA_MAX_TOKENS=${MEDGEMMA_MAX_TOKENS:-2048}

curl -sf "${MEDGEMMA_BASE_URL%/v1}/api/tags" > /dev/null \
  || { echo "No Ollama server at ${MEDGEMMA_BASE_URL%/v1}. Start it: ollama serve"; exit 1; }
mkdir -p output logs

if [ "${KEEP:-0}" != 1 ] && compgen -G "output/*_seed0.json" > /dev/null; then
  old="output/old_$(date +%Y%m%d_%H%M)"
  mkdir -p "$old" && mv output/*_seed0.json "$old"/
  echo "moved earlier output to $old"
fi

run() { 
  echo ">>> $1  ($(( $2 / 60 )) min allowed)"
  caffeinate -i python3 main.py --difficulty "$1" --num_samples "$QUESTIONS" \
      --data_root ./data --time_budget "$2" --resume 2>&1 | tee -a "logs/$1.log" | tail -4
}

START=$SECONDS
run basic "$BUDGET"

LEFT=$(( BUDGET - (SECONDS - START) ))
if [ "$LEFT" -lt 60 ]; then
  echo "No time left for the multi-agent arm."
else
  run adaptive "$LEFT"
fi

python3 analyze.py output/medgemma_medqa_basic_seed0.json \
                   output/medgemma_medqa_adaptive_seed0.json
