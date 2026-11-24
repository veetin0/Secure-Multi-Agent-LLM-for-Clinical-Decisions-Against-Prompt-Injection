#!/bin/bash
#SBATCH --job-name=mdagents-cpu
#SBATCH --partition=small
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/roihu_cpu_%j.out
set -euo pipefail

BASE=/scratch/${SLURM_JOB_ACCOUNT}/${USER}
source "$BASE/venv/bin/activate"
mkdir -p logs output

"$BASE/llama.cpp/build/bin/llama-server" -m "$BASE/medgemma-27b-q8_0.gguf" \
    --alias medgemma --jinja -c "${CTX:-32768}" -np 1 \
    -t "${SLURM_CPUS_PER_TASK}" --host 127.0.0.1 --port 8000 \
    > "logs/llama_${SLURM_JOB_ID}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  sleep 10
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "server did not start"; tail -30 "logs/llama_${SLURM_JOB_ID}.log"; exit 1; }
echo "server ready"

export MEDGEMMA_BASE_URL=http://127.0.0.1:8000/v1
export MEDGEMMA_MODEL=medgemma
export OPENAI_API_KEY=none
export MEDGEMMA_GUIDED=0 
export MDAGENTS_DATA=${MDAGENTS_DATA:-data} 
export MEDGEMMA_TIMEOUT=${MEDGEMMA_TIMEOUT:-3600} 
export MEDGEMMA_MAX_TOKENS=${MEDGEMMA_MAX_TOKENS:-4096}

python3 "${PY_SCRIPT:-main.py}" --difficulty "$1" --num_samples "$2" \
    --output "output/cpu_27b_${1}${3:+_$3}_seed0.json" --resume ${EXTRA_ARGS:-}
