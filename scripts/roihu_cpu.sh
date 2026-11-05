#!/bin/bash
# Run an experiment on a Roihu CPU node: MedGemma 27B (8-bit) served by
# llama.cpp, then main.py (or PY_SCRIPT) against it. Submit from roihu-cpu.csc.fi:
#
#   sbatch --account=<project> scripts/roihu_cpu.sh <difficulty> <num_samples> [tag]
#
# Environment (all optional):
#   PY_SCRIPT    script to run instead of main.py (e.g. classify_check.py)
#   EXTRA_ARGS   extra arguments for it (e.g. "--prompt paper --compare")
#   CTX          context window, default 32768
#   MDAGENTS_*   passed through to main.py as on the GPU side
# Needs scripts/setup_cpu.sh to have been run once.
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

# One slot, so the whole context window is available to the single request.
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
export MEDGEMMA_GUIDED=0          # guided decoding is vLLM-only
export MEDGEMMA_MAX_TOKENS=${MEDGEMMA_MAX_TOKENS:-4096}   # as on the GPU side

python3 "${PY_SCRIPT:-main.py}" --difficulty "$1" --num_samples "$2" \
    --output "output/cpu_27b_${1}${3:+_$3}_seed0.json" --resume ${EXTRA_ARGS:-}
