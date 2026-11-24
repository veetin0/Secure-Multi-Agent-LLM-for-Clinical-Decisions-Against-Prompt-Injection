#!/bin/bash
#SBATCH --job-name=mdagents
#SBATCH --account=project_2021639 
#SBATCH --partition=gpumedium 
#SBATCH --time=12:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72 
#SBATCH --gres=gpu:gh200:1 
#SBATCH --output=logs/roihu_%j.out


set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

DIFFICULTY=${1:-basic}
QUESTIONS=${2:-500}
MODEL_ID=${MODEL_ID:-google/medgemma-27b-text-it} 
PORT=8000

module purge
module load csc-tools 2>/dev/null
module load python-vllm/0.19 || { echo "Find the module name with: module spider vllm"; exit 1; }

export HF_HOME=${HF_HOME:-/scratch/$SLURM_JOB_ACCOUNT/$USER/hf-cache}
mkdir -p "$HF_HOME"
[ -f ~/.hf_token ] && export HF_TOKEN=$(tr -d '[:space:]' < ~/.hf_token)

echo "model $MODEL_ID | $DIFFICULTY | $QUESTIONS questions | node $(hostname) | job $SLURM_JOB_ID"

vllm serve "$MODEL_ID" --host 127.0.0.1 --port $PORT --served-model-name medgemma \
    --max-model-len 8192 --gpu-memory-utilization 0.90 > "logs/vllm_$SLURM_JOB_ID.log" 2>&1 &
VLLM=$!
trap 'kill $VLLM 2>/dev/null' EXIT

echo "waiting for the server (the first run downloads the weights; allow 20 min)"
for _ in $(seq 180); do
  curl -sf http://127.0.0.1:$PORT/health > /dev/null && break
  kill -0 $VLLM 2>/dev/null || { echo "vLLM exited:"; tail -30 "logs/vllm_$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
curl -sf http://127.0.0.1:$PORT/health > /dev/null || { echo "server never became healthy"; exit 1; }

export MEDGEMMA_BASE_URL=http://127.0.0.1:$PORT/v1
export MEDGEMMA_MODEL=medgemma
export MEDGEMMA_API_KEY=not-needed
export MEDGEMMA_MAX_TOKENS=${MEDGEMMA_MAX_TOKENS:-4096} 
export MEDGEMMA_GUIDED=1 
# Third argument = a tag for the output file, so different conditions never
# resume each other's file. PY_SCRIPT runs another script (e.g. make_attacks.py),
# EXTRA_ARGS adds flags (e.g. "--fixed --passages data/medqa/passages.jsonl").
python3 "${PY_SCRIPT:-main.py}" --difficulty "$DIFFICULTY" --num_samples "$QUESTIONS" --data_root ./data \
    --output "output/roihu_27b_${DIFFICULTY}${3:+_$3}_seed0.json" --resume ${EXTRA_ARGS:-}
