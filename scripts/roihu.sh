#!/bin/bash
#SBATCH --job-name=mdagents
#SBATCH --account=CHANGE_ME              # your CSC project, e.g. project_2001234
#SBATCH --partition=gpumedium            # gputest = 15 min, for a first try
#SBATCH --time=12:00:00                  # gpumedium allows up to 36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72               # 72 per GPU on Roihu
#SBATCH --gres=gpu:gh200:1               # one GH200 has 96 GiB: fits 27B in bf16
#SBATCH --output=logs/roihu_%j.out
#
# Start a vLLM server for MedGemma on this node, then run main.py against it.
# Run from the repository root, after `mkdir -p logs output`:
#
#   sbatch --partition=gputest --time=00:15:00 scripts/roihu.sh basic 3   # try it
#   sbatch scripts/roihu.sh basic    500
#   sbatch scripts/roihu.sh adaptive 500
#
# A job that hits its time limit loses nothing: submit the same line again.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

DIFFICULTY=${1:-basic}
QUESTIONS=${2:-500}
MODEL_ID=${MODEL_ID:-google/medgemma-27b-text-it}   # copy the exact id from Hugging Face
PORT=8000

module purge
module load csc-tools 2>/dev/null
module load python-vllm/0.19 || { echo "Find the module name with: module spider vllm"; exit 1; }

# Model weights are ~55 GB: cache them in project scratch, not in $HOME.
export HF_HOME=${HF_HOME:-/scratch/$SLURM_JOB_ACCOUNT/$USER/hf-cache}
mkdir -p "$HF_HOME"
# MedGemma is gated: accept its terms on Hugging Face and put a read token here.
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
export MEDGEMMA_MAX_TOKENS=${MEDGEMMA_MAX_TOKENS:-4096}   # the multi-agent path needs more than 2048
export MEDGEMMA_GUIDED=1                                  # vLLM can force the answer letter

python3 main.py --difficulty "$DIFFICULTY" --num_samples "$QUESTIONS" --data_root ./data \
    --output "output/roihu_27b_${DIFFICULTY}_seed0.json" --resume
