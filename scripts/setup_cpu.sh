#!/bin/bash
#SBATCH --job-name=setup-cpu
#SBATCH --partition=small
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/setup_cpu_%j.out
set -euo pipefail

BASE=/scratch/${SLURM_JOB_ACCOUNT}/${USER}
mkdir -p "$BASE"
cd "$BASE"
module load gcc cmake 2>/dev/null || module load cmake 2>/dev/null || true

if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi
cmake -S llama.cpp -B llama.cpp/build -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
      -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build -j "${SLURM_CPUS_PER_TASK:-16}" --target llama-server

if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -q --upgrade pip
pip install -q openai tqdm huggingface_hub
pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

if [ ! -f medgemma-27b-hf/config.json ]; then
  HF_TOKEN=$(cat ~/.hf_token) python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download('google/medgemma-27b-text-it', local_dir='medgemma-27b-hf',
                  token=os.environ['HF_TOKEN'])"
fi

if [ ! -f medgemma-27b-q8_0.gguf ]; then
  python3 llama.cpp/convert_hf_to_gguf.py medgemma-27b-hf \
          --outtype q8_0 --outfile medgemma-27b-q8_0.gguf
fi

ls -lh "$BASE"/medgemma-27b-q8_0.gguf "$BASE"/llama.cpp/build/bin/llama-server
echo "setup done"
