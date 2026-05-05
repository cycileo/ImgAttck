#!/usr/bin/env bash
#SBATCH --job-name=imgattck
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_DIR"

# Force Hugging Face/Transformers to use already-downloaded local cache files.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Override these at submit time if your cluster has a specific cache/env layout:
#   sbatch --export=ALL,PYTHON=/path/to/python,HF_HOME=/scratch/$USER/hf scripts/slurm_imgattck.sh
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
COMMAND="${IMGATTCK_COMMAND:-optimize-pixels}"
CONFIG="${1:-configs/default.yaml}"
if [ "$#" -gt 0 ]; then
  shift
fi

srun "$PYTHON" -m imgattck "$COMMAND" "$CONFIG" "$@"
