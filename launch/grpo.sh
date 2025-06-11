#!/bin/bash
set -euo pipefail

# Load functions from setup.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup.sh"

setup_tmux

setup_uv

setup_huggingface

setup_wandb

setup_dummy_openai_api_key

# commands for each pane
VLLM_CMD=(CUDA_VISIBLE_DEVICES=0 uv run vf-vllm \
  --model 'ljt019/Qwen3-1.7B-Battleship-SFT' \
  --port 8000 \
  --max-model-len 15310)

GRPO_CMD=(CUDA_VISIBLE_DEVICES=1 uv run accelerate launch \
  --config_file configs/zero3.yaml \
  --num_processes 1 \
  scripts/train_grpo.py)

SESSION="battleship"

# If the session already exists, attach; else create and run commands
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Attaching to existing tmux session '$SESSION'…"
  tmux attach -t "$SESSION"
  exit 0
fi

# Create new detached session running vLLM in pane 0
TMUX_NEW=(tmux new-session -d -s "$SESSION" "${VLLM_CMD[*]}")
"${TMUX_NEW[@]}"

# Split horizontally and run GRPO trainer in pane 1
TMUX_SPLIT=(tmux split-window -h -t "$SESSION":0 "${GRPO_CMD[*]}")
"${TMUX_SPLIT[@]}"

# Optional: better default layout
tmux select-layout -t "$SESSION":0 even-horizontal

# Attach to the session so the user sees both outputs
exec tmux attach -t "$SESSION"