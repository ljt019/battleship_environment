#!/bin/bash
set -euo pipefail

# Load helper functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup.sh"

setup_tmux

setup_uv

setup_huggingface

setup_dummy_openai_api_key

VLLM_CMD=(CUDA_VISIBLE_DEVICES=0 uv run vf-vllm \
  --model 'ljt019/Qwen3-1.7B-Battleship-SFT' \
  --port 8000 \
  --max-model-len 15310)

SESSION="vllm"

# If the session already exists, attach to it.
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux attach -t "$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" "${VLLM_CMD[*]}"

tmux split-window -h -t "$SESSION":0

tmux select-layout -t "$SESSION":0 even-horizontal

exec tmux attach -t "$SESSION"