#!/bin/bash
set -euo pipefail

# Load functions from setup.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup.sh"

setup_uv

huggingface-cli login

setup_huggingface

setup_wandb

setup_dummy_openai_api_key

CUDA_VISIBLE_DEVICES=1 uv run accelerate launch \
  --config_file configs/zero3.yaml \
  --num_processes 1 \
  scripts/train_grpo.py