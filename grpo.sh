#!/bin/bash

# Run GRPO training on GPU 1 (GPU 0 is used by the vLLM server).
CUDA_VISIBLE_DEVICES=1 uv run accelerate launch \
  --config_file configs/zero3.yaml \
  --num_processes 1 \
  scripts/train_grpo.py