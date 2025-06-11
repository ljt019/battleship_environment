#!/bin/bash

# Start vLLM server on GPU 0 serving the fine-tuned Battleship SFT model.
# --gpu-memory-utilization limits VRAM usage (0-1); tweak if you hit OOM.
# --tensor-parallel-size 1 because we are using a single GPU for inference.
# --max-model-len sets the maximum context length (prompt+completion) that vLLM will accept.
CUDA_VISIBLE_DEVICES=0 uv run vf-vllm \
  --model 'ljt019/Qwen3-1.7B-Battleship-SFT' \
  --port 8000 \
  --max-model-len 17336