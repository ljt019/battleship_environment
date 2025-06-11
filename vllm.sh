#!/bin/bash

CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model 'willcb/Qwen3-1.7B' --port 8000 --max-model-len 17336