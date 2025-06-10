#!/bin/bash

export OPENAI_API_KEY=asdf
CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model 'Qwen/Qwen3-14B-AWQ' --port 8000 --max-model-len 10240