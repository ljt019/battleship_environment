#!/bin/bash

export OPENAI_API_KEY=asdf
CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model 'Qwen/Qwen3-1.7B' --host 0.0.0.0 --port 8000
