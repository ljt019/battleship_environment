#!/bin/bash

export OPENAI_API_KEY=asdf
CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model 'unsloth/Qwen3-14B-GGUF' --port 8000 --quantization gguf --gguf-file Qwen3-14B-Q4_K_M.gguf