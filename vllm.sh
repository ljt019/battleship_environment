#!/bin/bash

source ./setup.sh

CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model 'Qwen/Qwen3-14B-AWQ' --port 8000 --max-model-len 10240