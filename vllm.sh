#!/bin/bash

CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model 'ljt019/Qwen3-1.7B-battleship-sft' --port 8000 --max-model-len 17336