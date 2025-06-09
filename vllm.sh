#!/bin/bash

export OPENAI_API_KEY=asdf
CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model 'ljt019/Qwen3-1.7B-Battleship-SFT' --host 0.0.0.0 --port 8000
