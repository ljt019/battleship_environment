#!/bin/bash

set -e  # Exit on any error

echo "Setting up battleship RLVR environment..."

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "uv already installed"
fi

# Add verifiers with all extras
echo "Adding verifiers[all]..."
uv add 'verifiers[all]'

# Install additional dependencies
echo "Installing huggingface-cli..."
uv pip install huggingface-hub

echo "Installing wandb..."
uv pip install wandb

# Sync all dependencies
echo "Syncing dependencies..."
uv sync

# Set dummy OpenAI API key for vLLM
echo "Setting dummy OpenAI API key for vLLM..."
export OPENAI_API_KEY=asdf

echo "Setup complete!"
echo ""
echo "To start the vLLM server:"
echo "  CUDA_VISIBLE_DEVICES=0,1 uv run vf-vllm --model 'Qwen/Qwen3-1.7B' --tensor-parallel-size 2 --host 0.0.0.0 --port 8000"
echo ""
echo "To run evaluation:"
echo "  uv run python scripts/eval/evaluate_model.py --api vllm --num-samples 100 --max-tokens 512" 