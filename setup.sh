#!/bin/bash

set -e  # Exit on any error

echo "Setting up battleship RLVR environment..."

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    # Verify uv is now available
    if ! command -v uv &> /dev/null; then
        echo "Error: uv installation failed or not found in PATH"
        exit 1
    fi
else
    echo "uv already installed"
fi

# Add verifiers with all extras
echo "Adding verifiers[all]..."
uv add 'verifiers[all]'

# Add dependencies to pyproject.toml instead of pip installing
echo "Adding huggingface-hub..."
uv add huggingface-hub

echo "Adding wandb..."
uv add wandb

# Sync all dependencies
echo "Syncing dependencies..."
uv sync

# Set dummy OpenAI API key for vLLM
echo "Setting dummy OpenAI API key for vLLM..."
export OPENAI_API_KEY=asdf

# Add uv to PATH permanently
echo "Adding uv to PATH..."
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'export OPENAI_API_KEY=asdf' >> ~/.bashrc