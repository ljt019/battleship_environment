#!/bin/bash

setup_tmux() {
    if ! command -v tmux &>/dev/null; then
        echo "Installing tmux..."
        sudo apt-get update -y
        sudo apt-get install -y tmux
    else
        echo "tmux already installed"
    fi
}

setup_uv() {
    if ! command -v uv &> /dev/null; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        if ! command -v uv &> /dev/null; then
            echo "Error: uv installation failed or not found in PATH"
            return 1

        uv sync
        uv pip install flash-attn --no-build-isolation
        fi
    else
        uv sync
        uv pip install flash-attn --no-build-isolation
        echo "uv already installed"
    fi
}

setup_huggingface() {
  # Check if already logged in to Hugging Face
    if ! uv run huggingface-cli whoami &>/dev/null; then
        echo "Logging in to Hugging Face Hub..."
        uv run huggingface-cli login
    else
        echo "Already logged in to Hugging Face Hub."
    fi
}

setup_wandb() {
  # Check if already logged in to Weights & Biases
    if ! uv run wandb status 2>&1 | grep -q "You are logged in"; then
        echo "Logging in to Weights & Biases..."
        uv run wandb login
    else
        echo "Already logged in to Weights & Biases."
    fi
}

setup_dummy_openai_api_key() {
    export OPENAI_API_KEY=asdf
}