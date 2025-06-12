#!/bin/bash
set -euo pipefail

setup_tmux() {
    if ! command -v tmux &>/dev/null; then
        echo "Installing tmux..."
        apt-get update -y
        apt-get install -y tmux
    else
        echo "tmux already installed"
    fi
}

setup_uv() {
    if ! command -v uv &> /dev/null; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Add the default installation path to PATH so that the newly
        # installed binary is immediately available to the rest of this script
        export PATH="$HOME/.local/bin:$PATH"

        # Verify that the installation succeeded
        if ! command -v uv &> /dev/null; then
            echo "Error: uv installation failed or not found in PATH"
            return 1
        fi
    fi

    # At this point uv is guaranteed to be available
    echo "Synchronising Python dependencies with uv..."
    uv sync
    echo "Installing flash-attn via uv..."
    uv pip install flash-attn --no-build-isolation
}

setup_huggingface() {
  if uv run huggingface-cli whoami 2>&1 | grep -q "Not logged in"; then
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