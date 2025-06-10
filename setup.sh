#!/bin/bash

set -e  # Exit on any error

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

# Add verifiers and flash-attn
uv add 'verifiers[all]'
uv pip install flash-attn --no-build-isolation

uv add huggingface-hub
uv add wandb

uv sync

# Set dummy OpenAI API key for vLLM
export OPENAI_API_KEY=asdf

# Add uv to PATH permanently
# echo "Adding uv to PATH..."
# echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
#echo 'export OPENAI_API_KEY=asdf' >> ~/.bashrc

# download and login to huggingface hub
pip install huggingface-hub

# Source bashrc to make changes active in current session
source ~/.bashrc

# login to huggingface hub
huggingface-cli login
uv run wandb login

clear

echo "Setup complete! Pod is ready to use."