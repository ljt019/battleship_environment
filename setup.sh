#!/bin/bash

snap install astral-uv --classic

uv add 'verifiers[all]' && uv pip install flash-attn --no-build-isolation

uv run hf auth login

uv run wandb login