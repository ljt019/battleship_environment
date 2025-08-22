#!/usr/bin/env python3
"""Inspect the generated training data to verify initial state is included."""

import json

from datasets import Dataset

# Load the dataset
dataset = Dataset.load_from_disk("data/training_data")

print(f"Dataset size: {len(dataset)}")
print("\nChecking first example...")

# Get the first example
example = dataset[0]

print("\n=== PROMPT MESSAGES ===")
for i, msg in enumerate(example["prompt"]):
    print(f"\nMessage {i + 1}:")
    print(f"Role: {msg['role']}")
    print(f"Content preview: {msg['content'][:200]}...")
    if len(msg["content"]) > 200:
        print(f"(truncated, total length: {len(msg['content'])} chars)")

print("\n=== COMPLETION MESSAGES (first 3) ===")
for i, msg in enumerate(example["completion"][:3]):
    print(f"\nMessage {i + 1}:")
    print(f"Role: {msg['role']}")
    print(f"Content preview: {msg['content'][:200]}...")
    if len(msg["content"]) > 200:
        print(f"(truncated, total length: {len(msg['content'])} chars)")

print(f"\nTotal completion messages: {len(example['completion'])}")
print(f"Game seed: {example['game_seed']}")
print(f"Temperature: {example['temperature']}")
print(f"Reward: {example['reward']}")
print(f"Turns: {example['turns']}")
