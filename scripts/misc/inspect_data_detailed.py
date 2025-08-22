#!/usr/bin/env python3
"""Detailed inspection of the message flow."""

import json

from datasets import Dataset

# Load the dataset
dataset = Dataset.load_from_disk("data/training_data")

example = dataset[0]

print("=== FULL MESSAGE FLOW ===\n")

# Show all prompt messages
print("PROMPT MESSAGES:")
for i, msg in enumerate(example["prompt"]):
    print(f"\n[{i}] Role: {msg['role']}")
    if 'value="start"' in msg.get("content", ""):
        print("    >>> CONTAINS INITIAL GAME STATE! <<<")
    print(f"    Content: {msg['content'][:150]}...")

print("\n" + "=" * 50 + "\n")

# Show first few completion messages
print("COMPLETION MESSAGES (first 5):")
for i, msg in enumerate(example["completion"][:5]):
    print(f"\n[{i}] Role: {msg['role']}")
    if 'value="start"' in msg.get("content", ""):
        print("    >>> CONTAINS INITIAL GAME STATE! <<<")
    if "<guess>" in msg.get("content", ""):
        print("    >>> CONTAINS A GUESS! <<<")
    print(f"    Content: {msg['content'][:150]}...")
