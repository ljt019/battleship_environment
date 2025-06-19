#!/usr/bin/env python3
"""
Sanity check script using REAL BattleshipGame and environment.
Tests reward ranges, distributions, and edge cases before GRPO training.
"""

import sys
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import asyncio

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Name of the dataset on Hugging Face hub to verify.
DATASET_NAME = "ljt019/battleship-synthetic-games-dataset"

# How many samples from the dataset to evaluate. -1 = use entire split.
NUM_SAMPLES_TO_EVAL = -1

# Concurrency for rubric scoring
MAX_CONCURRENT = 32

print("Importing Battleship modules…")
from src.battleship_grpo.battleship_env import BattleshipEnv
print("Imports successful")


def evaluate_dataset(env: BattleshipEnv, dataset_name: str = DATASET_NAME, num_samples: int = -1):
    """Download a dataset from the Hugging Face Hub and score rewards using the env rubric."""

    print(f"\n=== Loading dataset '{dataset_name}' from Hugging Face… ===")
    ds = load_dataset(dataset_name, split="train")

    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    print(f"Dataset loaded: {len(ds)} samples (evaluating {len(ds)}).")

    prompts = ds["prompt"]
    completions = ds["completion"]
    answers = ds["answer"] if "answer" in ds.column_names else ["" for _ in prompts]
    states = ds["state"] if "state" in ds.column_names else [{} for _ in prompts]
    tasks = ds["task"] if "task" in ds.column_names else ["default" for _ in prompts]

    print("Scoring rollouts with rubric (this may take a bit)…")
    rewards_dict = env.rubric.score_rollouts(
        prompts=prompts,
        completions=completions,
        answers=answers,
        states=states,
        tasks=tasks,
        max_concurrent=MAX_CONCURRENT,
        apply_weights=True,
    )

    rewards = rewards_dict["reward"]

    rewards_np = np.array(rewards)
    print("\nReward statistics:")
    print(f"  mean   : {rewards_np.mean():.3f}")
    print(f"  std    : {rewards_np.std():.3f}")
    print(f"  min/max: {rewards_np.min():.3f} / {rewards_np.max():.3f}")

    return rewards_dict


def main():
    """Evaluate reward rubric on a pre-generated synthetic dataset."""

    print("Battleship Reward Sanity Check – Dataset Mode")
    print("=" * 60)

    try:
        print("Initializing environment…")
        env = BattleshipEnv()
        print("Environment ready.\n")

        evaluate_dataset(env, DATASET_NAME, NUM_SAMPLES_TO_EVAL)

        print("\n✓ Finished dataset evaluation.")
        return 0
    except Exception as e:
        print(f"Fatal error in main: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    print("Starting sanity check script...")
    try:
        result = main()
        print(f"Script completed with result: {result}")
        sys.exit(result)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 