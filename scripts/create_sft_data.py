import time
import sys
import os
from openai import OpenAI
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.battleship_grpo import BattleshipEnv
from scripts.config import NUM_DATASET_SAMPLES, MAX_TOKENS_API, MAX_CONCURRENT_API, SEED, MAX_TURNS, DATASET_PATH, OPENROUTER_BASE_URL, OPENROUTER_API_KEY, OPENROUTER_MODEL

def main():
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY
    )

    vf_env = BattleshipEnv(
        seed=SEED,
        max_concurrent=MAX_CONCURRENT_API,
        max_turns=MAX_TURNS
    )

    sampling_args = {
        "max_tokens": MAX_TOKENS_API,
        "temperature": 0.7,
    }

    print(f"Starting test data generation")
    print(f"Target: Playing {NUM_DATASET_SAMPLES} games and keeping the top 50%")
    print(f"Model: {OPENROUTER_MODEL}")
    print(f"Max turns per game: {MAX_TURNS}")
    print()

    start_time = time.time()

    # Generate test batch
    results = vf_env.evaluate(
        client=client,
        model=OPENROUTER_MODEL,
        sampling_args=sampling_args,
        num_samples=NUM_DATASET_SAMPLES,
    )

    end_time = time.time()
    generation_time = end_time - start_time

    print(f"\nGenerated {len(results['prompt'])} games in {generation_time/60:.1f} minutes")
    print(f"Sample rewards: {results['reward'][:10]}")

    dataset = vf_env.make_dataset(results)
    print(f"\nDataset size before filtering: {len(dataset)}")

    # As long as there are at least 5 games, take the top 75%
    if len(dataset) >= 5: 
        import math
        top_count = max(1, math.ceil(0.75 * len(dataset)))
        dataset = dataset.sort("reward", reverse=True).select(range(top_count))
        print(f"Dataset size after filtering (top 75% = {top_count}): {len(dataset)}")
    else:
        dataset = dataset.sort("reward", reverse=True)
        print(f"Got fewer games than expected, keeping all {len(dataset)}")

    # Convert to per-turn samples (latest board + move)
    from datasets import Dataset

    turn_rows = []

    for example in dataset:
        full_prompt = example["prompt"]
        full_completion = example["completion"]
        msgs = full_prompt + full_completion
        episode_reward = example["reward"]

        static_msgs = full_prompt[:2]

        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue

            board_msg = None
            for j in range(i - 1, -1, -1):
                if msgs[j].get("role") == "user" and "<grid>" in msgs[j].get("content", ""):
                    board_msg = msgs[j]
                    break

            if board_msg is None:
                continue

            # Prompt for this sample = system + rules + latest board state
            prompt_turn = static_msgs + [board_msg]
            # Completion = ONLY the assistant's response for that turn
            completion_turn = [m]
            turn_rows.append({
                "prompt": prompt_turn,
                "completion": completion_turn,
                "reward": episode_reward,
                "answer": example.get("answer", ""),
                "task": example.get("task", None),
            })

    dataset = Dataset.from_list(turn_rows)
    print(f"Per-turn dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("No valid per-turn samples generated after format filtering. Exiting early.")
        return

    # Show quality stats
    rewards = dataset["reward"]
    print(f"\nFinal Dataset Quality:")
    print(f"   Best reward: {max(rewards):.3f}")
    print(f"   Worst reward: {min(rewards):.3f}")
    print(f"   Average reward: {sum(rewards)/len(rewards):.3f}")
    print(f"   Win rate: {sum(1 for r in rewards if r > 0.5)/len(rewards)*100:.1f}%")

    # Save locally
    print(f"\nSaving to {DATASET_PATH}")
    dataset.save_to_disk(DATASET_PATH)

    total_time = time.time() - start_time
    print(f"\nDone! Total time: {total_time/60:.1f} minutes")
    print(f"Generated {len(dataset)} battleship games")
    print(f"Run 'uv run scripts/upload_dataset.py' to upload to hub")

if __name__ == "__main__":
    main() 