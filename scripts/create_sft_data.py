import time
import sys
import os
from openai import OpenAI

# Add project root to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.battleship_grpo import BattleshipEnv
from scripts.config import DATASET_MODEL_SIZE, NUM_DATASET_SAMPLES, MAX_TOKENS_API, MAX_CONCURRENT_API, SEED, MAX_TURNS, DATASET_PATH, HUB_DATASET_NAME, VLLM_BASE_URL, VLLM_API_KEY, SFT_MODEL_NAME

def main():
    # Setup vLLM client
    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY
    )

    vf_env = BattleshipEnv(
        num_samples=NUM_DATASET_SAMPLES,
        num_eval_samples=NUM_DATASET_SAMPLES,
        seed=SEED,
        max_concurrent=MAX_CONCURRENT_API,
        max_turns=MAX_TURNS
    )

    sampling_args = {
        "max_tokens": MAX_TOKENS_API,
        "temperature": 0.7,
    }

    print(f"Starting test data generation")
    print(f"Target: Playing {NUM_DATASET_SAMPLES} games and keeping the top 40%")
    print(f"Model: Qwen3-{DATASET_MODEL_SIZE}")
    print(f"Max turns per game: {MAX_TURNS}")
    print()

    start_time = time.time()

    # Generate test batch
    results = vf_env.evaluate(
        client=client,
        model=SFT_MODEL_NAME,
        sampling_args=sampling_args,
        num_samples=NUM_DATASET_SAMPLES,
    )

    end_time = time.time()
    generation_time = end_time - start_time

    print(f"\nGenerated {len(results['prompt'])} games in {generation_time/60:.1f} minutes")
    print(f"Sample rewards: {results['reward'][:10]}")

    dataset = vf_env.make_dataset(results)
    print(f"\nDataset size before filtering: {len(dataset)}")

    # As long as there are at least 5 games, take the top 40%
    if len(dataset) >= 5: 
        top_count = max(1, len(dataset) * 2 // 5)  # 40% = 2/5
        dataset = dataset.sort("reward", reverse=True).select(range(top_count))
        print(f"Dataset size after filtering (top 40% = {top_count}): {len(dataset)}")
    else:
        dataset = dataset.sort("reward", reverse=True)
        print(f"Got fewer games than expected, keeping all {len(dataset)}")

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