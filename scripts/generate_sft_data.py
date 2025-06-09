"""
Generate SFT data using vLLM with local models.
"""

import os
import sys
import time
from openai import OpenAI

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import verifiers as vf
from battleship_env import BattleshipEnv
from config import NUM_DATASET_SAMPLES, MAX_TOKENS, MAX_CONCURRENT, NUM_DATASET_EVAL_SAMPLES, SEED, MAX_TURNS, DATASET_PATH, HUB_DATASET_NAME, VLLM_BASE_URL, VLLM_API_KEY, SFT_MODEL_NAME

# Setup vLLM client
client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY
)

vf_env = BattleshipEnv(
    num_samples=NUM_DATASET_SAMPLES, 
    seed=SEED,
    max_concurrent=MAX_CONCURRENT,
    max_turns=MAX_TURNS
)

sampling_args = {
    "max_tokens": MAX_TOKENS,
    "temperature": 0.7,
}

print(f"Starting test data generation")
print(f"Target: Playing {NUM_DATASET_SAMPLES} games and keeping the top 20%")
print(f"Model: {SFT_MODEL_NAME}")
print(f"Max turns per game: {MAX_TURNS}")
print()

start_time = time.time()

# Generate test batch
results = vf_env.evaluate(
    client=client,
    model=SFT_MODEL_NAME,
    sampling_args=sampling_args,
    num_samples=NUM_DATASET_SAMPLES
)

end_time = time.time()
generation_time = end_time - start_time

print(f"\nGenerated {len(results['prompt'])} games in {generation_time/60:.1f} minutes")
print(f"Sample rewards: {results['reward'][:10]}")

# Create SFT dataset - filter to top 20%
dataset = vf_env.make_dataset(results)
print(f"\nDataset size before filtering: {len(dataset)}")

if len(dataset) >= 5:  # Need at least 5 to get meaningful top 20%
    # Take top 20% of games
    top_count = max(1, len(dataset) // 5)  # 20% = 1/5
    dataset = dataset.sort("reward", reverse=True).select(range(top_count))
    print(f"Dataset size after filtering (top 20% = {top_count}): {len(dataset)}")
else:
    # Fallback if we got fewer games
    dataset = dataset.sort("reward", reverse=True)
    print(f"Got fewer games than expected, keeping all {len(dataset)}")

# Show quality stats
rewards = dataset["reward"]
print(f"\nFinal Dataset Quality:")
print(f"   Best reward: {max(rewards):.3f}")
print(f"   Worst reward: {min(rewards):.3f}")
print(f"   Average reward: {sum(rewards)/len(rewards):.3f}")
print(f"   Win rate: {sum(1 for r in rewards if r > 0.5)/len(rewards)*100:.1f}%")

# Save
print(f"\nSaving to {DATASET_PATH}")
dataset.save_to_disk(DATASET_PATH)

try:
    dataset.push_to_hub(HUB_DATASET_NAME)
    print(f"Pushed to hub: {HUB_DATASET_NAME}")
except Exception as e:
    print(f"Failed to push to hub: {e}")

total_time = time.time() - start_time
print(f"\nDone! Total time: {total_time/60:.1f} minutes")
print(f"Generated {len(dataset)} high-quality battleship games") 