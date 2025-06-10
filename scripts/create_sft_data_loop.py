import time
import sys
import os
from openai import OpenAI
from datasets import concatenate_datasets

# Add project root to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.battleship_grpo import BattleshipEnv
from scripts.config import DATASET_MODEL_SIZE, NUM_DATASET_SAMPLES, MAX_TOKENS_API, MAX_CONCURRENT_API, SEED, MAX_TURNS, DATASET_PATH, HUB_DATASET_NAME, VLLM_BASE_URL, VLLM_API_KEY, SFT_MODEL_NAME

# ----------------------------
# Configuration
# ----------------------------
NUM_ITERATIONS = 3  # Change this to control how many batches to generate

def generate_batch(client, iteration):
    """Generate one batch of battleship games"""
    print(f"\n{'='*50}")
    print(f"ITERATION {iteration + 1}/{NUM_ITERATIONS}")
    print(f"{'='*50}")
    
    vf_env = BattleshipEnv(
        num_samples=NUM_DATASET_SAMPLES,
        seed=SEED + iteration,  # Different seed per iteration
        max_concurrent=MAX_CONCURRENT_API,
        max_turns=MAX_TURNS
    )

    sampling_args = {
        "max_tokens": MAX_TOKENS_API,
        "temperature": 0.7,
    }

    print(f"Generating batch {iteration + 1}: {NUM_DATASET_SAMPLES} games")
    print(f"Model: Qwen3-{DATASET_MODEL_SIZE}")
    print(f"Max turns per game: {MAX_TURNS}")
    
    start_time = time.time()

    # Generate batch
    results = vf_env.evaluate(
        client=client,
        model=SFT_MODEL_NAME,
        sampling_args=sampling_args,
        num_samples=NUM_DATASET_SAMPLES,
    )

    batch_time = time.time() - start_time
    print(f"Generated {len(results['prompt'])} games in {batch_time/60:.1f} minutes")
    print(f"Sample rewards: {results['reward'][:10]}")

    # Convert to dataset
    dataset = vf_env.make_dataset(results)
    print(f"Dataset size before filtering: {len(dataset)}")

    # As long as there are at least 5 games, take the top 40%
    if len(dataset) >= 5: 
        top_count = max(1, len(dataset) * 2 // 5)  # 40% = 2/5
        dataset = dataset.sort("reward", reverse=True).select(range(top_count))
        print(f"Dataset size after filtering (top 40% = {top_count}): {len(dataset)}")
    else:
        dataset = dataset.sort("reward", reverse=True)
        print(f"Got fewer games than expected, keeping all {len(dataset)}")

    # Show batch quality stats
    rewards = dataset["reward"]
    print(f"Batch {iteration + 1} Quality:")
    print(f"   Best reward: {max(rewards):.3f}")
    print(f"   Worst reward: {min(rewards):.3f}")
    print(f"   Average reward: {sum(rewards)/len(rewards):.3f}")
    print(f"   Win rate: {sum(1 for r in rewards if r > 0.5)/len(rewards)*100:.1f}%")
    
    return dataset, batch_time

def main():
    # Setup vLLM client
    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY
    )

    print(f"Starting iterative data generation")
    print(f"Target: {NUM_ITERATIONS} batches × {NUM_DATASET_SAMPLES} games = {NUM_ITERATIONS * NUM_DATASET_SAMPLES} total games")
    print(f"Keeping top 40% from each batch")
    print(f"Expected final dataset size: ~{NUM_ITERATIONS * NUM_DATASET_SAMPLES * 2 // 5} samples")
    
    all_datasets = []
    total_generation_time = 0
    
    # Generate batches iteratively
    for i in range(NUM_ITERATIONS):
        batch_dataset, batch_time = generate_batch(client, i)
        all_datasets.append(batch_dataset)
        total_generation_time += batch_time
        
        print(f"Batch {i + 1} completed: {len(batch_dataset)} samples added")
        total_so_far = sum(len(d) for d in all_datasets)
        print(f"Total samples collected so far: {total_so_far}")
    
    # Combine all batches
    print(f"\n{'='*50}")
    print("COMBINING BATCHES")
    print(f"{'='*50}")
    
    if len(all_datasets) == 1:
        final_dataset = all_datasets[0]
    else:
        final_dataset = concatenate_datasets(all_datasets)
    
    print(f"Combined dataset: {len(final_dataset)} samples")
    
    # Show final quality stats
    rewards = final_dataset["reward"]
    print(f"\nFinal Combined Dataset Quality:")
    print(f"   Best reward: {max(rewards):.3f}")
    print(f"   Worst reward: {min(rewards):.3f}")
    print(f"   Average reward: {sum(rewards)/len(rewards):.3f}")
    print(f"   Win rate: {sum(1 for r in rewards if r > 0.5)/len(rewards)*100:.1f}%")

    # Save locally
    print(f"\nSaving to {DATASET_PATH}")
    final_dataset.save_to_disk(DATASET_PATH)

    print(f"\nGeneration Summary:")
    print(f"   Batches: {NUM_ITERATIONS}")
    print(f"   Games per batch: {NUM_DATASET_SAMPLES}")
    print(f"   Total games generated: {NUM_ITERATIONS * NUM_DATASET_SAMPLES}")
    print(f"   Final dataset size: {len(final_dataset)}")
    print(f"   Total generation time: {total_generation_time/60:.1f} minutes")
    print(f"   Average time per batch: {total_generation_time/NUM_ITERATIONS/60:.1f} minutes")
    
    print(f"\nRun 'uv run scripts/upload_dataset.py' to upload to hub")

if __name__ == "__main__":
    main()
