import asyncio
import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv

from battleship_environment import load_environment

################ Data Gen Config #################

NUM_SAMPLES = 400
NUM_GAMES = 1000
MAX_TURNS = 60
MAX_CONCURRENT = 10  # Number of concurrent rollouts

OPENROUTER_MODEL_NAME = "qwen/qwen3-235b-a22b-2507"

UPLOAD_TO_HUB = True
HUB_REPO_ID = "ljt019/battleship-sft-0825"

OUTPUT_PATH = "data/training_data"

MIN_SCORE_THRESHOLD = 0.3

##################################################


async def generate_single_rollout(
    env, client, model: str, seed: int, temp: float, answer: str
):
    """Generate a single rollout with given parameters."""
    try:
        # The environment's rollout will now automatically handle
        # adding the system prompt and initial game state
        completion, state = await env.rollout(
            client=client,
            model=model,
            prompt=[],  # Empty prompt - env will add system prompt and initial state
            answer=answer,
            sampling_args={"temperature": temp, "max_tokens": 200},
            state={"seed": seed},  # Pass seed for reproducible game setups
        )

        # Score the solution using the environment's rubric
        rewards = await env.rubric.score_rollout(
            prompt="Play Battleship game",
            completion=completion,
            answer=answer,
            state=state,
        )

        # Extract total reward and individual metrics from RolloutScore object
        total_reward = float(rewards.reward) if hasattr(rewards, "reward") else 0.0

        if total_reward > MIN_SCORE_THRESHOLD:
            # The completion from rollout contains the full conversation
            # We need to split it to get prompt and completion separately

            # Find the first assistant message - that's where completion starts
            first_assistant_idx = None
            for i, msg in enumerate(completion):
                if msg["role"] == "assistant":
                    first_assistant_idx = i
                    break

            if first_assistant_idx is not None:
                # Prompt is everything before the first assistant message
                prompt_messages = completion[:first_assistant_idx]
                # Completion is everything from the first assistant message onward
                completion_messages = completion[first_assistant_idx:]
            else:
                # Fallback: if no assistant message found, use system prompt
                prompt_messages = [{"role": "system", "content": env.system_prompt}]
                completion_messages = completion

            result = {
                "prompt": prompt_messages,
                "completion": completion_messages,
                "reward": total_reward,
                "victory": state.get("victory", False),
                "turns": state.get("turn", 0),
            }
            print(
                f"  ✓ Added sample (seed={seed}, temp={temp}, reward={total_reward:.3f}, victory={state.get('victory', False)})"
            )
            return result
        else:
            print(
                f"  ✗ Rejected sample (seed={seed}, temp={temp}, reward={total_reward:.3f})"
            )
            return None

    except Exception as e:
        print(f"  ✗ Error generating sample (seed={seed}, temp={temp}): {e}")
        import traceback

        print(f"     Full traceback: {traceback.format_exc()}")
        return None


async def generate_battleship_training_data(
    env, client, model: str, num_samples: int = 1000
) -> Dataset:
    """
    Generate diverse battleship game solutions for training using verifiers rollouts.
    Uses concurrent execution for faster generation.

    Args:
        env: BattleshipEnv instance
        client: OpenAI/Anthropic client for API calls
        model: Model name to use for generation
        num_samples: Number of training samples to generate

    Returns:
        HuggingFace Dataset with training examples
    """
    from asyncio import Semaphore

    # Create all rollout tasks
    tasks = []
    for i in range(num_samples):
        # Get a random game setup from dataset
        example = env.dataset[i % len(env.dataset)]
        seed = example.get("seed", i)
        answer = example["answer"]  # "victory" for battleship

        # Generate multiple solutions with different temperatures for diversity
        for temp in [0.3, 0.7, 1.0]:
            tasks.append((seed, temp, answer))

    print(
        f"Generating {len(tasks)} total rollouts with max concurrency of {MAX_CONCURRENT}"
    )

    # Create semaphore to limit concurrency
    semaphore = Semaphore(MAX_CONCURRENT)

    async def limited_rollout(seed, temp, answer):
        async with semaphore:
            return await generate_single_rollout(env, client, model, seed, temp, answer)

    # Execute all tasks concurrently with limited concurrency
    results = await asyncio.gather(
        *[limited_rollout(seed, temp, answer) for seed, temp, answer in tasks]
    )

    # Filter out None results (failed/rejected samples)
    valid_results = [r for r in results if r is not None]

    print(f"\nGenerated {len(valid_results)} high-quality training examples")
    return Dataset.from_list(valid_results)


async def save_and_upload_dataset(
    dataset: Dataset,
    output_path: str,
    upload_to_hub: bool = False,
    hub_repo_id: str = None,
):
    """Save dataset locally and optionally upload to Hugging Face Hub."""
    if not dataset or len(dataset) == 0:
        print("No training examples to save!")
        return

    # Save locally
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_file))

    print(f"\nDataset saved to {output_file}")
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Dataset features: {list(dataset.features.keys())}")

    # Show some stats
    if "victory" in dataset.features:
        victories = sum(dataset["victory"])
        print(
            f"Victory rate: {victories}/{len(dataset)} ({victories / len(dataset) * 100:.1f}%)"
        )

    if "score" in dataset.features:
        avg_score = sum(dataset["score"]) / len(dataset)
        print(f"Average score: {avg_score:.3f}")

    # Optionally upload to Hub
    if upload_to_hub and hub_repo_id:
        try:
            dataset.push_to_hub(hub_repo_id)
            print(f"Successfully uploaded dataset to {hub_repo_id}")
        except Exception as e:
            print(f"Failed to upload to Hub: {e}")


async def main():
    """Main function to generate and save training data."""
    from openai import AsyncOpenAI

    # Load environment variables from .env file
    load_dotenv()

    # Load battleship environment
    env = load_environment(num_games=NUM_GAMES, max_turns=MAX_TURNS)

    # Get required environment variables (no fallbacks)
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    # Validate required environment variables
    if not api_key:
        raise ValueError("OPENAI_API_KEY or OPENROUTER_API_KEY must be set")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL must be set")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    print("Starting battleship training data generation...")
    print(f"Using model: {OPENROUTER_MODEL_NAME}")
    print(f"Using base URL: {base_url}")
    print(
        f"Generating {NUM_SAMPLES} samples with {MAX_CONCURRENT} max concurrent rollouts"
    )
    print(f"Minimum score threshold: {MIN_SCORE_THRESHOLD}")

    # Generate training data
    dataset = await generate_battleship_training_data(
        env=env,
        client=client,
        model=OPENROUTER_MODEL_NAME,
        num_samples=NUM_SAMPLES,
    )

    # Save and optionally upload
    await save_and_upload_dataset(
        dataset=dataset,
        output_path=OUTPUT_PATH,
        upload_to_hub=UPLOAD_TO_HUB,
        hub_repo_id=HUB_REPO_ID,
    )


if __name__ == "__main__":
    asyncio.run(main())
