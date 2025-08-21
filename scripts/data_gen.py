import asyncio
import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv

from battleship_environment import load_environment

################ Data Gen Config #################

NUM_SAMPLES = 10  # each sample creates 3 examples at diff temps
NUM_GAMES = 1000
MAX_TURNS = 50

OPENROUTER_MODEL_NAME = "qwen/qwen3-235b-a22b-2507"

UPLOAD_TO_HUB = True
HUB_REPO_ID = "ljt019/battleship-sft-0825"

OUTPUT_PATH = "data"

MIN_SCORE_THRESHOLD = 0.6

##################################################


async def generate_battleship_training_data(
    env, client, model: str, num_samples: int = 1000
) -> Dataset:
    """
    Generate diverse battleship game solutions for training using verifiers rollouts.

    Args:
        env: BattleshipEnv instance
        client: OpenAI/Anthropic client for API calls
        model: Model name to use for generation
        num_samples: Number of training samples to generate

    Returns:
        HuggingFace Dataset with training examples
    """
    results = []

    for i in range(num_samples):
        print(f"Generating sample {i + 1}/{num_samples}")

        # Get a random game setup from dataset
        example = env.dataset[i % len(env.dataset)]
        seed = example.get("seed", i)
        answer = example["answer"]  # "victory" for battleship

        # Generate multiple solutions with different temperatures for diversity
        for temp in [0.3, 0.7, 1.0]:
            try:
                # Use verifiers' built-in rollout method
                # For multi-turn envs, start with system prompt
                system_prompt = (
                    env.system_prompt
                    if hasattr(env, "system_prompt")
                    else "You are playing Battleship."
                )
                completion, state = await env.rollout(
                    client=client,
                    model=model,
                    prompt=[{"role": "system", "content": system_prompt}],
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

                # Save high-quality solutions
                # Extract total reward and individual metrics from RolloutScore object
                total_reward = (
                    float(rewards.reward) if hasattr(rewards, "reward") else 0.0
                )

                # Extract individual reward function scores
                metrics = rewards.metrics if hasattr(rewards, "metrics") else {}
                victory_reward = metrics.get("victory_reward", 0.0)
                hit_reward = metrics.get("hit_reward", 0.0)
                strategic_hit_reward = metrics.get("strategic_hit_reward", 0.0)
                coverage_efficiency_reward = metrics.get(
                    "coverage_efficiency_reward", 0.0
                )
                format_reward = metrics.get("format_reward_func", 0.0)

                if total_reward > MIN_SCORE_THRESHOLD:
                    results.append(
                        {
                            "game_seed": seed,
                            "temperature": temp,
                            "completion": completion,
                            "reward": total_reward,
                            "victory_reward": victory_reward,
                            "hit_reward": hit_reward,
                            "strategic_hit_reward": strategic_hit_reward,
                            "coverage_efficiency_reward": coverage_efficiency_reward,
                            "format_reward": format_reward,
                            "victory": state.get("victory", False),
                            "turns": state.get("turn", 0),
                        }
                    )
                    print(
                        f"  ✓ Added sample (temp={temp}, reward={total_reward:.3f}, victory={state.get('victory', False)})"
                    )
                else:
                    print(
                        f"  ✗ Rejected sample (temp={temp}, reward={total_reward:.3f})"
                    )

            except Exception as e:
                print(f"  ✗ Error generating sample {i} (temp={temp}): {e}")
                import traceback

                print(f"     Full traceback: {traceback.format_exc()}")
                continue

    print(f"\nGenerated {len(results)} high-quality training examples")
    return Dataset.from_list(results)


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

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    print("Starting battleship training data generation...")
    print(f"Using model: {OPENROUTER_MODEL_NAME}")
    print(f"Using base URL: {base_url}")
    print(f"Generating {NUM_SAMPLES} samples")
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
