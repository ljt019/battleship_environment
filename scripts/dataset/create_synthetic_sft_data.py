import random
import re
from typing import List, Dict
import asyncio
import math

from datasets import Dataset
from openai import AsyncOpenAI

from battleship_grpo import BattleshipEnv

from config import (
    OPENROUTER_BASE_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)

from local_openai import LocalOpenAI, StrategyRunnerBackend
from local_openai.backend.strategies.information_gain import InformationGainStrategy

from tqdm import tqdm

DESIRED_TURNS  = 100
TOP_PERCENTILE = 0.001
MAX_TURNS       = 65

MAX_CONCURRENT  = 10

# ----- Derived value: minimum turns per game in worst-case
TURNS_PER_GAME_WORST = 17 if MAX_TURNS >= 17 else MAX_TURNS

# Number of raw games to run so that even in the worst case, after
# filtering to the top percentile, we still have ≥ DESIRED_TURNS turns.
NUM_SYNTHETIC_GAMES = math.ceil(DESIRED_TURNS / (TURNS_PER_GAME_WORST * TOP_PERCENTILE))

def main():
    local_client = LocalOpenAI(
        StrategyRunnerBackend(strategy=InformationGainStrategy())
    )

    vf_env = BattleshipEnv(
        num_samples=NUM_SYNTHETIC_GAMES,
        num_eval_samples=NUM_SYNTHETIC_GAMES,
        seed=random.randint(0, 1_000_000),
        max_concurrent=32,
        max_turns=MAX_TURNS,
    )

    results = vf_env.evaluate(
        client=local_client,
        model="dummy-model-name",
        num_samples=NUM_SYNTHETIC_GAMES,
    )

    games = list(vf_env.make_dataset(results))
    games.sort(key=lambda g: g["reward"], reverse=True)

    # Keep the best TOP_PERCENTILE fraction (at least one game)
    top_cutoff = max(1, int(len(games) * TOP_PERCENTILE))
    top_games = games[:top_cutoff]

    turn_rows: List[Dict] = []
    for game_idx, game in enumerate(top_games, start=1):
        turn_counter = 0 
        full_prompt = game["prompt"]
        full_completion = game["completion"]
        msgs = full_prompt + full_completion
        episode_reward = game["reward"]

        static_msgs = full_prompt[:2]

        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue

            board_msg = None
            for j in range(i - 1, -1, -1):
                if (
                    msgs[j].get("role") == "user"
                    and "<grid>" in msgs[j].get("content", "")
                ):
                    board_msg = msgs[j]
                    break

            if board_msg is None:
                continue

            prompt_turn = static_msgs + [board_msg]
            completion_turn = [m]

            turn_counter += 1

            turn_rows.append(
                {
                    "prompt": prompt_turn,
                    "completion": completion_turn,
                    "reward": episode_reward,
                    # Annotate the answer field with game and turn numbers for easier tracking
                    "answer": f"game {game_idx}, turn {turn_counter}",
                    "task": game.get("task", None),
                }
            )

    async def fill_reasonings(turn_rows: List[Dict]):
        """Populate empty <think> tags concurrently using AsyncOpenAI."""

        client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )

        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        # Progress bar for overall processing
        pbar = tqdm(total=len(turn_rows), desc="Filling in reasoning for turns", position=0)

        async def process_turn(turn: Dict):
            prompt = turn["prompt"]
            turn_completion = turn["completion"]

            # Only process turns with an empty think block
            if not re.search(r"<think>\s*</think>", turn_completion[0]["content"], re.DOTALL):
                return

            system_prompt = prompt[0]["content"]
            game_rules_prompt = prompt[1]["content"]
            turn_prompt = prompt[2]["content"]

            async with semaphore:
                response = await client.chat.completions.create(
                    model=OPENROUTER_MODEL,
                    temperature=0.3,
                    stop=["</think>"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": game_rules_prompt},
                        {"role": "user", "content": turn_prompt},
                        {"role": "assistant", "content": turn_completion[0]["content"]},
                        {
                            "role": "user",
                            "content": (
                                "You previously suggested a Battleship <guess>...</guess> move but left the <think>...</think> reasoning empty.\n"
                                "Respond ONLY with your reasoning wrapped exactly in the tags <think> … </think>.\n"
                                "Do not output anything else. Your <guess>...</guess> will be reused and the empty <think>...</think> will be replaced with your reasoning."
                            ),
                        },
                    ],
                )

            reply = response.choices[0].message.content.strip()

            # Clean and extract reasoning
            reply = reply.replace("\n", " ").strip()
            if reply.startswith("<think>"):
                reply = reply[len("<think>"):].lstrip()
            if reply.endswith("</think>"):
                reply = reply[: -len("</think>")].rstrip()

            reasoning = reply.strip()

            # Replace the empty think block
            turn_completion[0]["content"] = re.sub(
                r"<think>\s*</think>", f"<think>{reasoning}</think>", turn_completion[0]["content"], count=1
            )

            turn["completion"] = turn_completion

            # Update progress bar after each turn, regardless of whether it needed processing
            pbar.update(1)

        # Launch tasks with concurrency control
        await asyncio.gather(*(process_turn(t) for t in turn_rows))

        pbar.close()

    # Run the async processing only on the first DESIRED_TURNS turns
    turn_rows = turn_rows[:DESIRED_TURNS]

    # Run the async processing
    asyncio.run(fill_reasonings(turn_rows))

    dataset = Dataset.from_list(turn_rows)
    dataset.push_to_hub("battleship-synthetic-games-dataset")


if __name__ == "__main__":
    main() 