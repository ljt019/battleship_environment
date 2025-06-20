import random
from typing import List, Dict

from datasets import Dataset
from openai import OpenAI

from battleship_grpo import BattleshipEnv

from config import (
    OPENROUTER_BASE_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)

from local_openai import LocalOpenAI, StrategyRunnerBackend
from local_openai.backend.strategies.information_gain import InformationGainStrategy

TOP_PERCENTAGE = 1
NUM_SAMPLES = 1
MAX_TURNS = 3

def main():
    local_client = LocalOpenAI(
        StrategyRunnerBackend(strategy=InformationGainStrategy())
    )

    vf_env = BattleshipEnv(
        seed=random.randint(0, 1_000_000),
        max_concurrent=32,
        max_turns=MAX_TURNS,
    )

    results = vf_env.evaluate(
        client=local_client,
        model="dummy-model-name",
        num_samples=NUM_SAMPLES,
    )

    games = list(vf_env.make_dataset(results))
    games.sort(key=lambda g: g["reward"], reverse=True)
    top_n = int(len(games) * TOP_PERCENTAGE)
    top_games = games[:top_n]

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

    openrouter_client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    for turn in turn_rows:
        turn_prompt = turn["prompt"]
        turn_completion = turn["completion"]

        game_prompt = turn_prompt + turn_completion
        system_prompt = (
            """
            You are filling in the missing <think> traces for the following move in the game of battleship. 
            Generate your thought process as if you were deciding how to make the move, not as if it was already decided. 
            Only respond with the content of the <think> tag; everything in your response will be placed between the tags.
            """
        )

        openrouter_response = openrouter_client.chat.completions.create(
            model=OPENROUTER_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": game_prompt},
            ],
        )

        turn_completion[0]["content"] = (
            turn_completion[0]["content"]
            .replace("<think>", openrouter_response.choices[0].message.content)
            .replace("</think>", "")
        )
        turn["completion"] = turn_completion

    dataset = Dataset.from_list(turn_rows)
    dataset.push_to_hub("battleship-synthetic-games-dataset")


if __name__ == "__main__":
    main() 