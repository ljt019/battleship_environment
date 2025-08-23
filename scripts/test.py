import os

from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI

from battleship_environment import load_environment

################ Eval Config ################

MAX_TURNS = 5
NUM_GAMES = 100

OPENROUTER_MODEL_NAME = "qwen/qwen3-30b-a3b-instruct-2507"

#############################################

load_dotenv()

# Get required environment variables
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

client = OpenAI(api_key=api_key, base_url=base_url)

env = load_environment(max_turns=MAX_TURNS, num_games=NUM_GAMES)


results = env.evaluate(
    client,
    OPENROUTER_MODEL_NAME,
    num_examples=1,
    rollouts_per_example=1,
    max_concurrent=1,
)

# Extract data from results
result_dict = dict(list(results))
messages = result_dict["prompt"][0] + result_dict["completion"][0]

Dataset.from_list(messages).push_to_hub("battleship-env-test")
