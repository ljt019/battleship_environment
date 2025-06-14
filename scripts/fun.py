import os
import sys
from verifiers.parsers.xml_parser import XMLParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.battleship_grpo import BattleshipEnv
from src.battleship_grpo.battleship_env import BATTLESHIP_SYSTEM_PROMPT, BATTLESHIP_RULES
from scripts.config import SEED, MAX_TURNS



# Create the enviroment 
env = BattleshipEnv(
    num_samples=100,
    num_eval_samples=100,
    seed=SEED,
    max_turns=MAX_TURNS,
)

# setup parsers 
env_response_parser = XMLParser(fields=["state", "grid", "remaining", "result"])
model_response_parser = XMLParser(fields=["think", "guess"])

intial_response, state = env.env_response(
    messages=[],
    state={},
)

print(intial_response)

env_response = env.env_response(
    messages=[
        {"role": "system", "content": BATTLESHIP_SYSTEM_PROMPT},
        {"role": "user", "content": BATTLESHIP_RULES},
        {"role": "assistant", "content": "<think>I think i'll hit the center of the board</think>\n\n<guess>[e5]</guess>"},
    ],
    state=state,
)

print(env_response)

env_response, state = env.env_response(
    messages=[
        {"role": "system", "content": BATTLESHIP_SYSTEM_PROMPT},
        {"role": "user", "content": BATTLESHIP_RULES},
        {"role": "assistant", "content": "<think>I think i'll hit the center of the board</think>\n\n<guess>[e6]</guess>"},
    ],
    state=state,
)

print(env_response)