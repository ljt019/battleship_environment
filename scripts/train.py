import verifiers as vf

from battleship_environment import load_environment

############## Training Config ##############

MAX_TURNS = 10
NUM_GAMES = 100

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

RUN_NAME = "battleship-sft-0825"

#############################################

env = load_environment(max_turns=MAX_TURNS, num_games=NUM_GAMES)

model, tokenizer = vf.get_model_and_tokenizer(MODEL_NAME)

args = vf.grpo_defaults(run_name=RUN_NAME)

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
)

trainer.train()
