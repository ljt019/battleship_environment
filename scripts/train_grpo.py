import verifiers as vf
from scripts.battleship_env import BattleshipEnv
from config import MODEL_SIZE, SFT_MODEL_NAME, NUM_SAMPLES, NUM_EVAL_SAMPLES, MAX_TURNS, GRPO_RUN_NAME

model, tokenizer = vf.get_model_and_tokenizer(SFT_MODEL_NAME)

vf_env = BattleshipEnv(
    num_samples=NUM_SAMPLES, 
    num_eval_samples=20,  # Keep smaller for training
    max_concurrent=32,
    max_turns=MAX_TURNS
)

training_args=vf.grpo_defaults(run_name=GRPO_RUN_NAME)
training_args.num_iterations=1
training_args.per_device_train_batch_size=6
training_args.num_generations=12
training_args.gradient_accumulation_steps=4
training_args.max_prompt_length=1024
training_args.max_completion_length=4096
training_args.max_steps=100
training_args.mask_env_responses=True

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
