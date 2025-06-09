import verifiers as vf
from scripts.battleship_env import BattleshipEnv

size = '1.7B'
model_name = f'ljt019/Qwen3-{size}-Battleship-SFT'
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = BattleshipEnv(
    num_samples=2000, 
    num_eval_samples=20,
    max_concurrent=32,
)

run_name = f"battleship-grpo-{size}"
training_args=vf.grpo_defaults(run_name=run_name)
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
