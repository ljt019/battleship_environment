import verifiers as vf
from src.battleship_grpo import BattleshipEnv
from scripts.config import MODEL_SIZE, BATCH_SIZE, SFT_MODEL_NAME, GRPO_GRADIENT_ACCUMULATION_STEPS, MAX_PROMPT_LENGTH, MAX_COMPLETION_LENGTH, NUM_GRPO_SAMPLES, NUM_GRPO_EVAL_SAMPLES, MAX_TURNS, GRPO_RUN_NAME, MAX_CONCURRENT_API

model, tokenizer = vf.get_model_and_tokenizer(SFT_MODEL_NAME)

vf_env = BattleshipEnv(
    num_samples=NUM_GRPO_SAMPLES, 
    num_eval_samples=NUM_GRPO_EVAL_SAMPLES, 
    max_concurrent=MAX_CONCURRENT_API,
    max_turns=MAX_TURNS
)

training_args=vf.grpo_defaults(run_name=GRPO_RUN_NAME)
training_args.num_iterations=1
training_args.per_device_train_batch_size=BATCH_SIZE
training_args.num_generations=12
training_args.gradient_accumulation_steps=GRPO_GRADIENT_ACCUMULATION_STEPS
training_args.max_prompt_length=MAX_PROMPT_LENGTH
training_args.max_completion_length=MAX_COMPLETION_LENGTH
training_args.max_steps=100
training_args.mask_env_responses=True

def main():
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
        push_to_hub=True,
        hub_model_id=f"Qwen3-{MODEL_SIZE}-Battleship-GRPO",
    )
    trainer.train()

if __name__ == "__main__":
    main()
