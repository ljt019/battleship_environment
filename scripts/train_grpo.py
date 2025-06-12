import sys
import os
# Enable PyTorch allocator to reuse memory across segments
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import verifiers as vf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.battleship_grpo import BattleshipEnv
from scripts.config import (
    MODEL_SIZE,
    BATCH_SIZE,
    SFT_MODEL_NAME,
    GRPO_GRADIENT_ACCUMULATION_STEPS,
    MAX_PROMPT_LENGTH,
    MAX_COMPLETION_LENGTH,
    NUM_GRPO_SAMPLES,
    NUM_GRPO_EVAL_SAMPLES,
    MAX_TURNS,
    GRPO_RUN_NAME,
    MAX_CONCURRENT_API,
    GRPO_NUM_GENERATIONS,
)

# Load the model and immediately cast to bfloat16 to halve memory usage for activations/gradients.
model, tokenizer = vf.get_model_and_tokenizer("ljt019/Qwen3-1.7B-battleship-grpo")
model = model.to(torch.bfloat16)

vf_env = BattleshipEnv(
    num_samples=NUM_GRPO_SAMPLES, 
    num_eval_samples=NUM_GRPO_EVAL_SAMPLES, 
    max_concurrent=MAX_CONCURRENT_API,
    max_turns=MAX_TURNS
)

training_args=vf.grpo_defaults(run_name=GRPO_RUN_NAME)
training_args.num_iterations=75
training_args.per_device_train_batch_size=BATCH_SIZE
# Each prompt will generate `num_generations` completions per rollout.
# This **must** match the value used inside GRPOTrainer's async batch generator.
# We currently create 4 completions per prompt, so keep these in sync.
training_args.num_generations=GRPO_NUM_GENERATIONS
training_args.gradient_accumulation_steps=GRPO_GRADIENT_ACCUMULATION_STEPS
training_args.max_prompt_length=MAX_PROMPT_LENGTH
training_args.max_completion_length=MAX_COMPLETION_LENGTH
training_args.max_steps = 15000  # extended training horizon
training_args.mask_env_responses=True
training_args.bf16 = True  # enable automatic mixed-precision training (bf16)
training_args.gradient_checkpointing = True  # save memory at the cost of extra compute
training_args.beta = 0.05  # stronger KL penalty to keep divergence in check
training_args.learning_rate = 5e-7  # slightly lower LR for stability
training_args.save_strategy = "steps"
training_args.save_steps = 1000
training_args.save_total_limit = 15

def main():
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
        push_to_hub=True,
        hub_model_id=f"Qwen3-{MODEL_SIZE}-Battleship-GRPO-beta",
    )
    trainer.train()

if __name__ == "__main__":
    main()
