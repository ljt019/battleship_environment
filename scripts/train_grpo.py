import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import verifiers as vf
from src.battleship_env import BattleshipMultiTurnEnv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    model, tokenizer = vf.get_model_and_tokenizer("ljt019/Qwen3-1.7B-battleship-sft")

    # Use the corrected MultiTurnEnv following TextArenaEnv pattern
    env = BattleshipMultiTurnEnv(max_turns=10)
    
    run_name = "battleship-grpo-qwen3"
    training_args = vf.grpo_defaults(run_name=run_name)
    training_args.num_iterations = 3
    training_args.per_device_train_batch_size = 4
    training_args.num_generations = 16
    training_args.gradient_accumulation_steps = 2
    training_args.max_prompt_length = 4096  # Reasonable for multi-turn
    training_args.max_completion_length = 512  # Reasonable for multi-turn
    training_args.max_steps = 500
    training_args.push_to_hub = True
    training_args.hub_model_id = "ljt019/Qwen3-1.7B-battleship-rlvr"
    training_args.mask_env_responses = True  # Key parameter from TextArenaEnv
    
    logger.info("Starting GRPO training with truly deterministic game states...")
    
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        args=training_args
    )
    
    trainer.train()

    # Auto-shutdown to save GPU costs
    print("Training complete! Shutting down in 60 seconds...")
    import time
    import os
    time.sleep(60)  # Give time to see the completion message
    os.system("shutdown -h now")

if __name__ == "__main__":
    main()
