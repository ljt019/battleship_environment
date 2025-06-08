import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import verifiers as vf
from src.battleship_env import BattleshipSingleTurnEnv

def main():
    model, tokenizer = vf.get_model_and_tokenizer("ljt019/Qwen3-1.7B-battleship-sft")

    env = BattleshipSingleTurnEnv()
    
    run_name = "battleship-grpo-qwen3"
    training_args = vf.grpo_defaults(run_name=run_name)
    training_args.num_iterations = 3
    training_args.per_device_train_batch_size = 4
    training_args.num_generations = 16
    training_args.gradient_accumulation_steps = 2
    training_args.max_prompt_length = 2048
    training_args.max_completion_length = 256
    training_args.max_steps = 500
    training_args.push_to_hub = True
    training_args.hub_model_id = "ljt019/Qwen3-1.7B-battleship-rlvr"
    
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        args=training_args,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
