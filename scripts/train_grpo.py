import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import verifiers as vf
from src.battleship_env import BattleshipMultiTurnEnv

# Set up detailed logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DebugTokenizer:
    """Wrapper to debug tokenization issues"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.call_count = 0
    
    def __getattr__(self, name):
        return getattr(self.tokenizer, name)
    
    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        self.call_count += 1
        logger.debug(f"\n=== TOKENIZER CALL #{self.call_count} ===")
        logger.debug(f"Messages count: {len(messages)}")
        for i, msg in enumerate(messages):
            content_preview = msg['content'][:100] + ('...' if len(msg['content']) > 100 else '')
            logger.debug(f"  Message {i}: role='{msg['role']}', content='{content_preview}'")
        
        result = self.tokenizer.apply_chat_template(messages, tokenize=tokenize, **kwargs)
        
        if tokenize and isinstance(result, list):
            logger.debug(f"Tokenized result length: {len(result)}")
            logger.debug(f"First 20 tokens: {result[:20]}")
            logger.debug(f"Last 20 tokens: {result[-20:]}")
        else:
            logger.debug(f"Text result length: {len(result) if isinstance(result, str) else 'N/A'}")
        
        logger.debug("=== END TOKENIZER CALL ===\n")
        return result

def main():
    model, tokenizer = vf.get_model_and_tokenizer("ljt019/Qwen3-1.7B-battleship-sft")
    
    # Wrap tokenizer with debug wrapper
    debug_tokenizer = DebugTokenizer(tokenizer)

    env = BattleshipMultiTurnEnv(max_turns=5)  # Reduced from 10 to avoid token limits
    
    run_name = "battleship-grpo-qwen3"
    training_args = vf.grpo_defaults(run_name=run_name)
    training_args.num_iterations = 3
    training_args.per_device_train_batch_size = 4
    training_args.num_generations = 16
    training_args.gradient_accumulation_steps = 2
    training_args.max_prompt_length = 8192
    training_args.max_completion_length = 4096
    training_args.max_steps = 500
    training_args.push_to_hub = True
    training_args.hub_model_id = "ljt019/Qwen3-1.7B-battleship-rlvr"
    
    logger.info("Starting GRPO training with debug logging...")
    
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=debug_tokenizer,
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
