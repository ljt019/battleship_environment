import sys
import os
import verifiers as vf
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import MODEL_SIZE, BASE_MODEL_NAME, LEARNING_RATE, NUM_TRAIN_EPOCHS, BATCH_SIZE, SFT_GRADIENT_ACCUMULATION_STEPS, MAX_COMPLETION_LENGTH, SFT_OUTPUT_DIR

model, tokenizer = vf.get_model_and_tokenizer(BASE_MODEL_NAME, use_liger=False)
dataset = load_dataset('ljt019/battleship-sft-cleaned', split='train')

# split top 500 highest reward samples for warmup 
dataset = dataset.sort("reward", reverse=True)
dataset = dataset.select(range(500))    

# Print token count statistics
tok_counts = []
for row in dataset:
    # count tokens in (prompt, completion)
    messages = row['prompt'] + row['completion']  # type: ignore
    toks = tokenizer.apply_chat_template(
        messages,
        tokenize=True
    )
    tok_counts.append(len(toks))

print(f"Dataset size: {len(tok_counts)}")
print(f"Min tokens: {min(tok_counts)}")
print(f"Max tokens: {max(tok_counts)}")
print(f"Mean tokens: {sum(tok_counts) / len(tok_counts)}")
print(f"Median tokens: {sorted(tok_counts)[len(tok_counts) // 2]}")

args = SFTConfig(
    max_length=MAX_COMPLETION_LENGTH,
    output_dir=SFT_OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=SFT_GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    weight_decay=0.01,
    max_grad_norm=0.1,
    report_to="wandb",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=1,
    save_only_model=True,
    log_on_each_node=True,
    push_to_hub=True,
    hub_model_id=f"Qwen3-{MODEL_SIZE}-Battleship-SFT",
)

def main():
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,  # type: ignore
    )
    trainer.train()

if __name__ == "__main__":
    main()


