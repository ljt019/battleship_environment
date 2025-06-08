import verifiers as vf
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

def analyze_token_counts(dataset, tokenizer):
    """Analyze token counts in dataset"""
    tok_counts = []
    for row in dataset:
        # count tokens in (prompt, completion)
        messages = row['prompt'] + row['completion']
        toks = tokenizer.apply_chat_template(messages, tokenize=True)
        tok_counts.append(len(toks))

    print(f"Dataset size: {len(tok_counts)}")
    print(f"Min tokens: {min(tok_counts)}")
    print(f"Max tokens: {max(tok_counts)}")
    print(f"Mean tokens: {sum(tok_counts) / len(tok_counts):.1f}")
    print(f"Median tokens: {sorted(tok_counts)[len(tok_counts) // 2]}")
    
    return max(tok_counts)

def main():
    MODEL_NAME = "Qwen/Qwen3-1.7B"
    OUTPUT_DIR = "outputs/battleship_sft_model"
    
    model, tokenizer = vf.get_model_and_tokenizer(MODEL_NAME, use_liger=False)
    dataset = load_dataset('ljt019/battleship-rlvr-qwen3-dataset', split='train')
    
    # print stats about the dataset
    analyze_token_counts(dataset, tokenizer)

    args = SFTConfig(
        max_length=4096,
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        max_grad_norm=0.1,
        report_to="wandb",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        save_only_model=True,
        log_on_each_node=True,
        run_name="battleship-sft",
        push_to_hub=True,
        hub_model_id="ljt019/Qwen3-1.7B-battleship-sft"
    )
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset 
    )
    
    # Train
    print("Starting SFT training...")
    trainer.train()
    
    print(f"SFT training complete! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()


