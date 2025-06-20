from datasets import load_dataset, concatenate_datasets

first_dataset = load_dataset("ljt019/battleship-sft-synthetic", split="train")
second_dataset = load_dataset("ljt019/battleship-sft-synthetic-two", split="train")

# Merge the two datasets
merged_datasets = concatenate_datasets([first_dataset, second_dataset])

merged_datasets.push_to_hub("battleship-sft-synthetic-merged")
