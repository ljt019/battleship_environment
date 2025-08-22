import verifiers as vf
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_NAME = "ljt019/battleship-sft-0825"

model, tokenizer = vf.get_tokenizer(
    MODEL_NAME,
)
dataset = load_dataset(DATASET_NAME, split="train")

first_row = dataset[0]

messages = first_row["prompt"] + first_row["completion"]

toks = tokenizer.apply_chat_template(messages, tokenize=True)
print(len(toks))
