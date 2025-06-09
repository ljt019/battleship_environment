from datasets import Dataset
from huggingface_hub import login
import os

def upload_sft_dataset():
    print("Uploading battleship SFT dataset to HuggingFace Hub...")
    
    # Load the local dataset
    try:
        dataset = Dataset.load_from_disk("datasets/battleship_sft")
        print(f"Loaded dataset with {len(dataset)} examples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Run generate_reasoning_with_claude.py first")
        return False
    
    # Authenticate with HuggingFace
    try:
        login()
        print("HuggingFace authentication successful")
    except Exception as e:
        print(f"HuggingFace login failed: {e}")
        print("Run 'huggingface-cli login' first or set HUGGINGFACE_TOKEN")
        return False
    
    # Upload to Hub
    dataset_name = "ljt019/battleship-sft"
    
    try:
        print(f"Uploading to {dataset_name}...")
        
        dataset.push_to_hub(
            dataset_name,
            private=False,  # Set to True for private upload
            commit_message="Upload battleship SFT dataset with reasoning traces"
        )
        
        print(f"Dataset uploaded: https://huggingface.co/datasets/{dataset_name}")
        print(f"Total examples: {len(dataset)}")
        print("\nDataset includes:")
        print("- Strategic reasoning traces in <think> tags")
        print("- Expert battleship move decisions")
        print("- Conversation format for SFT training")
        return True
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return False

if __name__ == "__main__":
    upload_sft_dataset() 