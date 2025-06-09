from datasets import Dataset
from huggingface_hub import login
import os

def upload_board_dataset():
    print("Uploading battleship board states dataset to HuggingFace Hub...")
    
    # Load the local dataset
    try:
        dataset = Dataset.load_from_disk("datasets/battleship_board_states")
        print(f"Loaded dataset with {len(dataset)} examples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Run create_board_dataset.py first")
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
    dataset_name = "ljt019/battleship-board-states"
    
    try:
        print(f"Uploading to {dataset_name}...")
        
        dataset.push_to_hub(
            dataset_name,
            private=False,  # Set to True for private upload
            commit_message="Upload battleship board states dataset"
        )
        
        print(f"Dataset uploaded: https://huggingface.co/datasets/{dataset_name}")
        print(f"Total examples: {len(dataset)}")
        return True
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return False

if __name__ == "__main__":
    upload_board_dataset() 