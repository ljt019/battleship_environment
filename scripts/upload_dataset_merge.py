import sys
import os
from datasets import load_from_disk, load_dataset, concatenate_datasets

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import DATASET_PATH, HUB_DATASET_NAME

def main():
    print(f"Loading local dataset from {DATASET_PATH}")
    
    try:
        local_dataset = load_from_disk(DATASET_PATH)
        print(f"Loaded {len(local_dataset)} new samples")
        
        if 'reward' in local_dataset.column_names:
            rewards = local_dataset['reward']
            avg_reward = sum(rewards) / len(rewards)
            win_rate = sum(1 for r in rewards if r > 0.5) / len(rewards) * 100
            print(f"New samples - Average reward: {avg_reward:.3f}, Win rate: {win_rate:.1f}%")
        
        # Merge with existing remote dataset if it already exists
        try:
            print(f"Attempting to load existing dataset from {HUB_DATASET_NAME}")
            remote_dataset = load_dataset(HUB_DATASET_NAME, split="train")
            print(f"Found existing dataset with {len(remote_dataset)} rows – merging")

            combined_dataset = concatenate_datasets([remote_dataset, local_dataset])
            print(f"Combined dataset size: {len(combined_dataset)} rows")

        except Exception as e:
            print(f"No existing dataset found or failed to load ({e}). Creating new repo.")
            combined_dataset = local_dataset

        print(f"Pushing {len(combined_dataset)} rows to {HUB_DATASET_NAME} (will create a new commit)")
        combined_dataset.push_to_hub(HUB_DATASET_NAME, private=False, token=None, split='train')
        print("Upload complete – remote repository updated")
        
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print("Run dataset generation first: uv run scripts/create_sft_data.py")
        return 1
        
    except Exception as e:
        print(f"Upload failed: {e}")
        print("Make sure you're logged in: huggingface-cli login")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 