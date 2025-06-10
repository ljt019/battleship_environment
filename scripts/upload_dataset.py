import sys
import os
from datasets import load_from_disk

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import DATASET_PATH, HUB_DATASET_NAME

def main():
    print(f"Loading dataset from {DATASET_PATH}")
    
    try:
        dataset = load_from_disk(DATASET_PATH)
        print(f"Loaded {len(dataset)} samples")
        
        if 'reward' in dataset.column_names:
            rewards = dataset['reward']
            avg_reward = sum(rewards) / len(rewards)
            win_rate = sum(1 for r in rewards if r > 0.5) / len(rewards) * 100
            print(f"Average reward: {avg_reward:.3f}, Win rate: {win_rate:.1f}%")
        
        print(f"Uploading to {HUB_DATASET_NAME}")
        dataset.push_to_hub(HUB_DATASET_NAME)
        print("Upload complete")
        
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