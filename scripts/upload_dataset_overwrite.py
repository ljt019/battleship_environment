import sys
import os
from datasets import load_from_disk

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import DATASET_PATH, HUB_DATASET_NAME

"""Simple uploader: always overwrite the remote train split with the local dataset.
Run with:  uv run scripts/upload_dataset_overwrite.py
"""

def main():
    print(f"Loading local dataset from {DATASET_PATH}")
    try:
        local_dataset = load_from_disk(DATASET_PATH)
        sample_count = len(local_dataset)
        print(f"Loaded {sample_count} samples from disk.")

        # Abort if the dataset is empty so we don't wipe the remote with nothing.
        if sample_count == 0:
            print("ERROR: Local dataset is empty – aborting upload to avoid overwriting remote dataset with no data.")
            return 1

        print(f"Pushing to {HUB_DATASET_NAME} (overwrite mode)…")
        # Push in overwrite mode: single commit containing only local rows
        local_dataset.push_to_hub(HUB_DATASET_NAME, private=False, token=None, split="train")
        print("Upload complete – remote dataset replaced.")
    except FileNotFoundError:
        print(f"Dataset not found at {DATASET_PATH}. Generate it first.")
        return 1
    except Exception as e:
        print(f"Upload failed: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main()) 