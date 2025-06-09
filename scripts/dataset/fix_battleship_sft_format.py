"""
Fix the battleship SFT dataset format to be compatible with TRL SFTTrainer.

The original dataset has the issue where completion starts with a user message
instead of an assistant message, breaking chat template expectations.
"""

from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

def reformat_conversation(example):
    """Reformat the conversation to fix chat template issues"""
    prompt = example['prompt']  # This is already a list of messages
    completion = example['completion']  # This is a list of messages
    
    # Combine prompt and completion into a single conversation
    # Skip the first completion message if it's a user message (board state)
    # that should be part of the prompt instead
    full_conversation = prompt.copy()
    
    # If the first completion message is a user message (board state), 
    # add it to the prompt and start completion from the assistant response
    if completion and completion[0]['role'] == 'user':
        # Add the board state to the prompt
        full_conversation.append(completion[0])
        # The actual completion starts from the assistant message
        actual_completion = completion[1:]
    else:
        actual_completion = completion
    
    return {
        'messages': full_conversation + actual_completion,
        'stage': example['stage'],
        'move_number': example['move_number']
    }

def main():
    print("Loading original battleship SFT dataset...")
    dataset = load_dataset('ljt019/battleship-sft', split='train')
    
    print(f"Original dataset size: {len(dataset)}")
    print("Original format example:")
    print(f"  Prompt: {len(dataset[0]['prompt'])} messages")
    print(f"  Completion: {len(dataset[0]['completion'])} messages") 
    print(f"  First completion role: {dataset[0]['completion'][0]['role']}")
    
    # Transform the dataset
    print("\nReformatting conversations...")
    formatted_dataset = dataset.map(reformat_conversation, remove_columns=['prompt', 'completion'])
    
    print(f"Formatted dataset size: {len(formatted_dataset)}")
    print("New format example:")
    print(f"  Messages: {len(formatted_dataset[0]['messages'])} messages")
    print(f"  Message roles: {[msg['role'] for msg in formatted_dataset[0]['messages']]}")
    
    # Verify the format is correct
    print("\nVerifying format...")
    for i, example in enumerate(formatted_dataset):
        messages = example['messages']
        
        # Check that conversation flows properly
        prev_role = None
        for j, msg in enumerate(messages):
            role = msg['role']
            if j == 0 and role != 'system':
                print(f"  WARNING: Example {i} doesn't start with system message")
            if j > 0 and role == prev_role and role != 'user':
                print(f"  WARNING: Example {i} has consecutive {role} messages at position {j}")
            prev_role = role
    
    print("Format verification complete.")
    
    # Save locally first
    local_path = "datasets/battleship_sft_fixed"
    formatted_dataset.save_to_disk(local_path)
    print(f"Saved locally to: {local_path}")
    
    # Upload to HuggingFace Hub - overwrite the original dataset
    print("Uploading to HuggingFace Hub (overwriting original)...")
    try:
        formatted_dataset.push_to_hub(
            "ljt019/battleship-sft",
            commit_message="Fix chat template format - move board state to prompt"
        )
        print("✅ Successfully updated ljt019/battleship-sft")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("You may need to login first: huggingface-cli login")
    
    print("\nDataset fixing complete!")
    print("SFT script can now use the original dataset name: ljt019/battleship-sft")

if __name__ == "__main__":
    main() 