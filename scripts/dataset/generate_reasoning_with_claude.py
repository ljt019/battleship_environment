import json
from datasets import Dataset, load_dataset
from pathlib import Path
import anthropic
import os
from typing import List, Dict
import time
import re

def add_reasoning_to_move(client: anthropic.Anthropic, conversation_context: List[Dict], assistant_move: str) -> str:
    """Generate strategic reasoning for an assistant move given conversation context"""
    
    context_messages = conversation_context[-3:] if len(conversation_context) >= 3 else conversation_context
    
    context_str = ""
    for msg in context_messages:
        if msg["role"] == "user":
            content = msg["content"]
            if "board:" in content.lower() or "[" in content:
                context_str += f"Current situation:\n{content}\n\n"
    
    move_complexity = classify_move_complexity(assistant_move, context_messages)
    
    system_content = "You are a competitive battleship player. Make sure you read the game instructions carefully, and always follow the required format.\n\nIn each turn, think step-by-step inside <think>...</think> tags, then make your move inside <guess>...</guess> tags."
    
    rules_content = """You are playing Battleship against an opponent.
Your goal is to sink all enemy ships by guessing their locations.
The board shows:
  [?] = unknown squares
  [x] = hit (part of a ship)
  [o] = miss (water)
  [s] = sunk ship part

For each move, use <think>...</think> tags for reasoning, then <guess>[coordinate]</guess> for your move.
Make strategic moves to find and sink all ships efficiently."""

    reasoning_instructions = get_reasoning_instructions(move_complexity)
    
    prompt = f"""{rules_content}

{context_str}

{reasoning_instructions}

You need to make this exact move: {assistant_move}

Add your strategic reasoning before the move using this format:
<think>
[Your reasoning for why this move is strategically sound]
</think>

{assistant_move}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            temperature=0.6,
            system=system_content,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
        
    except Exception as e:
        print(f"Error generating reasoning: {e}")
        return f"""<think>
I'm choosing this move based on battleship strategy principles, targeting areas with high probability of containing ships or adjacent to previous hits.
</think>

{assistant_move}"""

def classify_move_complexity(assistant_move: str, context_messages: List[Dict]) -> str:
    """Classify the complexity of a move to determine reasoning depth"""
    
    coord_match = re.search(r'\[([a-j][0-9]+)\]', assistant_move)
    if not coord_match:
        return "strategic_decision"
    
    recent_context = "".join([msg.get("content", "") for msg in context_messages[-2:]])
    
    if "Hit!" in recent_context and "x" in recent_context:
        return "obvious_continuation"
    
    if len(context_messages) < 4:
        return "strategic_decision"
    
    return "standard_move"

def get_reasoning_instructions(move_complexity: str) -> str:
    """Get reasoning instructions based on move complexity"""
    
    base_instructions = """In your reasoning:
- Express uncertainty when appropriate using phrases like "likely", "probably", "I'm not certain but..."
- Consider 1-2 alternative moves briefly and explain why you chose this one instead
- Be natural and conversational in your thinking"""
    
    if move_complexity == "obvious_continuation":
        return f"""{base_instructions}
- Keep reasoning concise since this appears to be a clear follow-up move
- Focus on the immediate tactical situation"""
        
    elif move_complexity == "strategic_decision":
        return f"""{base_instructions}
- Provide detailed reasoning since this appears to be a strategic positioning decision
- Consider multiple factors: ship placement probabilities, search patterns, etc.
- Explain your overall strategy"""
        
    else:
        return f"""{base_instructions}
- Provide moderate detail in your reasoning
- Balance tactical and strategic considerations"""

def enhance_conversation_with_reasoning(conversation: List[Dict], client: anthropic.Anthropic) -> List[Dict]:
    """Add reasoning traces to assistant moves in a conversation"""
    
    enhanced_conversation = []
    
    for i, message in enumerate(conversation):
        if message["role"] == "assistant":
            current_move = message["content"]
            context = conversation[:i]
            enhanced_move = add_reasoning_to_move(client, context, current_move)
            
            enhanced_message = {
                "role": "assistant",
                "content": enhanced_move
            }
            enhanced_conversation.append(enhanced_message)
            time.sleep(0.3)
            
        else:
            enhanced_conversation.append(message)
    
    return enhanced_conversation

def process_dataset_entries(dataset_entries: List[Dict], client: anthropic.Anthropic, max_entries: int = None) -> List[Dict]:
    """Process dataset entries and add reasoning to conversations"""
    
    enhanced_entries = []
    entries_to_process = dataset_entries[:max_entries] if max_entries else dataset_entries
    
    for i, entry in enumerate(entries_to_process):
        print(f"Processing conversation {i+1}/{len(entries_to_process)}...")
        
        prompt = entry.get('prompt', [])
        completion = entry.get('completion', [])
        
        if not completion:
            print(f"Skipping entry {i+1} - no completion found")
            continue
        
        enhanced_completion = enhance_conversation_with_reasoning(completion, client)
        
        enhanced_entry = {
            "prompt": prompt,
            "completion": enhanced_completion
        }
        
        enhanced_entries.append(enhanced_entry)
        
        print(f"  Enhanced {len([m for m in completion if m['role'] == 'assistant'])} assistant moves")
    
    return enhanced_entries

def main():
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    print("Loading battleship conversation dataset from HuggingFace...")
    
    try:
        dataset = load_dataset("ljt019/battleship-board-states", split="train")
        print(f"Loaded {len(dataset)} conversation examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    max_entries = 5
    print(f"Processing {max_entries} conversations...")
    
    enhanced_entries = process_dataset_entries(list(dataset), client, max_entries)
    
    if enhanced_entries:
        enhanced_dataset = Dataset.from_list(enhanced_entries)
        enhanced_dataset.save_to_disk('datasets/battleship_sft')
        print(f"\nEnhanced SFT dataset saved with {len(enhanced_entries)} conversations")
        
        print("\n=== SAMPLE ENHANCED ENTRY ===")
        sample = enhanced_entries[0]
        print("PROMPT structure preserved")
        print(f"COMPLETION: {len(sample['completion'])} messages")
        
        for msg in sample["completion"]:
            if msg["role"] == "assistant":
                print(f"\nFirst enhanced assistant move:")
                print(msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"])
                break
    
    else:
        print("No conversations processed successfully")

if __name__ == "__main__":
    main() 