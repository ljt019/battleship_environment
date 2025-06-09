import json
from datasets import Dataset, load_dataset
from pathlib import Path
import anthropic
import os
from typing import List, Dict
import time
import re

def classify_game_stage(conversation: List[Dict]) -> str:
    """Classify game stage based on board state density"""
    # Find the last board state in conversation
    last_board_content = ""
    for msg in reversed(conversation):
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            bracket_count = len(re.findall(r'\[[?xos]\]', content))
            if bracket_count >= 10:  # This is a board state
                last_board_content = content
                break
    
    if not last_board_content:
        return "unknown"
    
    # Count different board states
    unknown_count = len(re.findall(r'\[\?\]', last_board_content))
    hit_count = len(re.findall(r'\[x\]', last_board_content))
    miss_count = len(re.findall(r'\[o\]', last_board_content))
    sunk_count = len(re.findall(r'\[s\]', last_board_content))
    
    total_moves = hit_count + miss_count + sunk_count
    
    # Classify based on move count and game state
    if total_moves <= 8:
        return "early"
    elif total_moves <= 20:
        return "mid" 
    else:
        return "late"

def create_stratified_examples(conversation: List[Dict], target_stages: List[str]) -> List[Dict]:
    """Create examples for specific game stages from a conversation"""
    examples = []
    stage_found = {stage: False for stage in target_stages}
    
    for i, msg in enumerate(conversation):
        if msg.get('role') == 'assistant':
            # Get conversation up to this point
            partial_conversation = conversation[:i+2]  # Include assistant move + env response
            
            if len(partial_conversation) < 3:  # Skip very early examples
                continue
                
            stage = classify_game_stage(partial_conversation)
            
            # Only take ONE example per stage per conversation
            if stage in target_stages and not stage_found[stage]:
                examples.append({
                    "conversation": partial_conversation,
                    "stage": stage,
                    "move_number": len([x for x in partial_conversation if x['role'] == 'assistant'])
                })
                stage_found[stage] = True
                
                # If we've found all needed stages, stop processing this conversation
                if all(stage_found[s] for s in target_stages):
                    break
    
    return examples

def _compact_conversation_history(messages: List[Dict]) -> List[Dict]:
    """Remove old board states from conversation, keeping only the most recent one"""
    compacted_messages = []
    last_board_message_idx = -1
    
    # Find the last message that contains a board (has multiple brackets indicating a board)
    for i, msg in enumerate(messages):
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            # Check if this looks like a board state (multiple [?], [x], [o], [s] patterns)
            bracket_count = len(re.findall(r'\[[?xos]\]', content))
            if bracket_count >= 10:  # Likely a board state
                last_board_message_idx = i
    
    # Keep all messages, but replace old board states with just the feedback
    for i, msg in enumerate(messages):
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            bracket_count = len(re.findall(r'\[[?xos]\]', content))
            
            # If this is an old board state (not the most recent), extract just the feedback
            if bracket_count >= 10 and i < last_board_message_idx:
                # Extract just the move result (HIT, MISS, etc.) from old board messages
                lines = content.split('\n')
                feedback_line = lines[0] if lines else content
                # Keep just the first line (move result) + "Next move:"
                compacted_content = f"{feedback_line}\nNext move:"
                compacted_messages.append({"role": "user", "content": compacted_content})
            else:
                # Keep full message (most recent board or non-board messages)
                compacted_messages.append(msg)
        else:
            # Keep all assistant messages unchanged
            compacted_messages.append(msg)
    
    return compacted_messages

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
            # time.sleep(0.3)  # Removed to speed up processing
            
        else:
            enhanced_conversation.append(message)
    
    return enhanced_conversation

def process_dataset_entries(dataset_entries: List[Dict], client: anthropic.Anthropic, max_entries: int = None) -> List[Dict]:
    """Process dataset entries with stratified sampling for early/mid/late game examples"""
    
    enhanced_entries = []
    entries_to_process = dataset_entries[:max_entries] if max_entries else dataset_entries
    
    # Target distribution: 40% early, 35% mid, 25% late
    # Ensure at least 1 of each stage for small samples
    target_counts = {
        "early": max(1, int(len(entries_to_process) * 0.4)),
        "mid": max(1, int(len(entries_to_process) * 0.35)), 
        "late": max(1, int(len(entries_to_process) * 0.25))
    }
    
    stage_counts = {"early": 0, "mid": 0, "late": 0}
    
    for i, entry in enumerate(entries_to_process):
        print(f"Processing conversation {i+1}/{len(entries_to_process)}...")
        
        prompt = entry.get('prompt', [])
        completion = entry.get('completion', [])
        
        if not completion:
            print(f"Skipping entry {i+1} - no completion found")
            continue
        
        # Create stratified examples from this conversation
        needed_stages = [stage for stage, count in stage_counts.items() 
                        if count < target_counts[stage]]
        
        if not needed_stages:
            print(f"  Target distribution reached, skipping remaining entries")
            break
            
        stage_examples = create_stratified_examples(completion, needed_stages)
        
        for example in stage_examples:
            stage = example["stage"]
            if stage_counts[stage] >= target_counts[stage]:
                continue
                
            conversation = example["conversation"]
            
            try:
                # Generate reasoning with full context
                enhanced_conversation = enhance_conversation_with_reasoning(conversation, client)
                
                # Compact the enhanced conversation
                compacted_conversation = _compact_conversation_history(enhanced_conversation)
            except Exception as e:
                print(f"❌ API Error: {e}")
                print(f"💾 Saving progress before exit: {len(enhanced_entries)} examples")
                if enhanced_entries:
                    emergency_dataset = Dataset.from_list(enhanced_entries)
                    emergency_path = f'datasets/battleship_sft_emergency_{len(enhanced_entries)}'
                    emergency_dataset.save_to_disk(emergency_path)
                    print(f"💾 Emergency save completed to {emergency_path}")
                raise e
            
            enhanced_entry = {
                "prompt": prompt,
                "completion": compacted_conversation,
                "stage": stage,
                "move_number": example["move_number"]
            }
            
            enhanced_entries.append(enhanced_entry)
            stage_counts[stage] += 1
            
            print(f"  Added {stage} game example (move {example['move_number']})")
            
            if stage_counts[stage] >= target_counts[stage]:
                print(f"  Completed {stage} game examples ({stage_counts[stage]}/{target_counts[stage]})")
        
        # Save progress every 5 conversations
        if (i + 1) % 5 == 0 and enhanced_entries:
            temp_dataset = Dataset.from_list(enhanced_entries)
            temp_path = f'datasets/battleship_sft_temp_{i+1}'
            temp_dataset.save_to_disk(temp_path)
            print(f"  💾 Saved progress: {len(enhanced_entries)} examples to {temp_path}")
    
    print(f"\nFinal distribution: {stage_counts}")
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
    
    max_entries = 50
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