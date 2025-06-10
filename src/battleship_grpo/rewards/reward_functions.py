import re
from typing import Dict, List, Any


def win_reward_func(completion, answer, **kwargs) -> float:
    """Reward for winning the game"""
    for msg in completion:
        if msg.get('role') == 'user':
            content = msg.get('content', '').lower()
            if 'victory!' in content or 'you won!' in content or 'all ships sunk' in content:
                return 1.0
    return 0.0


def efficiency_reward_func(completion, answer, **kwargs) -> float:
    """Reward for efficiency - applies to ALL games, not just wins"""
    num_moves = len([x for x in completion if x['role'] == 'assistant'])
    if num_moves > 0:
        # Exponential decay: 2^(-(moves-17)/10) 
        # 17 moves = 1.0, 25 moves = 0.57, 35 moves = 0.30
        return 2**(-max(0, num_moves-17)/10)
    return 0.0


def hit_reward_func(completion, answer, **kwargs) -> float:
    """Reward for hitting ships"""
    hit_count = 0
    for msg in completion:
        if msg.get('role') == 'user':
            content = msg.get('content', '').lower()
            # Count both regular hits and sunk hits
            if ('hit!' in content or 'hit and sunk!' in content) and 'miss' not in content:
                hit_count += 1
    return hit_count * 0.1  # 0.1 reward per hit


def sink_reward_func(completion, answer, **kwargs) -> float:
    """Reward for sinking ships"""
    sink_count = 0
    for msg in completion:
        if msg.get('role') == 'user':
            content = msg.get('content', '').lower()
            if 'hit and sunk!' in content or 'destroyed an entire ship' in content:
                sink_count += 1
    return sink_count * 0.3  # 0.3 reward per ship sunk


def format_reward_func(completion, answer, **kwargs) -> float:
    """Reward for proper move format"""
    assistant_messages = [x for x in completion if x['role'] == 'assistant']
    if not assistant_messages:
        return 0.0
    
    valid_format_count = 0
    for msg in assistant_messages:
        content = msg.get('content', '')
        # Check for proper <guess>[coordinate]</guess> format
        if re.search(r'<guess>\[[a-j][0-9]+\]</guess>', content, re.IGNORECASE):
            valid_format_count += 1
    
    return valid_format_count / len(assistant_messages)


def valid_move_reward_func(completion, answer, **kwargs) -> float:
    """Penalty for invalid moves (already played, out of bounds, etc.)"""
    invalid_count = 0
    total_moves = 0
    
    for msg in completion:
        if msg.get('role') == 'user':
            content = msg.get('content', '').lower()
            if 'invalid move' in content or 'invalid format' in content:
                invalid_count += 1
        elif msg.get('role') == 'assistant':
            # Count assistant moves that contain guess format
            if re.search(r'<guess>\[[a-j][0-9]+\]</guess>', msg.get('content', ''), re.IGNORECASE):
                total_moves += 1
    
    if total_moves == 0:
        return 0.0
    
    # Return fraction of valid moves (1.0 = all valid, 0.0 = all invalid)
    return max(0.0, (total_moves - invalid_count) / total_moves)


def setup_reward_rubric(rubric):
    """Add all reward functions to the rubric with their weights"""
    rubric.add_reward_func(win_reward_func, weight=2.0)        # Main objective
    rubric.add_reward_func(efficiency_reward_func, weight=0.5) # Encourage speed (reduced since always active)
    rubric.add_reward_func(hit_reward_func, weight=0.5)        # Reward progress  
    rubric.add_reward_func(sink_reward_func, weight=1.0)       # Reward major progress
    rubric.add_reward_func(format_reward_func, weight=0.5)     # Ensure proper format
    rubric.add_reward_func(valid_move_reward_func, weight=1.0) # Penalize invalid moves
    
    return rubric
