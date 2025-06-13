import re
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# Helper utilities used by follow-up and contradiction rewards
# ---------------------------------------------------------------------------

import string

_guess_re = re.compile(r'<guess>\[([a-j](?:10|[1-9]))\]</guess>', re.IGNORECASE)


def _coord_to_tuple(coord: str):
    """Convert 'b7' -> (1, 6). Returns None if malformed."""
    if not coord or len(coord) < 2:
        return None
    col = coord[0].lower()
    if col not in string.ascii_lowercase[:10]:
        return None
    try:
        row = int(coord[1:])
    except ValueError:
        return None
    if not (1 <= row <= 10):
        return None
    return (string.ascii_lowercase.index(col), row - 1)


def _is_adjacent(c1: str, c2: str) -> bool:
    t1, t2 = _coord_to_tuple(c1), _coord_to_tuple(c2)
    if t1 is None or t2 is None:
        return False
    dx, dy = abs(t1[0] - t2[0]), abs(t1[1] - t2[1])
    return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)


def _extract_board(content: str) -> Dict[str, str]:
    """Parse <board> compact grid to dict{coord:char}. Empty dict if absent."""
    m = re.search(r'<board>(.*?)</board>', content, re.I | re.S)
    if not m:
        return {}
    section = m.group(1)
    cell_map: Dict[str, str] = {}
    for tok in section.replace("\n", " ").split():
        if "=" not in tok:
            continue
        coord, val = tok.split("=", 1)
        cell_map[coord.lower()] = val.lower()
    return cell_map

# ---------------------------------------------------------------------------
# Follow-up reward (+0.15 when shooting next to an unresolved hit)
# ---------------------------------------------------------------------------


def follow_up_reward_func(completion: List[Dict[str, Any]], answer: str, **kwargs) -> float:
    reward = 0.0
    # walk through conversation: user feedback then assistant guess
    for i in range(len(completion) - 1):
        user_msg = completion[i]
        asst_msg = completion[i + 1]
        if user_msg.get('role') != 'user' or asst_msg.get('role') != 'assistant':
            continue

        board = _extract_board(user_msg.get('content', ''))
        hit_cells = [c for c, v in board.items() if v == 'x']
        if not hit_cells:
            continue

        gmatch = _guess_re.search(asst_msg.get('content', ''))
        if not gmatch:
            continue
        guess_coord = gmatch.group(1).lower()
        if any(_is_adjacent(guess_coord, hc) for hc in hit_cells):
            reward += 0.15
    return reward

# ---------------------------------------------------------------------------
# Contradiction penalty (-0.2 when claiming hit but feedback = miss/invalid)
# ---------------------------------------------------------------------------


_miss_re = re.compile(r'\bmiss\b|\binvalid move\b', re.I)
_hit_claim_re = re.compile(r'\bhit\b|\bsunk\b|\bdestroyed\b|\bvictory\b', re.I)
_think_re = re.compile(r'<think>(.*?)</think>', re.I | re.S)


def contradiction_penalty_func(completion: List[Dict[str, Any]], answer: str, **kwargs) -> float:
    penalty = 0.0
    for i in range(len(completion) - 1):
        asst_msg = completion[i]
        user_msg = completion[i + 1]
        if asst_msg.get('role') != 'assistant' or user_msg.get('role') != 'user':
            continue

        feedback = user_msg.get('content', '')
        if not _miss_re.search(feedback):
            continue

        think_match = _think_re.search(asst_msg.get('content', ''))
        think_text = think_match.group(1) if think_match else asst_msg.get('content', '')
        if _hit_claim_re.search(think_text):
            penalty -= 0.2
    return penalty


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
    think_pattern = re.compile(r'<think>.*?</think>', re.IGNORECASE | re.S)
    guess_pattern = re.compile(r'<guess>\[[a-j](?:10|[1-9])\]</guess>', re.IGNORECASE)

    for msg in assistant_messages:
        content = msg.get('content', '')
        # Message is valid only if **both** a think block and a properly formatted guess exist
        if think_pattern.search(content) and guess_pattern.search(content):
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
    
    # New signals
    rubric.add_reward_func(follow_up_reward_func, weight=1.0)   # +0.15 per good follow-up (weight 1)
    rubric.add_reward_func(contradiction_penalty_func, weight=1.0)  # −0.2 per contradiction
    
    return rubric
