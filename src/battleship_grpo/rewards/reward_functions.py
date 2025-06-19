import re
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# Helper utilities used by follow-up and contradiction rewards
# ---------------------------------------------------------------------------

import string

_guess_re = re.compile(r'<guess>\[([a-j](?:10|[1-9]))\]</guess>', re.IGNORECASE)
_strict_guess_re = re.compile(r'<guess>\[[a-j](?:10|[1-9])\]</guess>', re.IGNORECASE)


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
    """Parse <grid> ASCII board into coord->char dict. Supports either legacy
    <board> coord=value list or the new ASCII <grid>."""
    # New ASCII grid
    gmatch = re.search(r'<grid>(.*?)</grid>', content, re.S | re.I)
    if gmatch:
        section = gmatch.group(1).strip('\n')
        lines = [l.rstrip() for l in section.split('\n') if l.strip()]
        if not lines:
            return {}
        header = lines[0].strip()
        cols = [c for c in header if c.isalpha()]
        board_dict: Dict[str, str] = {}
        for row_line in lines[1:]:
            parts = row_line.split()
            if len(parts) < 11:
                continue  # malformed
            row_label = parts[0]
            try:
                row_idx = int(row_label)
            except ValueError:
                continue
            cells = parts[1:11]
            for col_letter, char in zip(cols, cells):
                board_dict[f"{col_letter}{row_idx}"] = char
        return board_dict

    # Legacy coord=value list
    board_match = re.search(r'<board>(.*?)</board>', content, re.S | re.I)
    if board_match:
        board_section = board_match.group(1)
        cell_map: Dict[str, str] = {}
        for token in board_section.replace("\n", " ").split():
            if "=" not in token:
                continue
            coord, val = token.split("=", 1)
            cell_map[coord.lower()] = val
        return cell_map
    return {}

# ---------------------------------------------------------------------------
# Follow-up reward (+0.25 when shooting next to an unresolved hit)
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
            reward += 0.25  # Increased from 0.15 for stronger strategic incentive
    
    return reward  # External clipping will handle the cap

_result_tag_re = re.compile(r'<result[^>]*value="([a-z]+)"', re.I)
_remaining_tag_re = re.compile(r'<remaining[^>]*?carrier="(\d)"[^>]*?battleship="(\d)"[^>]*?cruiser="(\d)"[^>]*?submarine="(\d)"[^>]*?destroyer="(\d)"', re.I)

def _count_remaining_from_tag(tag_match) -> int:
    counts = tag_match.groups()
    return sum(int(c) for c in counts)


def win_reward_func(completion, answer, **kwargs) -> float:
    """Sparse terminal reward split into two parts to reduce sparsity.
    +0.2 the first time the remaining ship count drops to 1.
    +1.2 the first time it drops to 0 (victory).
    
    With weight=3.0, this gives effective rewards of:
    - 0.6 for reaching the last ship
    - 3.6 for winning the game
    """
    reward = 0.0
    prev_remaining = None

    for msg in completion:
        if msg.get('role') != 'user':
            continue
        m = _remaining_tag_re.search(msg.get('content', ''))
        if not m:
            continue
        remaining_now = _count_remaining_from_tag(m)

        if prev_remaining is not None:
            # Transition from >1 to 1 => 4th ship sunk
            if prev_remaining > 1 and remaining_now == 1:
                reward += 0.2
            # Transition from >0 to 0 => final ship sunk
            if prev_remaining > 0 and remaining_now == 0:
                reward += 1.2  # Increased from 0.8 to ensure victory dominates

        prev_remaining = remaining_now

    return reward


def efficiency_reward_func(completion, answer, **kwargs) -> float:
    """Reward for efficiency - applies to ALL games, not just wins"""
    num_moves = len([x for x in completion if x['role'] == 'assistant'])
    if num_moves > 0:
        # Exponential decay: 2^(-(moves-17)/10) 
        # 17 moves = 1.0, 25 moves = 0.57, 35 moves = 0.30
        # Floor of 0.1 to prevent excessive penalty for necessary long games
        return max(0.1, 2**(-max(0, num_moves-17)/10))
    return 0.0


def hit_reward_func(completion, answer, **kwargs) -> float:
    """Diminishing−returns reward per sequential hit within a game.
    First hit   → +0.10
    Second hit  → +0.08
    Third+ hit  → +0.05 each
    We iterate chronologically through user feedback so the order is preserved.
    """
    reward = 0.0
    hits_so_far = 0

    for msg in completion:
        if msg.get('role') != 'user':
            continue

        m = _result_tag_re.search(msg.get('content', ''))
        if not (m and m.group(1).lower() in ('hit', 'sunk')):
            continue

        hits_so_far += 1
        if hits_so_far == 1:
            reward += 0.10
        elif hits_so_far == 2:
            reward += 0.08
        else:
            reward += 0.05

    return reward


def sink_reward_func(completion, answer, **kwargs) -> float:
    """Reward for sinking ships"""
    sink_count = 0
    for msg in completion:
        if msg.get('role') == 'user':
            m = _result_tag_re.search(msg.get('content', ''))
            if m and m.group(1).lower() == 'sunk':
                sink_count += 1
    return sink_count * 0.3


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
            # More flexible detection for invalid moves/formats
            if 'invalid' in content and ('move' in content or 'format' in content):
                invalid_count += 1
        elif msg.get('role') == 'assistant':
            # Count assistant moves that contain guess format using precompiled pattern
            if _strict_guess_re.search(msg.get('content', '')):
                total_moves += 1
    
    if total_moves == 0:
        return 0.0
    
    # Return fraction of valid moves (1.0 = all valid, 0.0 = all invalid)
    return max(0.0, (total_moves - invalid_count) / total_moves)


def explicit_invalid_move_penalty(completion: List[Dict[str, Any]], answer: str, **kwargs) -> float:
    """Explicit negative penalty for invalid moves to strongly deter them early in training"""
    invalid_count = 0
    for msg in completion:
        if msg.get('role') == 'user':
            content = msg.get('content', '').lower()
            # More flexible detection for invalid moves/formats
            if 'invalid' in content and ('move' in content or 'format' in content):
                invalid_count += 1
    return -0.25 * invalid_count  # Explicit penalty per invalid move


def partial_progress_reward(completion: List[Dict[str, Any]], answer: str, **kwargs) -> float:
    """Reward for making significant partial progress (sinking majority of ships)"""
    sink_count = 0
    for msg in completion:
        if msg.get('role') == 'user':
            m = _result_tag_re.search(msg.get('content', ''))
            if m and m.group(1).lower() == 'sunk':
                sink_count += 1
    
    # Reward for sinking 3 or more ships (majority of 5 total ships)
    if sink_count >= 3:
        return 0.5  # Significant reward for sinking majority of ships
    return 0.0


def _clip_reward(reward: float, min_val: float = -2.0, max_val: float = 2.0) -> float:
    """Clip reward values to prevent extreme outliers that could destabilize training"""
    return max(min_val, min(max_val, reward))


def setup_reward_rubric(rubric):
    """Improved rubric setup for more stable and efficient training
    
    Key improvements:
    - Stronger emphasis on winning (weight 3.0) with increased victory bonus
    - Better format adherence (weight 1.0) 
    - Explicit penalties for invalid moves
    - Strategic follow-up rewards (weight 1.0) with cap to prevent domination
    - Partial progress rewards to reduce sparsity
    - Reward clipping to prevent training instability
    
    Expected magnitudes for a perfect 18-move win:
    - Win: 1.4×3.0 = 4.2
    - Efficiency: ~0.9×0.5 = 0.45
    - Hits: ~0.33×0.5 = 0.17
    - Sinks: 1.5×1.0 = 1.5
    - Format: 1.0×1.0 = 1.0
    - Valid: 1.0×0.5 = 0.5
    - Follow-up: ~2.0×1.0 = 2.0 (clipped)
    - Total: ~9.8 (victory clearly dominates)
    """
    
    # Create clipped versions of reward functions
    def clipped_win_reward(*args, **kwargs):
        return _clip_reward(win_reward_func(*args, **kwargs))
    
    def clipped_follow_up_reward(*args, **kwargs):
        return _clip_reward(follow_up_reward_func(*args, **kwargs))
    
    def clipped_explicit_penalty(*args, **kwargs):
        return _clip_reward(explicit_invalid_move_penalty(*args, **kwargs))
    
    # Add reward functions with adjusted weights
    rubric.add_reward_func(clipped_win_reward, weight=3.0)                 # Strongly reward final victories
    rubric.add_reward_func(efficiency_reward_func, weight=0.5)             # Moderate incentive for speed
    rubric.add_reward_func(hit_reward_func, weight=0.5)                    # Small incremental rewards
    rubric.add_reward_func(sink_reward_func, weight=1.0)                   # Clear ship sinking reward
    rubric.add_reward_func(format_reward_func, weight=1.0)                 # Stronger emphasis on formatting
    rubric.add_reward_func(valid_move_reward_func, weight=0.5)             # Reduced to avoid overshadowing penalty
    rubric.add_reward_func(clipped_explicit_penalty, weight=1.0)           # Clear penalty for invalid moves
    rubric.add_reward_func(clipped_follow_up_reward, weight=1.0)           # Strategic follow-up (reduced weight)
    rubric.add_reward_func(partial_progress_reward, weight=1.0)            # Reduce sparsity by rewarding partial goals
    
    return rubric
