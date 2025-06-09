import random
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.battleship_game import BattleshipGame
from datasets import Dataset

def parse_coordinate(coord):
    """Convert coordinate like 'b6' to (row_idx, col_idx)"""
    cols = 'abcdefghij'
    col_letter = coord[0].lower()
    row_num = int(coord[1:])
    return row_num - 1, cols.index(col_letter)

def make_coordinate(row_idx, col_idx):
    """Convert (row_idx, col_idx) to coordinate like 'b6'"""
    cols = 'abcdefghij'
    return f"{cols[col_idx]}{row_idx + 1}"

def get_adjacent_squares(coord, board_size=10):
    """Get all valid adjacent coordinates (North, South, East, West)"""
    cols = 'abcdefghij'
    col_letter = coord[0].lower()
    row_num = int(coord[1:])
    col_idx = cols.index(col_letter)
    
    adjacent = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for d_row, d_col in directions:
        new_row = row_num + d_row
        new_col = col_idx + d_col
        
        if 1 <= new_row <= board_size and 0 <= new_col < board_size:
            adjacent.append(f"{cols[new_col]}{new_row}")
    
    return adjacent

def get_remaining_ships(sunk_ships_info):
    """Calculate which ship sizes are still active based on sunk ship tracking"""
    original_ship_sizes = [5, 4, 3, 3, 2]
    remaining_ships = original_ship_sizes.copy()
    
    for sunk_ship_size in sunk_ships_info:
        if sunk_ship_size in remaining_ships:
            remaining_ships.remove(sunk_ship_size)
    
    return remaining_ships

def can_ship_fit_here(game, start_row, start_col, ship_size, orientation):
    """Check if a ship can be legally placed at this position"""
    board_size = 10
    
    if orientation == 'horizontal':
        if start_col + ship_size > board_size:
            return False
        ship_squares = [(start_row, start_col + i) for i in range(ship_size)]
    else:
        if start_row + ship_size > board_size:
            return False
        ship_squares = [(start_row + i, start_col) for i in range(ship_size)]
    
    # Check each square - can't overlap with misses or sunk ships
    for row, col in ship_squares:
        coord = make_coordinate(row, col)
        state = game.board[coord]
        if state in ['o', 's']:
            return False
    
    return True

def calculate_ship_placement_probabilities(game, sunk_ships_info):
    """Calculate probability density for each square based on possible ship placements"""
    board_size = 10
    remaining_ships = get_remaining_ships(sunk_ships_info)
    
    if not remaining_ships:
        return {}
    
    probability_map = {}
    for row in range(board_size):
        for col in range(board_size):
            coord = make_coordinate(row, col)
            probability_map[coord] = 0
    
    # For each remaining ship, calculate all valid placements
    for ship_size in remaining_ships:
        for start_row in range(board_size):
            for start_col in range(board_size):
                
                if can_ship_fit_here(game, start_row, start_col, ship_size, 'horizontal'):
                    for i in range(ship_size):
                        coord = make_coordinate(start_row, start_col + i)
                        probability_map[coord] += 1
                
                if can_ship_fit_here(game, start_row, start_col, ship_size, 'vertical'):
                    for i in range(ship_size):
                        coord = make_coordinate(start_row + i, start_col)
                        probability_map[coord] += 1
    
    return probability_map

def find_optimal_move(game, sunk_ships_info):
    """Find the best next move using optimal battleship strategy"""
    valid_moves = game.get_valid_moves()
    if not valid_moves:
        return None
    
    # Priority 1: Target adjacent to active hits
    active_hits = [coord for coord, state in game.board.items() if state == 'x']
    if active_hits:
        adjacent_moves = []
        for hit_coord in active_hits:
            for adjacent_coord in get_adjacent_squares(hit_coord):
                if adjacent_coord in valid_moves:
                    adjacent_moves.append(adjacent_coord)
        
        if adjacent_moves:
            adjacent_moves = list(set(adjacent_moves))
            return random.choice(adjacent_moves)
    
    # Priority 2: Use probability density for hunt mode
    probability_map = calculate_ship_placement_probabilities(game, sunk_ships_info)
    
    best_probability = -1
    best_moves = []
    
    for move in valid_moves:
        probability = probability_map.get(move, 0)
        if probability > best_probability:
            best_probability = probability
            best_moves = [move]
        elif probability == best_probability:
            best_moves.append(move)
    
    if not best_moves:
        return random.choice(valid_moves)
    
    return random.choice(best_moves)

class GameSimulator:
    """Wrapper to simulate battleship games and track sunk ships"""
    
    def __init__(self):
        self.game = BattleshipGame()
        self.sunk_ships = []
        self.original_ship_sizes = [5, 4, 3, 3, 2]
    
    def _track_sunk_ship(self):
        """Track when a ship is sunk by inferring size from board state"""
        current_sunk_squares = sum(1 for state in self.game.board.values() if state == 's')
        expected_sunk_squares = sum(self.sunk_ships)
        newly_sunk_squares = current_sunk_squares - expected_sunk_squares
        
        if newly_sunk_squares > 0:
            remaining_ships = [size for size in self.original_ship_sizes if size not in self.sunk_ships]
            
            if newly_sunk_squares in remaining_ships:
                self.sunk_ships.append(newly_sunk_squares)
            else:
                if remaining_ships:
                    self.sunk_ships.append(min(remaining_ships))

def play_full_game(max_turns=50):
    """Play a complete battleship game and return the conversation"""
    simulator = GameSimulator()
    game = simulator.game
    
    system_prompt = {
        "role": "system",
        "content": "You are a competitive battleship player. Make sure you read the game instructions carefully, and always follow the required format.\n\nIn each turn, think step-by-step inside <think>...</think> tags, then make your move inside <guess>...</guess> tags."
    }
    
    user_prompt = {
        "role": "user", 
        "content": "You are playing Battleship against an opponent.\nYour goal is to sink all enemy ships by guessing their locations.\nThe board shows:\n  [?] = unknown squares\n  [x] = hit (part of a ship)\n  [o] = miss (water)\n  [s] = sunk ship part\n\nFor each move, wrap your coordinate in square brackets (e.g., [d6]).\nMake strategic moves to find and sink all ships efficiently.\nEnter your first guess to begin."
    }
    
    initial_board_message = {
        "role": "user",
        "content": f"Here's your starting board:\n\n{game.render()}\n\nMake your first move:"
    }
    
    conversation = [initial_board_message]
    turn_count = 0
    
    while not game.game_over and turn_count < max_turns:
        optimal_move = find_optimal_move(game, simulator.sunk_ships)
        if not optimal_move:
            break
            
        # Assistant move (reasoning will be added later by Claude)
        assistant_move = {
            "role": "assistant",
            "content": f"<guess>[{optimal_move}]</guess>"
        }
        conversation.append(assistant_move)
        
        observation, hit, sunk, game_over, invalid = game.step(optimal_move)
        
        if sunk:
            simulator._track_sunk_ship()
        
        if invalid:
            feedback = f"Invalid move! Try again."
        elif hit:
            if sunk:
                feedback = f"Hit and sunk!"
            else:
                feedback = f"Hit!"
        else:
            feedback = f"Miss."
        
        if game_over:
            env_response = {
                "role": "user",
                "content": f"{feedback}\n\n{game.render()}\n\n[GAME] You won! All ships sunk in {len(game.history)} moves."
            }
        else:
            env_response = {
                "role": "user",
                "content": f"{feedback}\n\n{game.render()}\n\nNext move:"
            }
        
        conversation.append(env_response)
        turn_count += 1
    
    final_reward = 2.0 if game_over else 1.0
    
    return {
        "prompt": [system_prompt, user_prompt],
        "completion": conversation,
        "answer": "victory" if game_over else "incomplete",
        "reward": final_reward
    }

def generate_multiturn_games_dataset(num_games=1000):
    """Generate multi-turn battleship games dataset"""
    print(f"Generating {num_games} complete battleship games...")
    examples = []
    
    successful_games = 0
    
    for i in range(num_games):
        try:
            game_example = play_full_game(max_turns=50)
            
            if len(game_example['completion']) >= 4:
                examples.append(game_example)
                if game_example['answer'] == 'victory':
                    successful_games += 1
        
        except Exception as e:
            print(f"Error in game {i}: {e}")
            continue
        
        if i % 100 == 0:
            print(f"Generated {len(examples)} games... ({i}/{num_games} attempts)")
    
    print(f"Created {len(examples)} game examples")
    print(f"Successful completions: {successful_games}/{len(examples)}")
    return examples

def main():
    """Generate and save the battleship multi-turn games dataset"""
    game_examples = generate_multiturn_games_dataset(num_games=1000)
    
    if not game_examples:
        print("ERROR: No game examples generated!")
        return
    
    dataset = Dataset.from_list(game_examples)
    
    print("\nSample game example:")
    print("-" * 40)
    sample = dataset[0]
    print("System prompt:", sample['prompt'][0]['content'][:100] + "...")
    print("User prompt:", sample['prompt'][1]['content'][:100] + "...")
    print("Conversation length:", len(sample['completion']))
    print("First move:", sample['completion'][1]['content'])
    print("First response:", sample['completion'][2]['content'][:100] + "...")
    print("Answer:", sample['answer'])
    print("Reward:", sample['reward'])
    
    output_path = "datasets/battleship_board_states"
    dataset.save_to_disk(output_path)
    print(f"\nMulti-turn games dataset saved to: {output_path}")
    print(f"Total examples: {len(dataset)}")
    print("\nNext step: Upload to hub, then run generate_reasoning_with_claude.py to add <think> tags")

if __name__ == "__main__":
    main() 