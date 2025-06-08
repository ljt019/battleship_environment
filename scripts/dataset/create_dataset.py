import random
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.battleship_game import BattleshipGame
from datasets import Dataset

def parse_coordinate(coord):
    """Convert coordinate like 'b6' to (row_idx, col_idx) - both 0-indexed"""
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
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E
    
    for d_row, d_col in directions:
        new_row = row_num + d_row
        new_col = col_idx + d_col
        
        if 1 <= new_row <= board_size and 0 <= new_col < board_size:
            adjacent.append(f"{cols[new_col]}{new_row}")
    
    return adjacent

def get_remaining_ships(sunk_ships_info):
    """Calculate which ship sizes are still active based on sunk ship tracking"""
    original_ship_sizes = [5, 4, 3, 3, 2]  # Carrier, Battleship, Cruiser, Sub, Destroyer
    remaining_ships = original_ship_sizes.copy()
    
    for sunk_ship_size in sunk_ships_info:
        if sunk_ship_size in remaining_ships:
            remaining_ships.remove(sunk_ship_size)
    
    return remaining_ships

def can_ship_fit_here(game, start_row, start_col, ship_size, orientation):
    """Check if a ship can be legally placed at this position"""
    board_size = 10
    
    # Calculate ship squares
    if orientation == 'horizontal':
        if start_col + ship_size > board_size:
            return False
        ship_squares = [(start_row, start_col + i) for i in range(ship_size)]
    else:  # vertical
        if start_row + ship_size > board_size:
            return False
        ship_squares = [(start_row + i, start_col) for i in range(ship_size)]
    
    # Check each square - can't overlap with misses ('o') or sunk ships ('s')
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
    
    # Initialize probability grid
    probability_map = {}
    for row in range(board_size):
        for col in range(board_size):
            coord = make_coordinate(row, col)
            probability_map[coord] = 0
    
    # For each remaining ship, calculate all valid placements
    for ship_size in remaining_ships:
        for start_row in range(board_size):
            for start_col in range(board_size):
                
                # Try horizontal placement
                if can_ship_fit_here(game, start_row, start_col, ship_size, 'horizontal'):
                    for i in range(ship_size):
                        coord = make_coordinate(start_row, start_col + i)
                        probability_map[coord] += 1
                
                # Try vertical placement
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
    
    # Priority 1: Target adjacent to active hits (most important rule)
    active_hits = [coord for coord, state in game.board.items() if state == 'x']
    if active_hits:
        adjacent_moves = []
        for hit_coord in active_hits:
            for adjacent_coord in get_adjacent_squares(hit_coord):
                if adjacent_coord in valid_moves:
                    adjacent_moves.append(adjacent_coord)
        
        if adjacent_moves:
            adjacent_moves = list(set(adjacent_moves))  # Remove duplicates
            return random.choice(adjacent_moves)
    
    # Priority 2: Use probability density for hunt mode
    probability_map = calculate_ship_placement_probabilities(game, sunk_ships_info)
    
    # Find moves with highest probability
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
        self.sunk_ships = []  # Track sunk ship sizes
        self.original_ship_sizes = [5, 4, 3, 3, 2]
        
    def play_smart_moves(self, max_moves):
        """Play moves using optimal strategy with some randomness for diversity"""
        num_moves = random.randint(0, max_moves)
        
        for _ in range(num_moves):
            if self.game.game_over or not self.game.get_valid_moves():
                break
                
            # 80% optimal, 20% suboptimal for training diversity
            if random.random() < 0.80:
                smart_move = find_optimal_move(self.game, self.sunk_ships)
                if smart_move:
                    move = smart_move
                else:
                    move = random.choice(self.game.get_valid_moves())
            else:
                move = random.choice(self.game.get_valid_moves())
            
            observation, hit, sunk, game_over, invalid = self.game.step(move)
            
            if sunk:
                self._track_sunk_ship()
    
    def _track_sunk_ship(self):
        """Track when a ship is sunk by inferring size from board state"""
        # Count current sunk squares
        current_sunk_squares = sum(1 for state in self.game.board.values() if state == 's')
        
        # Calculate newly sunk squares
        expected_sunk_squares = sum(self.sunk_ships)
        newly_sunk_squares = current_sunk_squares - expected_sunk_squares
        
        if newly_sunk_squares > 0:
            # Find remaining ship that matches newly sunk squares
            remaining_ships = [size for size in self.original_ship_sizes if size not in self.sunk_ships]
            
            if newly_sunk_squares in remaining_ships:
                self.sunk_ships.append(newly_sunk_squares)
            else:
                # Fallback: assume smallest remaining ship
                if remaining_ships:
                    self.sunk_ships.append(min(remaining_ships))

def create_training_example_with_tracking(simulator):
    """Create a training example in verifiers chat format"""
    board_state = simulator.game.render()
    optimal_move = find_optimal_move(simulator.game, simulator.sunk_ships)
    
    if not optimal_move:
        return None
    
    question = f"Given this battleship board state, what is the best next move?\n\n{board_state}"
    
    return {
        "prompt": [{"role": "user", "content": question}],
        "completion": [{"role": "assistant", "content": f"[{optimal_move}]"}]
    }

def generate_training_dataset(num_examples=3000):
    """Generate training dataset with mixed game stages"""
    print(f"Generating {num_examples} battleship training examples...")
    examples = []
    
    early_game_count = 0
    mid_game_count = 0
    late_game_count = 0
    
    for i in range(num_examples):
        simulator = GameSimulator()
        
        # Mix game stages for training diversity (33% each)
        if i % 3 == 0:
            max_moves = random.randint(0, 15)  # Early game
            stage = "early"
            early_game_count += 1
        elif i % 3 == 1:
            max_moves = random.randint(16, 35)  # Mid game
            stage = "mid"
            mid_game_count += 1
        else:
            max_moves = random.randint(36, 70)  # Late game
            stage = "late"
            late_game_count += 1
        
        simulator.play_smart_moves(max_moves)
        
        if simulator.game.game_over:
            continue
        
        # Force minimum move count for late games
        if stage == "late":
            actual_moves = len(simulator.game.history)
            if actual_moves < 36:
                additional_moves = 36 - actual_moves
                simulator.play_smart_moves(additional_moves)
        
        example = create_training_example_with_tracking(simulator)
        if example:
            examples.append(example)
        
        if i % 1000 == 0:
            print(f"Generated {len(examples)} examples... ({i}/{num_examples} attempts)")
    
    print(f"Created {len(examples)} training examples")
    print(f"Distribution: Early={early_game_count}, Mid={mid_game_count}, Late={late_game_count}")
    return examples

def main():
    """Generate and save the battleship dataset"""
    training_examples = generate_training_dataset(num_examples=3000)
    
    if not training_examples:
        print("ERROR: No training examples generated!")
        return
    
    dataset = Dataset.from_list(training_examples)
    
    # Show sample
    print("\nSample training example:")
    print("-" * 40)
    sample = dataset[0]
    print("Prompt:", sample['prompt'])
    print("Completion:", sample['completion'])
    
    # Save dataset
    output_path = "datasets/battleship_rlvr_qwen3_dataset"
    dataset.save_to_disk(output_path)
    print(f"\nDataset saved to: {output_path}")
    print(f"Total examples: {len(dataset)}")

if __name__ == "__main__":
    main()
