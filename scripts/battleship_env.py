import random
from typing import Tuple, List, Dict, Any
import re

from datasets import Dataset

from verifiers import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from battleship_game import BattleshipGame

BATTLESHIP_SYSTEM_PROMPT = """You are a competitive battleship player. Make sure you read the game instructions carefully, and always follow the required format.

In each turn, think step-by-step inside <think>...</think> tags, then make your move inside <guess>...</guess> tags."""

BATTLESHIP_RULES = """You are playing Battleship against an opponent.
Your goal is to sink all enemy ships by guessing their locations.

IMPORTANT - Board symbols:
  [?] = unknown squares (you haven't guessed here yet)
  [x] = hit (you found part of a ship)
  [o] = miss (you guessed here but found only water)
  [s] = sunk ship part (entire ship destroyed)

REQUIRED FORMAT: For each move, you must use this exact format:
<think>
[Your strategic reasoning]
</think>

<guess>[coordinate]</guess>

Example: <guess>[d6]</guess>

Strategy tips:
- After hitting a ship ([x]), try adjacent squares to find the rest of it
- Use probability to target areas likely to contain ships
- Remember ship sizes: Carrier(5), Battleship(4), Cruiser(3), Submarine(3), Destroyer(2)

Make strategic moves to find and sink all ships efficiently."""


class BattleshipEnv(MultiTurnEnv):
    """
    Battleship environment for GRPO training.
    """
    def __init__(self,
                 num_samples: int = 1000,
                 num_eval_samples: int = 100,
                 seed: int = 0,
                 **kwargs):
        self.num_samples = num_samples
        self.num_eval_samples = num_eval_samples
        self.seed = seed
        dataset, eval_dataset = self.create_hf_datasets()
        parser = XMLParser(fields=["think", "guess"], answer_field="guess")
        rubric = Rubric(parser=parser)
        
        def check_win_reward_func(completion, answer, **kwargs) -> float:
            """Reward for winning the game"""
            # Check if game was won (all ships sunk)
            for msg in completion:
                if msg.get('role') == 'user':
                    content = msg.get('content', '').lower()
                    if 'victory!' in content or 'you won!' in content or 'all ships sunk' in content:
                        return 1.0
            return 0.0
        
        def efficiency_reward_func(completion, answer, **kwargs) -> float:
            """Reward for winning efficiently (fewer moves)"""
            num_moves = len([x for x in completion if x['role'] == 'assistant'])
            win_reward = check_win_reward_func(completion, answer, **kwargs)
            if win_reward > 0 and num_moves > 0:
                # Reward efficiency: 1/(moves + penalty) where penalty makes 17 moves = 0.5 reward
                return win_reward / (num_moves + 16)
            return 0.0
        
        def move_format_reward_func(completion, answer, **kwargs) -> float:
            """Reward for proper move format"""
            assistant_messages = [x for x in completion if x['role'] == 'assistant']
            if not assistant_messages:
                return 0.0
            
            valid_moves = 0
            for msg in assistant_messages:
                content = msg.get('content', '')
                # Check for proper <guess>[coordinate]</guess> format
                if re.search(r'<guess>\[[a-j][0-9]+\]</guess>', content, re.IGNORECASE):
                    valid_moves += 1
            
            return valid_moves / len(assistant_messages)
        
        rubric.add_reward_func(check_win_reward_func, weight=1.0)
        rubric.add_reward_func(efficiency_reward_func, weight=1.0)
        rubric.add_reward_func(move_format_reward_func, weight=0.3)

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=BATTLESHIP_SYSTEM_PROMPT,
            parser=parser,
            rubric=rubric,
            message_type='chat',
            **kwargs
        )

    def is_completed(self,
                     messages: List[Dict[str, Any]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> bool:
        """Check if the battleship game is completed"""
        if 'game' not in state:
            return False
        
        game = state['game']
        return game.game_over

    def env_response(self,
                     messages: List[Dict[str, Any]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate environment response to player move"""
                 # Initialize game if needed
        if 'game' not in state:
            game = BattleshipGame()
            game.reset()  # This places ships randomly
            state['game'] = game
            
            # Return initial board state
            board_render = game.render()
            content = f"Here's your starting board:\n\n{board_render}\n\nMake your first move:"
            return {"role": "user", "content": content}, state
        
        game = state['game']
        
        # Parse the last assistant message for the move
        last_message = messages[-1]["content"]
        
        # Extract coordinate from <guess>[coordinate]</guess>
        coord_match = re.search(r'<guess>\[([a-j][0-9]+)\]</guess>', last_message, re.IGNORECASE)
        if not coord_match:
            # Fallback: look for any coordinate in brackets
            coord_match = re.search(r'\[([a-j][0-9]+)\]', last_message, re.IGNORECASE)
        
        if not coord_match:
            return {"role": "user", "content": "INVALID FORMAT! Please use: <guess>[coordinate]</guess>\nExample: <guess>[e5]</guess>\nTry again:"}, state
        
        coordinate = coord_match.group(1).lower()
        
        # Make the move using the proper game interface
        board_render, hit, sunk, game_over, invalid = game.step(coordinate)
        
        # Update state
        state['game'] = game
        
        # Generate result message
        if invalid:
            result = "INVALID MOVE - That square was already guessed or doesn't exist!"
        elif hit:
            if sunk:
                result = "HIT AND SUNK! You destroyed an entire ship!"
            else:
                result = "HIT! You found part of a ship - try adjacent squares!"
        else:
            result = "MISS - Only water here."
        
        # Generate response
        if game_over:
            content = f"{result}\n\n{board_render}\n\nVICTORY! You sunk all ships in {game.turn_count} moves!"
        else:
            content = f"{result}\n\n{board_render}\n\nRemember: [x]=hit, [o]=miss, [s]=sunk, [?]=unknown\nNext move:"
        
        return {"role": "user", "content": content}, state
    
    def create_hf_datasets(self) -> Tuple[Dataset, Dataset]:
        """Create HuggingFace datasets for training and evaluation"""
        dataset_rows = []
        eval_dataset_rows = []
        
        # Set seed for reproducibility
        random.seed(self.seed)
        
        for i in range(self.num_samples + self.num_eval_samples):
            question = BATTLESHIP_RULES
            # For battleship, we don't have a single "answer" like Wordle
            # The answer is determined by random ship placement during gameplay
            answer = f"game_{i}"  # Placeholder - actual game state determined during play
            
            if i < self.num_samples:
                dataset_rows.append({
                    "question": question,
                    "answer": answer
                })
            else:
                eval_dataset_rows.append({
                    "question": question,
                    "answer": answer
                })
        
        dataset = Dataset.from_list(dataset_rows)
        eval_dataset = Dataset.from_list(eval_dataset_rows)
        return dataset, eval_dataset 