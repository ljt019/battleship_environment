import random
import logging
from typing import Dict, List, Tuple, Any, Optional
import verifiers as vf
from datasets import load_dataset, Dataset
from src.battleship_game import BattleshipGame
from src.parser import BattleshipAnswerParser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BattleshipMultiTurnEnv(vf.MultiTurnEnv):
    """
    Multi-turn battleship environment following TextArenaEnv pattern.
    
    Key principles:
    1. Deterministic game states based on dataset
    2. Proper state management with reuse
    3. Clean separation of concerns
    """
    
    def __init__(self, max_turns: int = 10):
        # Load dataset and process it properly
        dataset = load_dataset('ljt019/battleship-rlvr-qwen3-dataset', split='train')
        
        # Convert to the format expected by MultiTurnEnv
        def process_example(example):
            question_content = example['prompt'][0]['content']
            answer_content = example['completion'][0]['content']
            return {
                'question': question_content,
                'answer': answer_content  # This will be the optimal move
            }
        
        dataset = dataset.map(process_example)
        
        # Create eval dataset (use a subset)
        eval_dataset = dataset.select(range(min(100, len(dataset))))
        
        super().__init__(
            max_turns=max_turns,
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt="You are an expert battleship player. Given a board state, choose the best next move by responding with coordinates in brackets like [d6].",
            parser=BattleshipAnswerParser(),
            rubric=vf.Rubric(),
        )
    
    def is_completed(self, messages: List[Dict[str, Any]], state: Dict[str, Any], **kwargs) -> bool:
        """Check if the game is completed"""
        if 'is_finished' in state and state['is_finished']:
            # Clean up the game state
            if 'game' in state:
                state.pop('game')
            return True
        return False
    
    def env_response(self, messages: List[Dict[str, Any]], state: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process the model's move and return environment response.
        Following TextArenaEnv pattern exactly.
        """
        # Initialize game if not exists (deterministic based on state['answer'])
        if 'game' not in state:
            # Create a deterministic game based on the optimal answer
            game = self._create_deterministic_game(state.get('answer', '[a1]'))
            state['game'] = game
            state['total_reward'] = 0
            state['moves_made'] = 0
            state['is_finished'] = False
            
            # Return initial prompt (this happens after the question is asked)
            return {
                'role': 'user',
                'content': 'Make your move.'
            }, state
        
        # Game already exists, process the move
        game = state['game']
        
        # Parse the last assistant message
        last_message = messages[-1]['content']
        parsed_move = self.parser.parse_answer(last_message)
        
        if not parsed_move:
            return {
                'role': 'user',
                'content': 'Invalid format. Use [coordinate] like [a1].'
            }, state
        
        # Check if move is valid
        if parsed_move not in game.get_valid_moves():
            return {
                'role': 'user',
                'content': 'Invalid move. Square already revealed.'
            }, state
        
        # Execute the move
        observation, hit, sunk, game_over, invalid = game.step(parsed_move)
        state['moves_made'] = state.get('moves_made', 0) + 1
        
        # Calculate reward
        move_reward = 0
        if invalid:
            move_reward = -5
            feedback = "Invalid."
        elif hit:
            if sunk:
                move_reward = 10
                feedback = "Hit and sunk!"
            else:
                move_reward = 1
                feedback = "Hit!"
        else:
            move_reward = -1
            feedback = "Miss."
        
        # Check if optimal move
        optimal_move = state.get('answer', '').strip('[]')
        if parsed_move == optimal_move:
            move_reward += 5
        
        # Win condition
        if game_over:
            move_reward += 100
            feedback = "Hit and sunk! You win!"
            state['is_finished'] = True
        
        # Max turns reached
        elif state['moves_made'] >= self.max_turns:
            state['is_finished'] = True
            feedback += " Game over - max turns reached."
        
        state['total_reward'] = state.get('total_reward', 0) + move_reward
        state['last_move_reward'] = move_reward
        
        # Return response
        if state['is_finished']:
            env_message = {"role": "user", "content": feedback}
        else:
            env_message = {"role": "user", "content": feedback + " Next move."}
        
        return env_message, state
    
    def _create_deterministic_game(self, optimal_answer: str) -> BattleshipGame:
        """
        Create a deterministic game state based on the optimal answer.
        This ensures the same dataset row always produces the same game.
        """
        # Extract coordinate from answer if it exists
        try:
            # Use the optimal answer as a seed for deterministic game generation
            seed = hash(optimal_answer) % 10000
            random.seed(seed)
            
            # Generate a game with some random moves already made
            game = BattleshipGame()
            
            # Make some random moves to create an interesting mid-game state
            num_moves = random.randint(0, 15)
            for _ in range(num_moves):
                valid_moves = game.get_valid_moves()
                if not valid_moves:
                    break
                move = random.choice(valid_moves)
                game.step(move)
                if game.game_over:
                    break
            
            return game
            
        except Exception as e:
            logger.warning(f"Error creating deterministic game: {e}, using default")
            return BattleshipGame()
    
    def get_parser(self):
        return self.parser 