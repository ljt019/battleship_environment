import random
import logging
from typing import Dict, List, Tuple, Any, Optional
import verifiers as vf
from datasets import load_dataset, Dataset
from src.battleship_game import BattleshipGame
from src.parser import BattleshipAnswerParser

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BattleshipMultiTurnEnv(vf.MultiTurnEnv):
    """
    Multi-turn battleship environment for GRPO training.
    
    Each episode:
    1. Generate a random battleship game state
    2. Model makes moves, environment responds with hit/miss/sunk
    3. Continue until game is won or max turns reached
    4. Provide rewards based on game performance
    """
    
    def __init__(self, max_turns: int = 50):
        # Load our actual battleship dataset - each row will be a different starting scenario
        dataset = load_dataset('ljt019/battleship-rlvr-qwen3-dataset', split='train')
        
        # Process dataset format for verifiers compatibility
        def process_example(example):
            question_content = example['prompt'][0]['content']
            answer_content = example['completion'][0]['content']
            return {
                'question': question_content,
                'answer': answer_content
            }
        
        dataset = dataset.map(process_example)
        
        super().__init__(
            max_turns=max_turns,
            dataset=dataset,
            system_prompt="You are an expert battleship player. Given a board state, choose the best next move by responding with coordinates in brackets like [d6].",
            parser=BattleshipAnswerParser(),
            rubric=vf.Rubric(),
        )
        self.game_generator = BattleshipGameGenerator()
        self.episode_count = 0
    
    def is_completed(self, messages, state, **kwargs):
        """Check if the game is completed (won or max turns reached)"""
        logger.debug(f"is_completed called - messages length: {len(messages)}, state keys: {list(state.keys())}")
        
        if 'game' not in state:
            logger.debug("Game not initialized yet")
            return False
        
        game = state['game']
        completed = game.game_over or len(messages) >= self.max_turns * 2  # *2 because model+env messages
        logger.debug(f"Game completed: {completed} (game_over: {game.game_over}, messages: {len(messages)}, max: {self.max_turns * 2})")
        return completed
    
    def env_response(self, messages, state, **kwargs):
        """
        Process the model's move and return environment response.
        Using ultra-simple, consistent response format to avoid tokenization issues.
        
        Returns: (message_dict, updated_state)
        """
        self.episode_count += 1
        logger.debug(f"\n=== ENV_RESPONSE CALL #{self.episode_count} ===")
        logger.debug(f"Current messages count: {len(messages)}")
        logger.debug(f"State keys: {list(state.keys())}")
        
        # Log the full conversation so far
        for i, msg in enumerate(messages):
            logger.debug(f"Message {i}: role='{msg['role']}', content='{msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}'")
        
        if 'game' not in state:
            logger.debug("INITIALIZING NEW GAME")
            # Initialize game from the dataset question (board state)
            game = self._parse_board_state_from_question(state.get('answer', '[a1]'))  # Fallback if no answer
            state['game'] = game
            state['total_reward'] = 0
            state['moves_made'] = 0
            state['optimal_move'] = state.get('answer', '[a1]')  # Store the dataset's optimal move
            
            logger.debug(f"Game initialized, optimal_move: {state['optimal_move']}")
            
            # Simple consistent initialization
            response_msg = {
                'role': 'user', 
                'content': "Make your move."
            }
            logger.debug(f"RETURNING INIT MESSAGE: {response_msg}")
            return response_msg, state
        
        logger.debug("PROCESSING PLAYER MOVE")
        game = state['game']
        
        # Get the last assistant message (model's move)
        if not messages:
            logger.error("No messages available!")
            return {'role': 'user', 'content': "Error: No messages"}, state
            
        last_message = messages[-1]['content']
        logger.debug(f"Last message content: '{last_message}'")
        
        # Parse and execute the move
        parsed_move = self.parser.parse_answer(last_message)
        logger.debug(f"Parsed move: '{parsed_move}'")
        
        if not parsed_move:
            response_msg = {
                'role': 'user',
                'content': "Invalid format. Use [coordinate] like [a1]."
            }
            logger.debug(f"RETURNING INVALID FORMAT: {response_msg}")
            return response_msg, state
        
        if parsed_move not in game.get_valid_moves():
            logger.debug(f"Invalid move - valid moves: {game.get_valid_moves()[:10]}...")  # Log first 10
            response_msg = {
                'role': 'user', 
                'content': "Invalid move. Square already revealed."
            }
            logger.debug(f"RETURNING INVALID MOVE: {response_msg}")
            return response_msg, state
        
        # Execute the move
        logger.debug(f"Executing move: {parsed_move}")
        observation, hit, sunk, game_over, invalid = game.step(parsed_move)
        logger.debug(f"Move result: hit={hit}, sunk={sunk}, game_over={game_over}, invalid={invalid}")
        
        # Calculate reward for this move
        move_reward = 0
        
        if invalid:
            move_reward = -5
            response = "Invalid."
        elif hit:
            if sunk:
                move_reward = 10
                response = "Hit and sunk!"
            else:
                move_reward = 1
                response = "Hit!"
        else:
            move_reward = -1
            response = "Miss."
        
        # Bonus for using the optimal move from dataset
        if parsed_move == state.get('optimal_move', '').strip('[]'):
            move_reward += 5
            logger.debug(f"Optimal move bonus applied!")
        
        # Win bonus
        if game_over:
            move_reward += 100
            response = "Hit and sunk! You win!"
            logger.debug("GAME WON!")
        
        # Update state
        state['total_reward'] = state.get('total_reward', 0) + move_reward
        state['moves_made'] = state.get('moves_made', 0) + 1
        state['last_move_reward'] = move_reward
        
        logger.debug(f"Move reward: {move_reward}, total_reward: {state['total_reward']}")
        
        # Ultra-simple response format to avoid tokenization issues
        if game_over:
            final_response = response
        else:
            final_response = response + " Next move."
        
        response_msg = {
            'role': 'user',
            'content': final_response
        }
        
        logger.debug(f"RETURNING GAME RESPONSE: {response_msg}")
        logger.debug(f"Updated state keys: {list(state.keys())}")
        logger.debug("=== END ENV_RESPONSE ===\n")
        
        return response_msg, state
    
    def _parse_board_state_from_question(self, optimal_move_hint):
        """
        Parse board state from the dataset question text.
        For now, generate a random game since parsing the full board state is complex.
        TODO: Implement proper board state parsing from the text.
        """
        logger.debug(f"Generating game with optimal_move_hint: {optimal_move_hint}")
        # This is a simplified version - ideally we'd parse the actual board state
        # from the question text, but that's quite complex
        return self.game_generator.generate_mixed_scenario()
    
    def get_parser(self):
        return self.parser


class BattleshipGameGenerator:
    """Generate diverse battleship game states for training"""
    
    @staticmethod
    def generate_mixed_scenario() -> BattleshipGame:
        """Generate a random game scenario with mixed stages"""
        scenario_type = random.choice(['early', 'mid', 'targeting'])
        
        if scenario_type == 'early':
            return BattleshipGameGenerator.generate_early_game()
        elif scenario_type == 'mid':
            return BattleshipGameGenerator.generate_mid_game()
        else:
            return BattleshipGameGenerator.generate_targeting_scenario()
    
    @staticmethod
    def generate_early_game() -> BattleshipGame:
        """Generate early game state (0-10 moves)"""
        game = BattleshipGame()
        num_moves = random.randint(0, 10)
        for _ in range(num_moves):
            if not game.get_valid_moves():
                break
            move = random.choice(game.get_valid_moves())
            game.step(move)
        return game
    
    @staticmethod
    def generate_mid_game() -> BattleshipGame:
        """Generate mid game state (10-25 moves)"""
        game = BattleshipGame()
        num_moves = random.randint(10, 25)
        for _ in range(num_moves):
            if not game.get_valid_moves():
                break
            move = random.choice(game.get_valid_moves())
            game.step(move)
        return game
    
    @staticmethod
    def generate_targeting_scenario() -> BattleshipGame:
        """Generate a game state with hits for targeting practice"""
        game = BattleshipGame()
        
        # Play moves until we get at least one hit
        max_attempts = 50
        attempts = 0
        hits_found = 0
        
        while hits_found == 0 and attempts < max_attempts:
            if not game.get_valid_moves():
                break
            move = random.choice(game.get_valid_moves())
            _, hit, _, _, _ = game.step(move)
            if hit:
                hits_found += 1
            attempts += 1
        
        return game 