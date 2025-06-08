import random
from typing import Dict, List, Tuple, Any, Optional
import verifiers as vf
from datasets import load_dataset, Dataset
from src.battleship_game import BattleshipGame
from src.parser import BattleshipAnswerParser


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
            dataset=dataset,  # Each row provides a starting board state and optimal move
            system_prompt="You are an expert battleship player. Given a board state, choose the best next move by responding with coordinates in brackets like [d6].",
            parser=BattleshipAnswerParser(),
            rubric=vf.Rubric()
        )
        self.game_generator = BattleshipGameGenerator()
    
    def is_completed(self, messages, state, **kwargs):
        """Check if the game is completed (won or max turns reached)"""
        if 'game' not in state:
            return False
        
        game = state['game']
        return game.game_over or len(messages) >= self.max_turns * 2  # *2 because model+env messages
    
    def env_response(self, messages, state, **kwargs):
        """
        Process the model's move and return environment response.
        Following TextArenaEnv pattern: use dataset question to initialize game state.
        
        Returns: (message_dict, updated_state)
        """
        if 'game' not in state:
            # Initialize game from the dataset question (board state)
            # state['answer'] contains the optimal move from the dataset
            game = self._parse_board_state_from_question(state.get('answer', '[a1]'))  # Fallback if no answer
            state['game'] = game
            state['total_reward'] = 0
            state['moves_made'] = 0
            state['optimal_move'] = state.get('answer', '[a1]')  # Store the dataset's optimal move
            
            # The question already contains the board state and instructions
            # So we don't need to generate initial content - just indicate the game is ready
            return {
                'role': 'user', 
                'content': "Make your move:"
            }, state
        
        game = state['game']
        
        # Get the last assistant message (model's move)
        last_message = messages[-1]['content']
        
        # Parse and execute the move
        parsed_move = self.parser.parse_answer(last_message)
        if not parsed_move:
            return {
                'role': 'user',
                'content': "Invalid move format! Please respond with coordinates in brackets like [d6]. Try again:"
            }, state
        
        if parsed_move not in game.get_valid_moves():
            return {
                'role': 'user', 
                'content': f"Invalid move {parsed_move}! That square is already revealed. Choose an unrevealed square. Try again:"
            }, state
        
        # Execute the move
        observation, hit, sunk, game_over, invalid = game.step(parsed_move)
        
        # Calculate reward for this move
        move_reward = 0
        response_parts = []
        
        if invalid:
            move_reward = -5
            response_parts.append("Invalid move!")
        elif hit:
            if sunk:
                move_reward = 10
                response_parts.append(f"HIT and SUNK! Great shot at {parsed_move}!")
            else:
                move_reward = 1
                response_parts.append(f"HIT at {parsed_move}!")
        else:
            move_reward = -1
            response_parts.append(f"Miss at {parsed_move}")
        
        # Bonus for using the optimal move from dataset
        if parsed_move == state.get('optimal_move', '').strip('[]'):
            move_reward += 5
            response_parts.append("Great choice!")
        
        # Win bonus
        if game_over:
            move_reward += 100
            response_parts.append("GAME WON! All ships destroyed!")
        
        # Update state
        state['total_reward'] = state.get('total_reward', 0) + move_reward
        state['moves_made'] = state.get('moves_made', 0) + 1
        state['last_move_reward'] = move_reward
        
        # Prepare response - keep it simple to avoid tokenization issues
        if game_over:
            response_content = f"{response_parts[0]} {response_parts[-1]}"
        else:
            response_content = f"{response_parts[0]} {observation} Your move:"
        
        return {
            'role': 'user',
            'content': response_content
        }, state
    
    def _parse_board_state_from_question(self, optimal_move_hint):
        """
        Parse board state from the dataset question text.
        For now, generate a random game since parsing the full board state is complex.
        TODO: Implement proper board state parsing from the text.
        """
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