import random
from typing import Dict, List, Tuple, Any, Optional
import verifiers as vf
from src.battleship_game import BattleshipGame
from src.parser import BattleshipAnswerParser


class BattleshipSingleTurnEnv(vf.SingleTurnEnv):
    """
    Single turn battleship environment for RLVR training.
    
    Each episode:
    1. Generate a random battleship game state
    2. Ask model for next move
    3. Execute move and provide verifiable reward
    """
    
    def __init__(self):
        # Load the same dataset we used for SFT training
        from datasets import load_dataset
        dataset = load_dataset('ljt019/battleship-rlvr-qwen3-dataset', split='train')
        
        # Initialize parent
        super().__init__(
            dataset=dataset,
            system_prompt="You are an expert battleship player. Given a board state, choose the best next move.",
            parser=BattleshipAnswerParser(),
            rubric=None  # We handle rewards in check_answer
        )
        self.game = None
    
    def check_answer(self, problem: str, answer: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the proposed move and return verifiable reward.
        
        Args:
            problem: The game state (unused, we use self.game)
            answer: Model's proposed move (e.g., "d6")
        
        Returns:
            (is_correct, info_dict) where is_correct is based on reward
        """
        if not self.game:
            return False, {"error": "No active game"}
        
        # Parse the answer
        parsed_move = self.parser.parse_answer(answer)
        if not parsed_move:
            return False, {
                "error": "Invalid format",
                "reward": -5,
                "move": answer,
                "reason": "Could not parse move from answer"
            }
        
        # Check if move is valid
        if parsed_move not in self.game.get_valid_moves():
            return False, {
                "error": "Invalid move", 
                "reward": -5,
                "move": parsed_move,
                "reason": "Move not in valid moves list"
            }
        
        # Execute the move
        observation, hit, sunk, game_over, invalid = self.game.step(parsed_move)
        
        # Calculate reward
        reward = 0
        reason = []
        
        if invalid:
            reward = -5
            reason.append("Invalid move")
        elif hit:
            if sunk:
                reward = 10  # Bonus for sinking ship
                reason.append(f"Hit and sunk ship!")
            else:
                reward = 1   # Standard hit
                reason.append("Hit!")
        else:
            reward = -1  # Miss penalty
            reason.append("Miss")
        
        # Win bonus
        if game_over:
            reward += 100
            reason.append("Game won!")
        
        # Consider "correct" if reward > 0 (hits are good)
        is_correct = reward > 0
        
        info = {
            "reward": reward,
            "hit": hit,
            "sunk": sunk, 
            "game_over": game_over,
            "move": parsed_move,
            "reason": "; ".join(reason),
            "board_after": observation
        }
        
        return is_correct, info
    
    def get_parser(self) -> BattleshipAnswerParser:
        """Return the parser for this environment"""
        return self.parser


class BattleshipGameGenerator:
    """Generate diverse battleship game states for training"""
    
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