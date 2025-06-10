import random
from typing import Tuple, List, Dict, Any, Union
import re
from copy import deepcopy

from datasets import Dataset
from openai import OpenAI

from verifiers import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from .battleship_game import BattleshipGame
from .rewards import setup_reward_rubric

BATTLESHIP_SYSTEM_PROMPT = """You are the player in this Battleship game. This is YOUR game - you are making the moves, not helping someone else.
The current board state will be sent to you as user messages, this is not a real user, simply how you receive the board state.
Read the board state and make your next move in the required format."""

BATTLESHIP_RULES = """Your goal is to sink all enemy ships by guessing their locations.

IMPORTANT - Board symbols:
  [?] = unknown squares (you haven't guessed here yet)
  [x] = hit (you found part of a ship)
  [o] = miss (you guessed here but found only water)
  [s] = sunk ship part (entire ship destroyed)

REQUIRED FORMAT: For each move, you must use this exact format:

<guess>[coordinate]</guess>

Example: <guess>[d6]</guess>

Strategy tips:
- After you hit a ship ([x]), try adjacent squares to find the rest of it
- Use probability to target areas likely to contain ships  
- Remember ship sizes: Carrier(5), Battleship(4), Cruiser(3), Submarine(3), Destroyer(2)

Make your next strategic move to find and sink all ships efficiently."""


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
        parser = XMLParser(fields=["guess"], answer_field="guess")
        rubric = Rubric(parser=parser)
        
        # Set up reward functions from the rewards module
        rubric = setup_reward_rubric(rubric)

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
            content = f"Here's the starting board:\n\n{board_render}\n\nYou are the player. Make your first move:"
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
            content = f"{result}\n\n{board_render}\n\nRemember: [x]=hit, [o]=miss, [s]=sunk, [?]=unknown\nYou are the player, the user message is just the game state not an actual user. The former moves were made by you, now make your next move:"
        
        return {"role": "user", "content": content}, state
    
    def _compact_conversation_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove old board states from conversation, keeping only the most recent one"""
        compacted_messages = []
        last_board_message_idx = -1
        
        # Find the last message that contains a board (has multiple brackets indicating a board)
        for i, msg in enumerate(messages):
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                # Check if this looks like a board state (multiple [?], [x], [o], [s] patterns)
                bracket_count = len(re.findall(r'\[[?xos]\]', content))
                if bracket_count >= 10:  # Likely a board state
                    last_board_message_idx = i
        
        # Keep all messages, but replace old board states with just the feedback
        for i, msg in enumerate(messages):
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                bracket_count = len(re.findall(r'\[[?xos]\]', content))
                
                # If this is an old board state (not the most recent), extract just the feedback
                if bracket_count >= 10 and i < last_board_message_idx:
                    # Extract just the move result (HIT, MISS, etc.) from old board messages
                    lines = content.split('\n')
                    feedback_line = lines[0] if lines else content
                    # Keep just the first line (move result) + "Next move:"
                    compacted_content = f"{feedback_line}\nNext move:"
                    compacted_messages.append({"role": "user", "content": compacted_content})
                else:
                    # Keep full message (most recent board or non-board messages)
                    compacted_messages.append(msg)
            else:
                # Keep all assistant messages unchanged
                compacted_messages.append(msg)
        
        return compacted_messages
    
    def rollout(self,
                client: OpenAI,
                model: str,
                prompt: Union[str, List[Dict[str, Any]]],
                answer: str,
                sampling_args: Dict[str, Any] = {},
                **kwargs: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a multi-turn rollout with the environment, with conversation compacting.
        """
        is_completed = False
        state = {'answer': answer}
        assert isinstance(prompt, list)
        messages = deepcopy(prompt) 
        completion = []
        turn = 0
        
        while not is_completed:
            if self.is_completed(messages, state, **kwargs):
                is_completed = True
                break
            
            # Initialize the game if needed - do this BEFORE model response
            if 'game' not in state:
                env_msg, state = self.env_response(messages, state, **kwargs)
                messages.append(env_msg)
                completion.append(env_msg)
            
            # Apply conversation compacting before sending to model
            # compacted_messages = self._compact_conversation_history(messages)
            compacted_messages = messages  # Use full history - Qwen3 has 131k context window
            
            response = self.get_model_response(
                prompt=compacted_messages,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type=self.message_type
            )
            has_error = response.startswith("[ERROR]")
            messages.append({"role": "assistant", "content": response})
            completion.append({"role": "assistant", "content": response})
            turn += 1
            if self.is_completed(messages, state, **kwargs) or turn >= self.max_turns or has_error:
                is_completed = True
            else:
                env_msg, state = self.env_response(messages, state, **kwargs)
                messages.append(env_msg)
                completion.append(env_msg)
        return completion, state
    
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