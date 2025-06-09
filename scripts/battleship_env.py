import random
from typing import Tuple, List, Dict, Any, Union
import re
from copy import deepcopy

from datasets import Dataset
from openai import OpenAI

from verifiers import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from battleship_game import BattleshipGame

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
            for msg in assistant_messages:
                content = msg.get('content', '')
                # Check for proper <guess>[coordinate]</guess> format
                if re.search(r'<guess>\[[a-j][0-9]+\]</guess>', content, re.IGNORECASE):
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
        
        rubric.add_reward_func(win_reward_func, weight=2.0)        # Main objective
        rubric.add_reward_func(efficiency_reward_func, weight=0.5) # Encourage speed (reduced since always active)
        rubric.add_reward_func(hit_reward_func, weight=0.5)        # Reward progress  
        rubric.add_reward_func(sink_reward_func, weight=1.0)       # Reward major progress
        rubric.add_reward_func(format_reward_func, weight=0.5)     # Ensure proper format
        rubric.add_reward_func(valid_move_reward_func, weight=1.0) # Penalize invalid moves

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
        
        # Initialize the game and show the initial board state before first model response
        if 'game' not in state:
            env_msg, state = self.env_response(messages, state, **kwargs)
            messages.append(env_msg)
            completion.append(env_msg)
        
        while not is_completed:
            if self.is_completed(messages, state, **kwargs):
                is_completed = True
                break
            
            # Apply conversation compacting before sending to model
            compacted_messages = self._compact_conversation_history(messages)
            
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